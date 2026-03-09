"""QEC maintenance Gymnasium environment and supporting managers."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pymatching
import stim
from gymnasium import spaces


def load_config(path=None):
    """Load config from JSON. If path is None, use config.json in package directory."""
    if path is None:
        path = Path(__file__).parent / "config.json"
    with open(path) as f:
        return json.load(f)


class NoiseManager:
    def __init__(self, num_data, num_ancilla, config):
        self.num_data = num_data
        self.num_ancilla = num_ancilla
        self.cfg = config
        # Noise changing rates, the smaller the tc, the faster the noise changes
        self.theta_anc = 1.0 / config["tc_anc"]
        self.theta_data = 1.0 / config["tc_data"]
        # Noise amplitudes
        self.sigma_anc = config["std_panc"] * np.sqrt(2 * self.theta_anc)
        self.sigma_px = config["std_px"] * np.sqrt(2 * self.theta_data)
        self.sigma_pz = config["std_pz"] * np.sqrt(2 * self.theta_data)
        # Initial error rates
        self.p_anc = np.ones(num_ancilla) * config["panc0"]
        self.p_px = np.ones(num_data) * config["px0"]
        self.p_pz = np.ones(num_data) * config["pz0"]

    def step_drift(self):
        noise = np.random.normal(0, 1, self.num_data)
        self.p_px = np.clip(
            self.p_px + -self.theta_data * self.p_px  + self.sigma_px * noise,
            1e-9, 0.4,
        )
        noise = np.random.normal(0, 1, self.num_data)
        self.p_pz = np.clip(
            self.p_pz + -self.theta_data * self.p_pz  + self.sigma_pz * noise,
            1e-9, 0.4,
        )
        noise = np.random.normal(0, 1, self.num_ancilla)
        self.p_anc = np.clip(
            self.p_anc + -self.theta_anc * self.p_anc + self.sigma_anc * noise,
            1e-9, 0.4,
        )

    def reset_qubit(self, global_idx):
        if global_idx >= self.num_data:
            rel = global_idx - self.num_data
            self.p_anc[rel] = self.mu_anc[rel]

    def get_px(self, idx):
        return self.p_px[idx]

    def get_pz(self, idx):
        return self.p_pz[idx]

    def get_p_anc(self, abs_idx):
        return self.p_anc[abs_idx - self.num_data]


class AncillaManager:
    # This is not relevant if use_hook_errors = False
    def __init__(self, num_ancillas, cooldown):
        self.timers = np.zeros(num_ancillas, dtype=int)
        self.cooldown = cooldown

    def request_fix(self, rel_idx):
        if self.timers[rel_idx] == 0:
            self.timers[rel_idx] = self.cooldown
            return True
        return False

    def step(self):
        fixed = []
        for i in range(len(self.timers)):
            if self.timers[i] > 0:
                self.timers[i] -= 1
                if self.timers[i] == 0:
                    fixed.append(i)
        return fixed

    def is_offline(self, rel_idx):
        return self.timers[rel_idx] > 0


class QECMaintenanceEnv(gym.Env):
    def __init__(self, config=None, render_mode=None):
        if config is None:
            config = load_config()
        self.cfg = config
        self.distance = config["distance"]
        self.render_mode = render_mode
        # How do we want to display the environment?
        self.debug = self.render_mode == "human"

        # Build the lattice
        self.lattice = self._build_rotated_surface_lattice(self.distance)
        
        self.num_data = self.distance * self.distance   # Number of data qubits
        self.num_ancilla = len(self.lattice["ancilla_pos"])  # Number of ancilla qubits

        self.noise_manager = NoiseManager(self.num_data, self.num_ancilla, config)  # Manage the noise on the data and ancilla qubits
        self.ancilla_manager = AncillaManager(self.num_ancilla, config["cooldown_time"])  # Manage the cooldown time of the ancilla qubits 
        self.sim = stim.TableauSimulator()  # Simulate the logical operations on the data and ancilla qubits

        self.qubit_errors = np.zeros((self.num_data, 2), dtype=int)
        self.syndrome_buffer = np.zeros(
            (config["rounds_history"], self.num_ancilla), dtype=int
        )
        self.last_syndrome = np.zeros(self.num_ancilla, dtype=int)
        self.latest_raw_syndrome = np.zeros(self.num_ancilla, dtype=int)
        self.reference_syndrome = None

        self.logical_z_indices = [
            self.lattice["data_lookup"][(i, 0)] for i in range(self.distance)
        ]
        self.logical_x_indices = [
            self.lattice["data_lookup"][(0, j)] for j in range(self.distance)
        ]

        obs_size = (config["rounds_history"] * self.num_ancilla) + self.num_ancilla
        self.observation_space = spaces.MultiBinary(obs_size)
        self.use_hook_errors = config.get("use_hook_errors", False)
        if self.use_hook_errors:
            self.action_space = spaces.MultiDiscrete([self.num_data + 2, self.num_ancilla + 1])
        else:
            self.action_space = spaces.Discrete(self.num_data + 2)

        self.step_count = 0
        self.check_period = config.get("check_period", 10)
        self.max_steps = config.get("max_steps", 1000)
        self._setup_pymatching()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.qubit_errors.fill(0)
        self.syndrome_buffer.fill(0)
        self.last_syndrome.fill(0)
        self.step_count = 0

        if self.render_mode == "human":
            print("\n" + "=" * 80)
            print("RESET - NEW EPISODE (Unified Protection)")
            print("=" * 80)

        self._run_cycle(inject_noise=False)
        raw_meas = np.array(
            self.sim.current_measurement_record()[-self.num_ancilla :], dtype=int
        )
        self.reference_syndrome = raw_meas.copy()
        self.latest_raw_syndrome.fill(0)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        if self.use_hook_errors:
            corr_act, fix_act = action
        else:
            corr_act = int(np.asarray(action).flat[0])
            fix_act = 0

        if self.debug:
            print(f"\n--- STEP {self.step_count} ---")

        maintenance_cost = 0.0
        if fix_act > 0 and self.ancilla_manager.request_fix(fix_act - 1):
            maintenance_cost = 0.5

        fixed_list = self.ancilla_manager.step()
        for idx in fixed_list:
            self.noise_manager.reset_qubit(self.num_data + idx)
        self.noise_manager.step_drift()

        if corr_act < self.num_data:
            self.sim.x(corr_act)
            self.qubit_errors[corr_act, 0] ^= 1
            if self.debug:
                print(f"Action: Corrected X on Qubit {corr_act}")

        self._apply_idle_noise()
        self._run_cycle(inject_noise=True)

        raw_meas = np.array(
            self.sim.current_measurement_record()[-self.num_ancilla :], dtype=int
        )
        self.latest_raw_syndrome = (raw_meas ^ self.reference_syndrome).astype(int)
        diff_syndrome = (raw_meas ^ self.last_syndrome).astype(int)
        self.last_syndrome = raw_meas.copy()

        for i in range(self.num_ancilla):
            if self.ancilla_manager.is_offline(i):
                diff_syndrome[i] = 0

        self.syndrome_buffer = np.roll(self.syndrome_buffer, -1, axis=0)
        self.syndrome_buffer[-1] = diff_syndrome

        reward = 1.0 - maintenance_cost
        terminated = False

        if self.step_count % self.check_period == 0:
            if self._evaluate_final_state():
                reward = -1000.0
                terminated = True
                if self.debug:
                    print(f"*** LOGICAL FAILURE DETECTED at step {self.step_count} ***")
            elif self.debug:
                print(f"--- Passed logical check at step {self.step_count} ---")

        if not terminated and self.step_count >= self.max_steps:
            terminated = True
            reward = 1000.0
            if self.debug:
                print("*** MAX STEPS REACHED: EPISODE SUCCESS ***")

        if corr_act == self.num_data + 1:
            terminated = True
            if self._evaluate_final_state():
                reward = -1000.0
            else:
                reward = 1000.0

        return self._get_obs(), reward, terminated, False, {}

    def _evaluate_final_state(self):
        """Returns True if logical failure (decoder wrong), False if state is OK."""
        x_errs = self.qubit_errors[:, 0]
        z_errs = self.qubit_errors[:, 1]
        syndrome_x = (self.Hx @ x_errs) % 2
        syndrome_z = (self.Hz @ z_errs) % 2
        pred_logical_x = self.matcher_x.decode(syndrome_x)[0]
        pred_logical_z = self.matcher_z.decode(syndrome_z)[0]
        actual_logical_x = np.sum(x_errs[self.logical_z_indices]) % 2
        actual_logical_z = np.sum(z_errs[self.logical_x_indices]) % 2
        failed_x = pred_logical_x != actual_logical_x
        failed_z = pred_logical_z != actual_logical_z
        return failed_x or failed_z

    def _apply_idle_noise(self):
        for i in range(self.num_data):
            if np.random.random() < self.noise_manager.get_px(i):
                self.sim.x(i)
                self.qubit_errors[i, 0] ^= 1
                if self.debug:
                    print(f"  Idle Noise: X-error on data {i}")
            if np.random.random() < self.noise_manager.get_pz(i):
                self.sim.z(i)
                self.qubit_errors[i, 1] ^= 1
                if self.debug:
                    print(f"  Idle Noise: Z-error on data {i}")

    def _run_cycle(self, inject_noise=True):
        anc_indices = [self.num_data + i for i in range(self.num_ancilla)]
        self.sim.do(stim.Circuit("R " + " ".join(str(i) for i in anc_indices)))

        for idx in range(self.num_ancilla):
            anc_qubit = self.num_data + idx
            if self.lattice["ancilla_types"][idx] == "X":
                self.sim.h(anc_qubit)

        max_neighbors = 4
        for step in range(max_neighbors):
            for idx in range(self.num_ancilla):
                anc_qubit = self.num_data + idx
                type_ = self.lattice["ancilla_types"][idx]
                neighbors = self.lattice["neighbors"][idx]

                if step < len(neighbors):
                    data_qubit = neighbors[step]
                    if data_qubit is None:
                        continue
                    if type_ == "X":
                        self.sim.cx(anc_qubit, data_qubit)
                    else:
                        self.sim.cx(data_qubit, anc_qubit)
                    if inject_noise and self.cfg.get("use_hook_errors", False):
                        self._inject_single_hook(anc_qubit, neighbors, step, type_)

        for idx in range(self.num_ancilla):
            anc_qubit = self.num_data + idx
            if self.lattice["ancilla_types"][idx] == "X":
                self.sim.h(anc_qubit)
            if inject_noise and np.random.random() < self.cfg["readout_error"]:
                self.sim.x(anc_qubit)
        self.sim.measure_many(*anc_indices)

    def _inject_single_hook(self, anc_qubit, neighbors, current_step_idx, type_):
        p_anc = self.noise_manager.get_p_anc(anc_qubit)
        if np.random.random() >= (p_anc / 4.0):
            return
        future_neighbors = [n for n in neighbors[current_step_idx + 1 :] if n is not None]
        if type_ == "X":
            self.sim.x(anc_qubit)
            for fn in future_neighbors:
                self.qubit_errors[fn, 0] ^= 1
            if self.debug and future_neighbors:
                print(f"  [Hook Error] X-Ancilla {anc_qubit} -> neighbors {future_neighbors}")
        else:
            self.sim.z(anc_qubit)
            for fn in future_neighbors:
                self.qubit_errors[fn, 1] ^= 1
            if self.debug and future_neighbors:
                print(f"  [Hook Error] Z-Ancilla {anc_qubit} -> neighbors {future_neighbors}")

    def _build_rotated_surface_lattice(self, d):
        lat = {"ancilla_pos": [], "ancilla_types": [], "neighbors": [], "data_lookup": {}}
        idx = 0
        for r in range(d):
            for c in range(d):
                lat["data_lookup"][(r, c)] = idx
                idx += 1

        for r in range(-1, d):
            for c in range(-1, d):
                sum_rc = r + c
                stab_type = "Z" if sum_rc % 2 == 0 else "X"
                if stab_type == "X":
                    potential_ns = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                else:
                    potential_ns = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                valid_ns = []
                valid_count = 0
                for pr, pc in potential_ns:
                    if (pr, pc) in lat["data_lookup"]:
                        valid_ns.append(lat["data_lookup"][(pr, pc)])
                        valid_count += 1
                    else:
                        valid_ns.append(None)
                if valid_count < 2:
                    continue
                actual_ns = [n for n in valid_ns if n is not None]
                is_horizontal_pair = (
                    len(actual_ns) == 2 and abs(actual_ns[0] - actual_ns[1]) == 1
                )
                is_vertical_pair = (
                    len(actual_ns) == 2 and abs(actual_ns[0] - actual_ns[1]) == d
                )
                if stab_type == "X" and is_horizontal_pair:
                    continue
                if stab_type == "Z" and is_vertical_pair:
                    continue
                lat["ancilla_pos"].append((r + 0.5, c + 0.5))
                lat["ancilla_types"].append(stab_type)
                lat["neighbors"].append(valid_ns)
        return lat

    def _get_obs(self):
        return np.concatenate(
            [
                self.syndrome_buffer.flatten(),
                (self.ancilla_manager.timers > 0).astype(np.uint8),
            ]
        ).astype(np.uint8)

    def _setup_pymatching(self):
        z_stabs = [i for i, t in enumerate(self.lattice["ancilla_types"]) if t == "Z"]
        x_stabs = [i for i, t in enumerate(self.lattice["ancilla_types"]) if t == "X"]
        self.Hx = np.zeros((len(z_stabs), self.num_data), dtype=np.uint8)
        for i, stab_idx in enumerate(z_stabs):
            for qubit_idx in self.lattice["neighbors"][stab_idx]:
                if qubit_idx is not None:
                    self.Hx[i, qubit_idx] = 1
        self.Hz = np.zeros((len(x_stabs), self.num_data), dtype=np.uint8)
        for i, stab_idx in enumerate(x_stabs):
            for qubit_idx in self.lattice["neighbors"][stab_idx]:
                if qubit_idx is not None:
                    self.Hz[i, qubit_idx] = 1
        Lx = np.zeros((1, self.num_data), dtype=np.uint8)
        Lx[0, self.logical_z_indices] = 1
        Lz = np.zeros((1, self.num_data), dtype=np.uint8)
        Lz[0, self.logical_x_indices] = 1
        self.matcher_x = pymatching.Matching.from_check_matrix(self.Hx, faults_matrix=Lx)
        self.matcher_z = pymatching.Matching.from_check_matrix(self.Hz, faults_matrix=Lz)

    def render(self, step=0, action=None):
        from render import render_qec
        render_qec(self, step=step, action=action)
