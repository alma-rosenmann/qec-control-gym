import gymnasium as gym
import stim
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- CONFIGURATION ---
CONFIG = {
    "distance": 5,           # Code Distance (d=3, 5, 7)
    
    # Noise Baselines
    "panc0": 0.00,           
    "px0": 0.001,            
    "pz0": 0.001,            
    "readout_error": 0.00,   
    
    # Drift Parameters
    "std_panc": 0.000,
    "std_px": 0.0005,
    "std_pz": 0.0005,
    "tc_anc": 20.0,
    "tc_data": 50.0,
    
    "cooldown_time": 5,
    "rounds_history": 5
}

class NoiseManager:
    def __init__(self, num_data, num_ancilla, config):
        self.num_data = num_data
        self.num_ancilla = num_ancilla
        self.cfg = config
        self.mu_anc = np.ones(num_ancilla) * config['panc0']
        self.mu_px  = np.ones(num_data) * config['px0']
        self.mu_pz  = np.ones(num_data) * config['pz0']
        self.p_anc = self.mu_anc.copy()
        self.p_px  = self.mu_px.copy()
        self.p_pz  = self.mu_pz.copy()
        self.theta_anc = 1.0 / config['tc_anc']
        self.sigma_anc = config['std_panc'] * np.sqrt(2 * self.theta_anc)
        self.theta_data = 1.0 / config['tc_data']
        self.sigma_px = config['std_px'] * np.sqrt(2 * self.theta_data)
        self.sigma_pz = config['std_pz'] * np.sqrt(2 * self.theta_data)

    def step_drift(self):
        noise = np.random.normal(0, 1, self.num_data)
        self.p_px = np.clip(self.p_px + -self.theta_data * (self.p_px - self.mu_px) + self.sigma_px * noise, 1e-9, 0.4)
        noise = np.random.normal(0, 1, self.num_data)
        self.p_pz = np.clip(self.p_pz + -self.theta_data * (self.p_pz - self.mu_pz) + self.sigma_pz * noise, 1e-9, 0.4)
        noise = np.random.normal(0, 1, self.num_ancilla)
        self.p_anc = np.clip(self.p_anc + -self.theta_anc * (self.p_anc - self.mu_anc) + self.sigma_anc * noise, 1e-9, 0.4)

    def reset_qubit(self, global_idx):
        if global_idx >= self.num_data:
            rel = global_idx - self.num_data
            self.p_anc[rel] = self.mu_anc[rel]

    def get_px(self, idx): return self.p_px[idx]
    def get_pz(self, idx): return self.p_pz[idx]
    def get_p_anc(self, abs_idx): return self.p_anc[abs_idx - self.num_data]


class AncillaManager:
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
                if self.timers[i] == 0: fixed.append(i)
        return fixed

    def is_offline(self, rel_idx): return self.timers[rel_idx] > 0


class QECMaintenanceEnv(gym.Env):
    def __init__(self, config=CONFIG, debug=True):
        self.cfg = config
        self.distance = config['distance']
        self.debug = debug
        
        self.lattice = self._build_rotated_surface_lattice(self.distance)
        self.num_data = self.distance * self.distance
        self.num_ancilla = len(self.lattice['ancilla_pos'])
        
        self.noise_manager = NoiseManager(self.num_data, self.num_ancilla, config)
        self.ancilla_manager = AncillaManager(self.num_ancilla, config['cooldown_time'])
        self.sim = stim.TableauSimulator()
        
        self.qubit_errors = np.zeros((self.num_data, 2), dtype=int)
        self.syndrome_buffer = np.zeros((config['rounds_history'], self.num_ancilla), dtype=int)
        self.last_syndrome = np.zeros(self.num_ancilla, dtype=int)
        self.latest_raw_syndrome = np.zeros(self.num_ancilla, dtype=int)
        self.reference_syndrome = None
        
        # Pre-calculate Logical Chains
        self.logical_z_indices = [self.lattice['data_lookup'][(i, 0)] for i in range(self.distance)]
        self.logical_x_indices = [self.lattice['data_lookup'][(0, j)] for j in range(self.distance)]

        obs_size = (config['rounds_history'] * self.num_ancilla) + self.num_ancilla
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.num_data + 2, self.num_ancilla + 1])
        
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.qubit_errors.fill(0)
        self.syndrome_buffer.fill(0)
        self.last_syndrome.fill(0)
        self.step_count = 0
        
        # Physical state: |0>, tracking both X and Z error chains
        if self.debug:
            print("\n" + "="*80)
            print("RESET - NEW EPISODE (Unified Protection)")
            print("="*80)
            print("  Physical State: |0>")
            print("  Logical Goals: Protect against BOTH X-chains (Bit Flip) and Z-chains (Phase Flip)")
        
        # Burn-in
        self._run_cycle(inject_noise=False)
        
        raw_meas = np.array(self.sim.current_measurement_record()[-self.num_ancilla:], dtype=int)
        self.reference_syndrome = raw_meas.copy()
        self.latest_raw_syndrome.fill(0)
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        corr_act, fix_act = action
        
        if self.debug:
            print(f"\n--- STEP {self.step_count} ---")
        
        # 1. Maintenance
        maintenance_cost = 0.0
        if fix_act > 0 and self.ancilla_manager.request_fix(fix_act-1):
            maintenance_cost = 0.5
        
        fixed_list = self.ancilla_manager.step()
        for idx in fixed_list: self.noise_manager.reset_qubit(self.num_data + idx)
        self.noise_manager.step_drift()
        
        # 2. Correction
        if corr_act < self.num_data:
            self.sim.x(corr_act)
            self.qubit_errors[corr_act, 0] ^= 1 
            if self.debug: print(f"Action: Corrected X on Qubit {corr_act}")
        
        # 3. Physics
        self._apply_idle_noise()
        self._run_cycle(inject_noise=True)
        
        raw_meas = np.array(self.sim.current_measurement_record()[-self.num_ancilla:], dtype=int)
        self.latest_raw_syndrome = (raw_meas ^ self.reference_syndrome).astype(int)
        
        diff_syndrome = (raw_meas ^ self.last_syndrome).astype(int)
        self.last_syndrome = raw_meas.copy()
        
        for i in range(self.num_ancilla):
            if self.ancilla_manager.is_offline(i): diff_syndrome[i] = 0
            
        self.syndrome_buffer = np.roll(self.syndrome_buffer, -1, axis=0)
        self.syndrome_buffer[-1] = diff_syndrome
        
        # 4. Reward & Termination
        reward = 1.0 - maintenance_cost
        terminated = False
        
        if self._check_logical_failure():
            reward = -1000.0
            terminated = True
            if self.debug: print(f"*** LOGICAL FAILURE DETECTED ***")
        
        if corr_act == self.num_data + 1:  # Abort action
            terminated = True
            if reward > 0: reward = -10.0
            
        return self._get_obs(), reward, terminated, False, {}

    def _check_logical_failure(self):
        # Check Vertical Chain (X-Errors)
        z_parity = 0
        for idx in self.logical_z_indices: z_parity += self.qubit_errors[idx, 0] 
        
        # Check Horizontal Chain (Z-Errors)
        x_parity = 0
        for idx in self.logical_x_indices: x_parity += self.qubit_errors[idx, 1]

        # Fail if EITHER parity is odd
        return (z_parity % 2 != 0) or (x_parity % 2 != 0)

    def _apply_idle_noise(self):
        for i in range(self.num_data):
            # X-Error
            if np.random.random() < self.noise_manager.get_px(i):
                self.sim.x(i)
                self.qubit_errors[i, 0] ^= 1
                if self.debug: print(f"  Idle Noise: X-error on data {i}")
            # Z-Error
            if np.random.random() < self.noise_manager.get_pz(i):
                self.sim.z(i)
                self.qubit_errors[i, 1] ^= 1
                if self.debug: print(f"  Idle Noise: Z-error on data {i}")

    def _run_cycle(self, inject_noise=True):
        """
        Runs the cycle in 4 SYNCHRONIZED phases to mimic parallel execution.
        Time 1: All Ancillas interact with Neighbor 1
        Time 2: All Ancillas interact with Neighbor 2
        ...
        """
        # 1. Reset Ancillas
        anc_indices = [self.num_data + i for i in range(self.num_ancilla)]
        self.sim.do(stim.Circuit("R " + " ".join(str(i) for i in anc_indices)))
        
        # 2. PREPARE (Hadamard for X-Ancillas)
        for idx in range(self.num_ancilla):
            anc_qubit = self.num_data + idx
            if self.lattice['ancilla_types'][idx] == 'X':
                self.sim.h(anc_qubit)

        # 3. Synchronized interaction phases (time-sliced execution)
        # Standard surface code has 4 interaction steps (North, West, East, South)
        max_neighbors = 4 
        
        for step in range(max_neighbors):
            # Each ancilla interacts with its step-th neighbor in this time slice
            for idx in range(self.num_ancilla):
                anc_qubit = self.num_data + idx
                type_ = self.lattice['ancilla_types'][idx]
                neighbors = self.lattice['neighbors'][idx]
                
                # Skip if ancilla has no neighbor at this step
                if step < len(neighbors):
                    data_qubit = neighbors[step]
                    
                    # Apply CNOT
                    if type_ == 'X': 
                        self.sim.cx(anc_qubit, data_qubit) # Control->Target
                    else:            
                        self.sim.cx(data_qubit, anc_qubit) # Target<-Control
                    
                    # Inject hook error for this interaction
                    if inject_noise:
                        self._inject_single_hook(anc_qubit, neighbors, step, type_)

        # 4. Measure (Hadamard for X-ancillas, then measure all)
        for idx in range(self.num_ancilla):
            anc_qubit = self.num_data + idx
            if self.lattice['ancilla_types'][idx] == 'X':
                self.sim.h(anc_qubit)
            
            # Readout error (applied before measurement)
            if inject_noise and np.random.random() < self.cfg['readout_error']:
                self.sim.x(anc_qubit)

        self.sim.measure_many(*anc_indices)

    def _inject_single_hook(self, anc_qubit, neighbors, current_step_idx, type_):
        """
        Injects an error that propagates only to neighbors visited after the current step.
        """
        p_anc = self.noise_manager.get_p_anc(anc_qubit)
        
        # Probability is per-step (divide total ancilla error by 4 steps)
        if np.random.random() < (p_anc / 4.0):
            # Error propagates to neighbors not yet visited in this cycle
            future_neighbors = neighbors[current_step_idx+1:]
            
            if type_ == 'X':
                self.sim.x(anc_qubit) 
                for fn in future_neighbors: 
                    self.qubit_errors[fn, 0] ^= 1
                
                if self.debug:
                    print(f"  [Hook Error] X-Ancilla {anc_qubit} failed at step {current_step_idx}!")
                    if future_neighbors:
                        print(f"    -> Injected X-errors onto future neighbors: {future_neighbors}")
                    else:
                        print(f"    -> Late fault (self-blinding only, no data errors)")

            else: # Z-stabilizer
                self.sim.z(anc_qubit)
                for fn in future_neighbors: 
                    self.qubit_errors[fn, 1] ^= 1
                
                if self.debug:
                    print(f"  [Hook Error] Z-Ancilla {anc_qubit} failed at step {current_step_idx}!")
                    if future_neighbors:
                        print(f"    -> Injected Z-errors onto future neighbors: {future_neighbors}")
                    else:
                        print(f"    -> Late fault (self-blinding only, no data errors)")


    def _build_rotated_surface_lattice(self, d):
        lat = {'ancilla_pos': [], 'ancilla_types': [], 'neighbors': [], 'data_lookup': {}}
        idx = 0
        for r in range(d):
            for c in range(d):
                lat['data_lookup'][(r,c)] = idx
                idx += 1
        
        for r in range(-1, d):
            for c in range(-1, d):
                potential_ns = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
                valid_ns = []
                for pr, pc in potential_ns:
                    if (pr, pc) in lat['data_lookup']:
                        valid_ns.append(lat['data_lookup'][(pr, pc)])
                
                if len(valid_ns) < 2: continue
                
                sum_rc = r + c
                if sum_rc % 2 == 0: stab_type = 'Z'
                else:               stab_type = 'X'
                
                is_horizontal_pair = (len(valid_ns) == 2 and abs(valid_ns[0] - valid_ns[1]) == 1)
                is_vertical_pair   = (len(valid_ns) == 2 and abs(valid_ns[0] - valid_ns[1]) == d)
                
                if stab_type == 'X' and is_horizontal_pair: continue 
                if stab_type == 'Z' and is_vertical_pair: continue 

                lat['ancilla_pos'].append((r + 0.5, c + 0.5))
                lat['ancilla_types'].append(stab_type)
                lat['neighbors'].append(valid_ns)
        return lat

    def _get_obs(self):
        return np.concatenate([self.syndrome_buffer.flatten(), 
                               (self.ancilla_manager.timers > 0).astype(float)])

    def render(self, step=0, action=None):
        d = self.distance
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        # Both X and Z errors are fully visible (both can cause logical failure)
        red_alpha = 1.0
        blue_alpha = 1.0
        
        # Draw Connections
        for idx, neighbors in enumerate(self.lattice['neighbors']):
            anc_r, anc_c = self.lattice['ancilla_pos'][idx]
            for d_idx in neighbors:
                for (dr, dc), lookup_idx in self.lattice['data_lookup'].items():
                    if lookup_idx == d_idx:
                        ax.plot([anc_c, dc], [anc_r, dr], color='gray', alpha=0.15, zorder=0)
                        break

        # Draw Data Qubits
        for (r, c), idx in self.lattice['data_lookup'].items():
            px = self.noise_manager.get_px(idx)
            intensity = np.clip(px / 0.05, 0, 1)
            circle = mpatches.Circle((c, r), 0.15, facecolor=(1, 1-intensity*0.3, 1-intensity*0.3), edgecolor='black', zorder=10)
            ax.add_patch(circle)
            ax.text(c, r, str(idx), ha='center', va='center', fontsize=8, zorder=11)
            
            if self.qubit_errors[idx, 0]: 
                 ax.add_patch(mpatches.Circle((c-0.2, r), 0.06, color='red', alpha=red_alpha, zorder=12))
            if self.qubit_errors[idx, 1]: 
                 ax.add_patch(mpatches.Circle((c+0.2, r), 0.06, color='blue', alpha=blue_alpha, zorder=12))

        # Draw Ancillas
        for idx, (anc_r, anc_c) in enumerate(self.lattice['ancilla_pos']):
            type_ = self.lattice['ancilla_types'][idx]
            is_violated = (self.latest_raw_syndrome[idx] == 1)
            is_offline = self.ancilla_manager.is_offline(idx)
            
            if is_offline: fill_color = 'lightgray'
            else:          fill_color = 'aliceblue' if type_ == 'X' else 'mistyrose'
            
            edge_color = 'blue' if type_ == 'X' else 'red'
            lw = 1
            if is_violated:
                edge_color = 'orange'; lw = 3
            if action and action[1] == idx + 1: 
                edge_color = 'gold'; lw = 3

            rect = mpatches.Rectangle((anc_c - 0.2, anc_r - 0.2), 0.4, 0.4, 
                                      facecolor=fill_color, edgecolor=edge_color, linewidth=lw, zorder=5)
            ax.add_patch(rect)
            ax.text(anc_c, anc_r, str(idx + self.num_data), ha='center', va='center', fontsize=7, color='black', zorder=6)

        ax.set_xlim(-0.8, d - 0.2)
        ax.set_ylim(d - 0.2, -0.8) 
        
        ax.set_title(f"Quantum Memory (Dual Protection)\nStep {step}", fontweight='bold')
        
        handles = [
            mpatches.Patch(edgecolor='orange', linewidth=3, facecolor='none', label='Violated Stab'),
            mpatches.Circle((0,0), color='red', alpha=1.0, label='Bit Flip Chain (Deadly)'),
            mpatches.Circle((0,0), color='blue', alpha=1.0, label='Phase Flip Chain (Deadly)')
        ]
        ax.legend(handles=handles, loc='upper right')
        plt.show()

# --- DEMO ---
if __name__ == "__main__":
    TEST_CONFIG = CONFIG.copy()
    # High Noise Demo
    TEST_CONFIG['panc0'] = 0.1
    TEST_CONFIG['px0'] = 0.0
    TEST_CONFIG['pz0'] = 0.0
    TEST_CONFIG['readout_error'] = 0.0
    TEST_CONFIG['std_panc'] = 0.1
    TEST_CONFIG['std_px'] = 0.0
    TEST_CONFIG['std_pz'] = 0.0

    env = QECMaintenanceEnv(TEST_CONFIG, debug=True)
    
    print("Starting Unified Protection Demo...")
    
    for ep in range(3):
        obs, _ = env.reset()
        env.render(step=0)
    
        for i in range(5):
            obs, reward, done, _, _ = env.step([env.num_data, 0])
            print(f"  Step {i+1} Reward: {reward}")
            env.render(step=i+1)
            
            if done:
                print(f"  FAILED at step {i+1}!")
                break