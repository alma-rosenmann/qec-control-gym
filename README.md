# Quantum Error Correction Maintenance Environment

A Gymnasium environment for reinforcement learning on quantum error correction with non-stationary noise and sensor maintenance. This environment simulates a **rotated surface code** where the agent must simultaneously correct errors and manage the health of measurement ancillas.

## Project layout

- **`config.json`** — Default environment parameters (distance, noise, drift, cooldown, etc.).
- **`env.py`** — `NoiseManager`, `AncillaManager`, `QECMaintenanceEnv`, and `load_config()`.
- **`render.py`** — Lattice visualization (`render_qec(env, step, action)`).
- **`qec_control.py`** — Entry point: loads `config.json` into `CONFIG`, re-exports `QECMaintenanceEnv` and `load_config`; run as `python qec_control.py` for the demo.

## Overview

This environment models a realistic quantum error correction scenario where:
- **Noise drifts over time** (Ornstein-Uhlenbeck process)
- **Sensors (ancillas) can be taken offline** for recalibration (temporarily reducing observability)
- **Hook errors** can occur mid-cycle, propagating to multiple data qubits
- The agent must protect against **both bit-flip (X) and phase-flip (Z) errors** simultaneously
- The agent must balance **error correction** (applying Pauli-X to data qubits) with **sensor maintenance** (taking ancillas offline to reset their error rates)

## Architecture

### Components

#### 1. `NoiseManager`
Manages drifting error rates for all qubits using an Ornstein-Uhlenbeck process:
- Tracks individual error rates for data qubits (X and Z) and ancilla qubits
- Applies drift each step: `p_new = p_old - θ(p_old - μ) + σ·noise`
- Can reset ancilla error rates to baseline when recalibrated

#### 2. `AncillaManager`
Manages the maintenance state machine:
- Tracks cooldown timers for each ancilla
- When an ancilla is fixed, it goes offline for `cooldown_time` steps
- Returns list of ancillas that just finished recalibration

#### 3. `QECMaintenanceEnv`
The main Gymnasium environment implementing a rotated surface code:
- **Distance**: Configurable code distance (default: 5)
- **Lattice**: Rotated surface code with X and Z stabilizers in checkerboard pattern
- **Dual Protection**: Protects against both X-errors (bit flips) and Z-errors (phase flips)
- **Logical Chains**: 
  - Vertical chain (Logical Z): Detects X-errors
  - Horizontal chain (Logical X): Detects Z-errors

### Observation Space

A flat float array of size `(rounds_history × num_ancilla) + num_ancilla`:
1. **Syndrome History**: Last `rounds_history` rounds of differential syndrome measurements (flattened)
2. **Ancilla Status**: Binary flags indicating which ancillas are offline (1 = offline, 0 = online)

### Action Space

The action space is `MultiDiscrete([num_data + 2, num_ancilla + 1])`, representing two independent decisions the agent must make each step.

#### Dimension 0 - Correction Action

The agent can apply error corrections to data qubits:

- **`0` to `num_data-1`**: Apply **Pauli-X correction** to data qubit `i`
  - This flips the qubit state: `|0⟩ ↔ |1⟩`
  - Used to correct bit-flip (X) errors on data qubits
  - The correction is applied immediately before the physics cycle
  - Note: In this environment, only X corrections are available (Z corrections would require a different gate set)

- **`num_data`**: **Wait** (no correction)
  - Continue without applying any corrections
  - Useful when no errors are detected or when waiting for more information

- **`num_data + 1`**: **ABORT** (terminate episode)
  - The agent declares that the logical qubit has been corrupted
  - Episode terminates immediately
  - Receives a penalty of `-10.0` if the system was still healthy
  - This is a "safe failure" - the system knows it failed, unlike silent failures

#### Dimension 1 - Maintenance Action

The agent can manage the health of measurement ancillas:

- **`0`**: **No maintenance**
  - All ancillas remain online and operational

- **`1` to `num_ancilla`**: **Fix ancilla `i-1`** (recalibration)
  - Takes ancilla `i-1` offline for `cooldown_time` steps (default: 5 steps)
  - During this time:
    - The ancilla's error rate is reset to baseline (`panc0`)
    - The ancilla's syndrome measurements are **masked** (set to 0 in observations)
    - The agent cannot see errors detected by this ancilla
  - After `cooldown_time` steps, the ancilla comes back online with refreshed error rates
  - **Cost**: `-0.5` reward penalty per maintenance action
  - **Trade-off**: Fixing noisy ancillas improves future measurements but temporarily reduces observability

#### Action Examples

```python
# Correct qubit 5 and fix ancilla 2
action = [5, 3]  # Correction: qubit 5, Maintenance: ancilla 2 (index 3 means ancilla 2)

# Wait and fix ancilla 0
action = [env.num_data, 1]  # Correction: wait, Maintenance: ancilla 0

# Correct qubit 3, no maintenance
action = [3, 0]  # Correction: qubit 3, Maintenance: none

# Abort episode
action = [env.num_data + 1, 0]  # Correction: abort, Maintenance: none
```

#### Action Execution Order

Each step, actions are processed in this order:
1. **Maintenance**: If requested, ancilla goes offline and timer starts
2. **Cooldown timers**: Decrement, ancillas that finish come back online
3. **Noise drift**: All qubit error rates update via Ornstein-Uhlenbeck process
4. **Correction**: If requested, Pauli-X applied to data qubit
5. **Physics cycle**: Idle noise, syndrome extraction, measurements
6. **Reward calculation**: Based on logical state and maintenance costs

### Reward Function

| Outcome | Reward | Description |
|---------|--------|-------------|
| Success | `+1.0` | Logical information preserved |
| Maintenance Cost | `-0.5` | Penalty for taking an ancilla offline |
| Declared Failure (ABORT) | `-10.0` | Agent chose to abort (safe failure) |
| Silent Failure | `-1000.0` | Logical qubit flipped without abort (catastrophic) |

**Key Design**: Silent failures are penalized 100× more than declared failures, encouraging conservative behavior when uncertain.

## Physics Model

### Syndrome Extraction Cycle

The environment runs a synchronized 4-phase cycle:

1. **Reset**: All ancillas reset to |0⟩
2. **Prepare**: X-stabilizer ancillas get Hadamard gates
3. **Interact**: Synchronized CNOT gates in 4 time steps:
   - Each ancilla interacts with its neighbors in sequence
   - Hook errors can occur during interactions
4. **Measure**: X-stabilizer ancillas get Hadamard gates, then all ancillas are measured

### Error Types

1. **Idle Noise**: X and Z errors on data qubits during idle time (based on drifted error rates)
2. **Hook Errors**: Mid-cycle ancilla faults that propagate to future neighbors in the CNOT sequence
   - X-stabilizer hook error: Z-error on ancilla → propagates as Z-errors to future data neighbors
   - Z-stabilizer hook error: X-error on ancilla → propagates as X-errors to future data neighbors
3. **Readout Errors**: Measurement flips (does not affect data qubits)

### Logical Failure Detection

The environment checks both logical chains:
- **X-error chain (vertical)**: Odd parity of X-errors along the vertical logical Z chain → failure
- **Z-error chain (horizontal)**: Odd parity of Z-errors along the horizontal logical X chain → failure

If **either** chain has odd parity, logical failure occurs.

## Configuration

Default configuration is loaded from **`config.json`**. You can override with a dict or load another file:

```python
from qec_control import QECMaintenanceEnv, CONFIG, load_config

# Use default config (from config.json)
env = QECMaintenanceEnv(config=CONFIG, render_mode="human")

# Or load a custom config file
custom = load_config("path/to/config.json")
env = QECMaintenanceEnv(config=custom)
```

Main keys in `config.json` (or the CONFIG dict):

```python
CONFIG = {
    "distance": 5,              # Code distance (d=3, 5, 7, ...)
    
    # Noise Baselines
    "panc0": 0.00,             # Base ancilla error rate
    "px0": 0.001,              # Base data X error rate
    "pz0": 0.001,              # Base data Z error rate
    "readout_error": 0.00,     # Measurement flip probability
    
    # Drift Parameters (Ornstein-Uhlenbeck)
    "std_panc": 0.000,         # Ancilla drift standard deviation
    "std_px": 0.0005,          # Data X drift standard deviation
    "std_pz": 0.0005,          # Data Z drift standard deviation
    "tc_anc": 20.0,            # Ancilla time constant
    "tc_data": 50.0,           # Data time constant
    
    # Environment Settings
    "cooldown_time": 5,        # Steps an ancilla is offline when fixed
    "rounds_history": 5        # Number of syndrome rounds in observation
}
```

## Usage

### Basic Example

```python
from qec_control import QECMaintenanceEnv, CONFIG

# Create environment
env = QECMaintenanceEnv(config=CONFIG, render_mode="human")

# Reset
obs, info = env.reset()

# Step 1: Wait and observe (no corrections, no maintenance)
action = [env.num_data, 0]  # [Wait, No maintenance]
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward}")  # Should be 1.0 (success, no maintenance cost)

# Step 2: Correct qubit 5 (apply Pauli-X to fix bit-flip error)
action = [5, 0]  # [Correct qubit 5, No maintenance]
obs, reward, done, truncated, info = env.step(action)

# Step 3: Fix ancilla 2 (takes it offline for cooldown_time steps)
action = [env.num_data, 3]  # [Wait, Fix ancilla 2]
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward}")  # Should be 0.5 (1.0 - 0.5 maintenance cost)

# Step 4: Correct qubit 3 while ancilla 2 is still offline
action = [3, 0]  # [Correct qubit 3, No maintenance]
obs, reward, done, truncated, info = env.step(action)
# Note: Ancilla 2's measurements are masked in the observation

# Step 5: Abort if logical failure is suspected
action = [env.num_data + 1, 0]  # [Abort, No maintenance]
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward}")  # -10.0 if system was healthy, episode terminates
```

### Visualization

```python
env.render(step=0, action=None)
```

The render function displays:
- **Data qubits** (circles): Red dots = X-errors, Blue dots = Z-errors
- **Ancillas** (squares): Orange border = violated stabilizer, Gray = offline
- **Connections**: Gray lines showing stabilizer connections

## Installation

```bash
pip install gymnasium stim numpy matplotlib
```

## Key Features

1. **Non-Stationary Noise**: Error rates drift over time using Ornstein-Uhlenbeck process
2. **Partial Observability**: Offline ancillas are masked (syndrome set to 0)
3. **Dual Protection**: Simultaneously protects against X and Z errors
4. **Hook Errors**: Realistic mid-cycle faults that propagate to multiple qubits
5. **Safety-First Rewards**: Heavy penalty for silent failures encourages conservative behavior
6. **Synchronized Execution**: 4-phase cycle mimics parallel quantum hardware execution

## Testing

Run the demo:

```bash
python qec_control.py
```

This will run 3 episodes with high noise to demonstrate error accumulation and logical failure detection.

## References

- **Stim**: Google's fast Clifford circuit simulator
- **Surface Code**: The rotated surface code is a standard QEC code for fault-tolerant quantum computing
- **Ornstein-Uhlenbeck Process**: Models mean-reverting noise drift in physical systems
