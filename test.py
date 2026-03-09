"""
QEC control gym: entry point and backward compatibility.
Config is loaded from config.json. Override with env.load_config(path) or pass a dict to QECMaintenanceEnv(config=...).
"""
from env import QECMaintenanceEnv, load_config

CONFIG = load_config()

__all__ = ["QECMaintenanceEnv", "CONFIG", "load_config"]

if __name__ == "__main__":

    env = QECMaintenanceEnv(CONFIG, render_mode="human")
    print("Starting Unified Protection Demo...")

    for ep in range(1):
        obs, _ = env.reset()
        if env.render_mode == "human":
            env.render(step=0)

        for i in range(100):
            action = env.num_data if not env.use_hook_errors else [env.num_data, 0]
            obs, reward, done, _, _ = env.step(action)
            if env.render_mode == "human":
                print(f"  Step {i+1} Reward: {reward}")
                env.render(step=i + 1)
            if done:
                if env.render_mode == "human":
                    print(f"  FAILED at step {i+1}!")
                break
