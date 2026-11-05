import numpy as np
import time
from orca_sim import OrcaHandRight

env = OrcaHandRight(render_mode="human")
obs, info = env.reset(seed=42)

print("Simulation running. Close the MuJoCo window or press Ctrl+C in the terminal to exit.")

try:
    # Changed from a 10,000 limit to an infinite loop so the window stays open
    while True:
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # WHAT IS HAPPENING: Pausing the script for 10 milliseconds every loop.
        # WHY: This prevents the CPU from running at lightspeed, syncing the 
        # physics engine roughly to real-world time so you can watch it.
        time.sleep(0.01)
        
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    # This safely catches when you press Ctrl+C to force quit
    pass
finally:
    env.close()
    print("Simulation closed. (Any segfault after this line is just Ubuntu graphics cleanup and can be ignored).")
