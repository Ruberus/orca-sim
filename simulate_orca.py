import numpy as np
import time
from orca_sim import OrcaHandRight

class PIDController:
    def __init__(self, kp, ki, kd, num_motors):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(num_motors)
        self.prev_error = np.zeros(num_motors)

    def calculate(self, current_pos, target_pos, dt):
        error = target_pos - current_pos
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        action = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        return action, error, self.integral, derivative

env = OrcaHandRight(render_mode="human")
obs, info = env.reset(seed=42)

num_motors = env.action_space.shape[0]
dt = 0.01

pid = PIDController(kp=0.5, ki=0.0, kd=0.05, num_motors=num_motors)

current_pos = np.zeros(num_motors)
target_pos = np.full(num_motors, 1.0) 

print("PID Controller active. Watching tabular telemetry for all motors...")

step_counter = 0

try:
    while True:
        action, p_err, i_err, d_err = pid.calculate(current_pos, target_pos, dt)
        
        if step_counter % 100 == 0:
            print(f"\n=================== STEP {step_counter:05d} ===================")
            print(" ID |   P(Err) |   I(Int) | D(Deriv) |   Action ")
            print("-" * 52)
            
            for m in range(num_motors):
                print(f" {m:02d} | {p_err[m]:8.4f} | {i_err[m]:8.4f} | {d_err[m]:8.4f} | {action[m]:8.4f}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        current_pos += action * dt
        
        time.sleep(dt)
        step_counter += 1
        
        if terminated or truncated:
            obs, info = env.reset()
            current_pos = np.zeros(num_motors)
            pid.integral = np.zeros(num_motors)
            pid.prev_error = np.zeros(num_motors)
            step_counter = 0
finally:
    env.close()
