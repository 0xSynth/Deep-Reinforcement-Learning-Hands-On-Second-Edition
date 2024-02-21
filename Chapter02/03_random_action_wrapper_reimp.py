import random
import gymnasium as gym
from gymnasium import ActionWrapper

class RandomAction(ActionWrapper):
    def __init__(self, env, epsilon=0.2):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action
    
if __name__ == "__main__":
    env = RandomAction(gym.make("Breakout-v4"))
    obs, _ = env.reset()
    total_reward = 0.0
    
    while True:
        obs, reward, terminated, truncated, _ = env.step(0)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f'Reward got: {total_reward:.2f}')