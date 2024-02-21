import random
from typing import List

class Environment:
    def __init__(self, n_steps: int):
        self.steps_left = n_steps
        
    def observations(self) -> List[int]:
        return [0, 0, 0] # observations returned by the environment

    def action_space(self) -> List[int]:
        return [0, 1] #        

    def is_done(self) -> bool:
        return self.steps_left <= 0
    
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()
    
class Agent:
    def __init__(self):
        self.total_reward = 0.0
        
    def step(self, env: Environment):
        current_obs = env.observations()
        actions = env.action_space()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

if __name__ == "__main__":
    env = Environment(20)
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print(f"Total reward got: {agent.total_reward:.2f}")