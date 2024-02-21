import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("Breakout-v4")

    total_reward = 0.0
    total_steps = 0

    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        total_steps += 1
        if terminated or truncated:
            break

    print(f'Episode done in {total_steps} steps, total reward {total_reward:.2f}')
