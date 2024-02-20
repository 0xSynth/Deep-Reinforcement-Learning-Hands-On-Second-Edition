import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

before_training = "before_training.mp4"

if __name__ == "__main__":
    env = gym.make("Breakout-v4", render_mode="rgb_array")
    video = VideoRecorder(env, before_training)

    total_reward = 0.0
    total_steps = 0
    obs_  = env.reset()

    while True:
        env.render()
        video.capture_frame()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated or truncated:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    env.close()
    env.env.close()
    video.close()
