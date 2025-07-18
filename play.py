# play.py
import gymnasium as gym
from stable_baselines3 import DQN
import time

# Load the trained model
model = DQN.load("dqn_model.zip")

# Create the same environment
env = gym.make("ALE/Tennis-v5", render_mode="human")

# Run a few episodes
episodes = 5

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Greedy policy
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        time.sleep(0.01)
    print(f"Episode {episode + 1} Reward: {total_reward}")

env.close()
