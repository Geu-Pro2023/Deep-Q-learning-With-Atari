# train.py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os

# Choose the policy you want to test: 'MlpPolicy' or 'CnnPolicy'
policy_type = "CnnPolicy"

# Create the environment
env = gym.make("ALE/Tennis-v5", render_mode="rgb_array")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Evaluation callback setup (optional but recommended)
eval_env = gym.make("ALE/Tennis-v5", render_mode="rgb_array")
eval_env = DummyVecEnv([lambda: Monitor(eval_env)])

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

# Set hyperparameters (you can change these for tuning)
model = DQN(
    policy=policy_type,
    env=env,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    exploration_fraction=0.1,
    verbose=1,
    tensorboard_log="./dqn_tennis_tensorboard/"
)

# Train the agent
model.learn(total_timesteps=100000, callback=eval_callback)

# Save the model
model.save("dqn_model.zip")

env.close()
eval_env.close()
