import json
from ray.rllib.algorithms.ppo import PPO
from custom_env.trendtesting_env import TradingEnv  
from gymnasium.wrappers import EnvCompatibility
from ray.tune.registry import register_env
import numpy as np

def create_compatible_trading_env(env_config):
    env = TradingEnv(env_config)
    return EnvCompatibility(env)

register_env("wrapped_trading_env", create_compatible_trading_env)

def test_trained_model(model_info_path='model_info1.json', data_filepath='testbase_data.csv', num_episodes=50):
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    env_config = {
        "data_filepath": data_filepath,
        "least_episode_size": 75
    }
    env = TradingEnv(env_config)

    checkpoint_path = model_info['checkpoint_path']
    best_trial_config = model_info['best_trial_config']
    trainer = PPO(env="wrapped_trading_env", config=best_trial_config)
    trainer.restore(checkpoint_path)

    rewards = []
    results = []

    for i in range(num_episodes):
        episode_reward = 0
        done = False
        obs = env.reset()
        actions = []
        states = [obs]
        episode_rewards = []
        info_list = []

        #prev_action = np.zeros(env.action_space.shape)  # Initialize with zeros
        #prev_reward = 0.0
        
        while not done:
            state = trainer.get_policy().get_initial_state()
            action, state_out, _ = trainer.compute_single_action(obs, state=state, full_fetch=True) #prev_action=prev_action, prev_reward=prev_reward,
            obs, reward, done, info = env.step(action)
            info_list.append(info["current_price"])
            actions.append(action)
            states.append(obs)
            episode_rewards.append(reward)
            
            episode_reward += reward

            # Update previous action and reward for the next step
            #prev_action = action
            #prev_reward = reward

        rewards.append(episode_reward)
        results.append({
            'episode': i,
            'reward': episode_reward,
            'actions': actions,
            'current_price': info_list,
            'states': states,
            'episode_rewards': episode_rewards
        })

    return rewards, results

if __name__ == "__main__":
    rewards, results = test_trained_model()
    # You can add any additional logging or printing here if needed
    print(f"Average reward over {len(rewards)} episodes: {sum(rewards)/len(rewards)}")
