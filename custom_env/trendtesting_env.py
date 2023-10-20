import numpy as np
import gym
import gymnasium
from gymnasium import spaces
from collections import deque
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, env_config):
        super(TradingEnv, self).__init__()

        data = pd.read_csv(env_config["data_filepath"])
        self.historical_data = data['Open']
        self.returns = data['ret']
        self.sma = data['new_column']
        #self.window_size = env_config["window_size"]
        self.unrealized_pnl_history = deque([0.0]*5, maxlen=5)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3), dtype=np.float32)
        
        self.max_episode_steps = env_config["least_episode_size"]  #20
        np.random.seed(123)

        self.reset()


    def step(self, action):
        current_price = self.historical_data[self.current_step]
        current_ret = 0
        unrealized_pnl = 0
        if action == 0:  # Buy
            if self.position == 0:
                self.position = 1
                self.buy_price = current_price
                #reward = 0
            elif self.position == -1:
                r = (self.sell_price - current_price)/self.sell_price
                current_ret = np.log(1 + r)
                self.position = 0
            else:
                unrealized_pnl = np.log(current_price/self.buy_price)
        elif action == 1:  # Sell
            if self.position == 0:
                self.position = -1
                self.sell_price = current_price
                #reward = 0
            elif self.position == 1:
                current_ret = np.log(current_price/self.buy_price)
                self.position = 0
            else:
                r = (self.sell_price - current_price)/self.sell_price
                unrealized_pnl = np.log(1 + r)
        else:  # Hold
            if self.position == 1:
                unrealized_pnl = np.log(current_price/self.buy_price)
            elif self.position == -1:
                r = (self.sell_price - current_price)/self.sell_price
                unrealized_pnl = np.log(1 + r)
            
        reward = current_ret
        
        self.unrealized_pnl_history.append(unrealized_pnl)
        unrealized_pnl_array = np.array(self.unrealized_pnl_history)[-1]
        new_returns = self.returns[self.current_step]
        new_sma = self.sma[self.current_step]
        
        self.state = np.array([new_returns, new_sma, unrealized_pnl_array]).reshape(1, -1)
        self.current_step += 1
        
        done = False
        if self.current_step >= self.last_step: #len(self.historical_data) - 1:
            done = True
            #print("End of episode. State:", self.state)

        info = {
            "current_price": current_price
        }

        return self.state, reward, done, info


    def reset(self):
        self.current_step = 1
        self.position = 0
        self.buy_price = 0
        self.sell_price = 0
        self.unrealized_pnl_history = deque([0.0]*5, maxlen=5)
        unrealized_pnl_array = np.array(self.unrealized_pnl_history)[-1]
        self.current_step += np.random.randint(0, len(self.returns) - 1 - self.max_episode_steps )
        self.last_step = np.random.randint(self.current_step + self.max_episode_steps, min(self.current_step + 10*self.max_episode_steps, len(self.returns) - 1) )
        initial_returns = self.returns[self.current_step]
        initial_sma = self.sma[self.current_step]
        self.state = np.array([initial_returns, initial_sma, unrealized_pnl_array]).reshape(1, -1)
        return self.state
    