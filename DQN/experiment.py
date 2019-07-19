import numpy as np
import torch
import os
import functools

# import matplotlib.pyplot as plt
from tqdm import trange

from live import live
from environment import ForexEnv
from agents import RandomAgent
from agents import DQNAgent
from agents import Forex_reward_function
from feature import ForexIdentityFeature
import time

if __name__ == '__main__':
    cur = 'EURUSD'
    reward_path = './'+ cur +'/results/'+ time.strftime("%Y%m%d-%H%M%S") +'/'
    agent_path = './'+ cur +'/agents/' + time.strftime("%Y%m%d-%H%M%S") +'/'

    if not os.path.exists(reward_path):
        os.makedirs(reward_path)
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)

    env = ForexEnv(mode = 'train')

    # train dqn agents
    number_seeds = 1
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = DQNAgent(
            action_set=[0, 1, 2],
            reward_function=functools.partial(Forex_reward_function),
            feature_extractor=ForexIdentityFeature(),
            hidden_dims=[50, 50],
            learning_rate=5e-4,
            buffer_size=50000,
            # batch_size=16,
            batch_size=64,
            num_batches=100,
            starts_learning=1000,
            final_epsilon=0.02,
            discount=0.99,
            target_freq=10,
            verbose=False,
            print_every=10)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=500,
            max_timesteps=3600,
            verbose=True,
            print_every=10)

        file_name = '|'.join(['dqn', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
