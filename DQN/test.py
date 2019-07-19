"""
test and visualize trained cartpole agents
"""
import numpy as np
import torch
import pandas as pd
# from gym.envs.classic_control import rendering
# import time
# import skvideo.io
import functools

from environment import ForexEnv
from agents import RandomAgent
from agents import DQNAgent
from agents import Forex_reward_function
from feature import ForexIdentityFeature

def test(agent, environment, max_timesteps, n):
    """
    return observation and action data for one episode
    """
    # observation_history is a list of tuples (observation, termination signal)
    new_env = environment.reset_eval(max_timesteps * n)
    observation_history = [(new_env[0],new_env[1],new_env[2], False)]
    action_history = []

    t = 0
    done = False
    while not done:
        action = agent.act(observation_history, action_history)
        timestamp, state, price_record, done = environment.step(action)
        action_history.append(action)
        observation_history.append((timestamp, state, price_record, done))
        t += 1
        done = done or (t == max_timesteps)
    print(action_history)

    return observation_history, action_history

# def cap_reward(seed,cur = 'AUDUSD',lag = 16, length = 360):
#     filename = './data/' + cur + '_lag_' + str(lag) + '.csv'
#     df= pd.read_csv(filename).reset_index(drop = True)

if __name__=='__main__':
    dqn_model_path = './AUDUSD/agents/20190526-174236/dqn|0.pt'

    np.random.seed(321)
    torch.manual_seed(123)

    env = ForexEnv(mode = 'eval')
    eps = 23
    rewards = []

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(Forex_reward_function),
        feature_extractor=ForexIdentityFeature(),
        hidden_dims=[10, 10],
        test_model_path=dqn_model_path)

    for e in range(eps):
        observation_history, action_history = test(
            agent=agent,
            environment=env,
            max_timesteps=3600,
            n=e)
        r = torch.sum(agent.get_episode_reward(observation_history, action_history))
        print('reward %.5f' % r)
        rewards.append(r)
        # print(action_history)
        if e == eps -1:
            print(agent.get_episode_reward(observation_history, action_history))
            print('short', action_history.count(0))
            print('neutral', action_history.count(1))
            print('long', action_history.count(2))

    reward = torch.mean(torch.stack(rewards))

    print('agent %s, cumulative reward %.4f' % (str(agent), reward))


