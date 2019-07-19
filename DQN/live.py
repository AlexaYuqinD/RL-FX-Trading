import numpy as np
import torch
import matplotlib.pyplot as plt
import functools


def live(agent, environment, num_episodes, max_timesteps, 
    verbose=False, print_every=10):
    """
    Logic for operating over episodes. 
    max_timesteps is maximum number of time steps per episode. 
    """
    observation_data = [] #(self.timestamp, self.state, self.price_record)
    action_data = []
    rewards = []

    if verbose:
        print("agent: %s, number of episodes: %d" % (str(agent), num_episodes))
    
    for episode in range(num_episodes):
        agent.reset_cumulative_reward()
        new_env = environment.reset()
        observation_history = [(new_env[0],new_env[1],new_env[2], False)]
        # observation_history = [(environment.reset()[0],environment.reset()[1],environment.reset()[2], False)]
        action_history = []
        
        t = 0
        done = False
        while not done:
            action = agent.act(observation_history, action_history)
            # action = 0
            timestamp, state, price_record, done = environment.step(action)
            action_history.append(action)
            observation_history.append((timestamp, state, price_record, done))
            t += 1
            done = done or (t == max_timesteps)

        agent.update_buffer(observation_history, action_history)
        # print('uploading')
        agent.learn_from_buffer()
        # print('learning')

        observation_data.append(observation_history)
        action_data.append(action_history)
        rewards.append(agent.cummulative_reward)

        if verbose and (episode % print_every == 0):
            print("ep %d,  reward %.5f" % (episode, agent.cummulative_reward))
            print('short', action_history.count(0))
            print('neutral', action_history.count(1))
            print('long', action_history.count(2))
        if episode % (5 * print_every) == 0:
            test(agent, environment, 10, max_timesteps)

    return observation_data, action_data, rewards

def test(agent, environment, num_episodes, max_timesteps, verbose=True, print_every=10):
    observation_data = [] #(self.timestamp, self.state, self.price_record)
    action_data = []
    rewards = []
    agent.test_mode = True
    print("agent: %s, number of episodes: %d" % (str(agent), num_episodes))
    print (20*"=", "start testing on eval set", 20*"=")
    for episode in range(10):

        agent.reset_cumulative_reward()
        new_env = environment.reset_fixed(episode * 3600)
        observation_history = [(new_env[0], new_env[1], new_env[2], False)]
        # observation_history = [(environment.reset()[0],environment.reset()[1],environment.reset()[2], False)]
        action_history = []

        t = 0
        done = False
        while not done:
            action = agent.act(observation_history, action_history)
            # action = 0
            timestamp, state, price_record, done = environment.step(action)
            action_history.append(action)
            observation_history.append((timestamp, state, price_record, done))
            t += 1
            done = done or (t == max_timesteps)

        agent.update_buffer(observation_history, action_history)
        # print('uploading')
        #agent.learn_from_buffer()
        # print('learning')

        observation_data.append(observation_history)
        action_data.append(action_history)
        rewards.append(agent.cummulative_reward)

        if verbose and (episode % print_every == 0):
            print("ep %d,  reward %.5f" % (episode, agent.cummulative_reward))
            print('short', action_history.count(0))
            print('neutral', action_history.count(1))
            print('long', action_history.count(2))
    print ('The sum of the reward is {}'.format(np.sum(np.array(rewards))))
    print (20 * "=", "finishing testing on eval set...", 20 * "=")
    return observation_data, action_data, rewards

### Example of usage
from environment import ForexEnv
from agents import RandomAgent
from agents import DQNAgent
from agents import Forex_reward_function
from feature import ForexIdentityFeature

if __name__=='__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    env = ForexEnv()

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(Forex_reward_function),
        feature_extractor=ForexIdentityFeature(),
        hidden_dims=[50, 50],
        learning_rate=5e-4, 
        buffer_size=5000,
        batch_size=12,
        num_batches=100, 
        starts_learning=5000, 
        final_epsilon=0.02, 
        discount=0.99, 
        target_freq=10,
        verbose=False, 
        print_every=10)

    observation_data, action_data, rewards = live(
                            agent=agent,
                            environment=env,
                            num_episodes=5,
                            max_timesteps=5,
                            verbose=True,
                            print_every=50)

    agent.save('./dqn.pt')
