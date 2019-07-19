T = 3617
m = 10
to_draw = np.sort(Pad['timestamp'].unique())
ccy = np.sort(Pad['currency pair'].unique())
min_history = 500 # min episode length

    
def generate_episode(n,cur):
    _max = to_draw.shape[0]
    _end = min(n+T, _max)
    timeframe = to_draw[n:_end]
    other_bid = np.zeros((timeframe.shape[0],ccy.shape[0]-1))
    other_ask = np.zeros((timeframe.shape[0],ccy.shape[0]-1))
    i = 0
    for elem in ccy:
        tmp = Pad[Pad['currency pair'] == elem]
        if elem == cur:
            target_bid = tmp[tmp.timestamp.isin(timeframe)]['bid price'].values
            target_ask = tmp[tmp.timestamp.isin(timeframe)]['ask price'].values
        else:
            other_bid[:,i] = tmp[tmp.timestamp.isin(timeframe)]['bid price'].values
            other_ask[:,i] = tmp[tmp.timestamp.isin(timeframe)]['ask price'].values
            i += 1
    return target_bid, target_ask, other_bid, other_ask

def features(price_path,m):
    features = np.zeros((price_path.shape[0]-m,m))
    for i in range(m):
        features[:,i] = (np.log(price_path) - np.log(np.roll(price_path, i+1)))[m:]
    return features

def get_features(target_bid, target_ask, other_bid, other_ask, m):
    feature_span = features(target_bid,m)
    feature_span = np.append(feature_span, features(target_ask,m), axis = 1)
    for i in range(other_bid.shape[1]):
        feature_span = np.append(feature_span, features(other_bid[:,i],m), axis = 1)
    for j in range(other_ask.shape[1]):
        feature_span = np.append(feature_span, features(other_ask[:,j],m), axis = 1)
    return feature_span

def draw_episode(m, cur, min_history):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    n = np.random.randint(to_draw.shape[0] - min_history)
    target_bid, target_ask, other_bid, other_ask = generate_episode(n,cur)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized

def draw_train_episode(m, cur, min_history):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    to_draw_train = to_draw[:int(to_draw.shape[0]*0.6)]
    n = np.random.randint(to_draw_train.shape[0] - min_history)
    target_bid, target_ask, other_bid, other_ask = generate_episode(n,cur)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized

def draw_test_episode(m, cur, min_history):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    to_draw_test = to_draw[int(to_draw.shape[0]*0.8):]
    n = np.random.randint(to_draw_test.shape[0] - min_history)
    target_bid, target_ask, other_bid, other_ask = generate_episode(n,cur)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized

def draw_eval_episode(m, cur, min_history, offset):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    n = int(to_draw.shape[0]*0.6) + (offset * 3000 % (int(to_draw.shape[0]*0.8) - int(to_draw.shape[0]*0.6)))
    target_bid, target_ask, other_bid, other_ask = generate_episode(n,cur)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized


import gym
import gym_banana
import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable


env = gym.make('Banana-v0')
env.seed(1)
torch.manual_seed(1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(256, 1, bias = True)
        torch.nn.init.xavier_uniform(self.affine1.weight)
        self.tanh = nn.Tanh()
        self.saved_log_probs = []
        self.rewards = 0
        self.actions = []

    def forward(self, x):
        x = self.affine1(x)
        action = self.tanh(x)
        return action


global policy
policy = Policy()

optimizer = optim.SGD(policy.parameters(), lr=1e-1)
eps = np.finfo(np.float32).eps.item()


NUM_OF_EVAL_DATA = 10
EPOCHS = 20
PATH = './best_model_AUDUSD.pth'
train_loss = []
eval_reward = []

def main():
    best_accumulative_return = 0
    for epoch in range(EPOCHS):
        for i_episode in range(200):
            ask = np.zeros((1, 1))
            bid = np.zeros((1,1 ))
            previous_action = np.array([0, 0, 1])
            while ask.shape[0] <= 3600 and bid.shape[0]<=3600:
                target_bid, target_ask, feature_span = draw_train_episode(16, 'AUDUSD', 1000)
                bid, ask, features = target_bid[1:]*1e3, target_ask[1:]*1e3, feature_span
            for t in range(3600):  # Don't infinite loop while learning
                state = feature_span[t]
                action = policy(torch.from_numpy(state).float())
                price_change = (ask[t+1] - ask[t]) + (bid[t+1] - bid[t])
                reward = torch.sum(action * price_change)            
                policy.rewards += reward
                previous_action = action

            optimizer.zero_grad()
            loss = - policy.rewards / 3600
            loss.backward(retain_graph=True)
            optimizer.step()
            if i_episode % 50 == 0 :
                print('Epoch: {} Episode: {} The loss of training is {}'.format(epoch, i_episode, loss.item()))
                train_loss.append(loss.item())
            policy.rewards = 0
        # test after running 1000 episodes
        policy.eval()
        print ("start evaluating...")
        with torch.no_grad():
            accumulative_reward_test = 0
            for j in range(NUM_OF_EVAL_DATA):
                ask = np.zeros((1, 1))
                bid = np.zeros((1,1 ))
                previous_action = np.array([0, 0, 1])
                while ask.shape[0] <= 3600 and bid.shape[0]<=3600:
                    target_bid, target_ask, feature_span = draw_eval_episode(16, 'AUDUSD', 1000, j)
                    bid, ask, features = target_bid[1:]*1e3, target_ask[1:]*1e3, feature_span
                for t in range(3600):  # Don't infinite loop while learning
                    state = feature_span[t]
                    action = policy(torch.from_numpy(state).float())
                    price_change = (ask[t+1] - ask[t]) + (bid[t+1] - bid[t])
                    reward = torch.sum(action * price_change)            
                    accumulative_reward_test += reward
                    previous_action = action
            print ("Evaluating on {} datapoint and return is {}".format(NUM_OF_EVAL_DATA, accumulative_reward_test))
            # saving the parameters if the reward is larger than previous one
            eval_reward.append(accumulative_reward_test * 1.0 / NUM_OF_EVAL_DATA)
            if (accumulative_reward_test * 1.0 / NUM_OF_EVAL_DATA > best_accumulative_return):
                torch.save(policy.state_dict(), PATH)
                best_accumulative_return = accumulative_reward_test * 1.0 / NUM_OF_EVAL_DATA 
        print (80*"=")
        policy.train()
    

if __name__ == '__main__':
    main()
