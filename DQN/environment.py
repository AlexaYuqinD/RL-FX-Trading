import numpy as np
import pandas as pd
import torch
# import datetime
# from pro_data import CreateFeature

# Pad = pd.read_csv('PadData_v2.csv')
# currency = list(np.sort(Pad['currency pair'].unique()))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class Environment(object):
    """
    generic class for environments
    """
    def reset(self):
        """
        returns initial observation
        """
        pass

    def step(self, action):
        """
        returns (observation, termination signal)
        """
        pass


class ForexEnv(Environment):
    """
    Observation:
        self.timestamp
        self.state = 4 + 144 + 3 = 151
            time sin(2 pi t/T) 4
            log returns: bid, ask price of target currency 2*16
            log returns: mid price of non-target currency 7*16
            position 3
        self.price_record
            bid, ask price of target currency

    Actions:
        0  short
        1   neutral
        2   long

    Starting State:
        random start within training set

    Episode Termination:
        none
    """

    def __init__(self, cur = 'EURUSD', lag = 16, min_history = 1000, mode = 'train', week = 1):
        self.ccy = cur
        self.lag = lag
        self.min_history = min_history
        self.index = None
        self.state = None
        self.price_record = None
        # self.df = CreateFeature(self.ccy, self.lag).reset_index(drop = True)
        trainname = './data/train_' + self.ccy + '_lag_' + str(self.lag) + '_week' + str(week) + '.csv'
        evalname = './data/eval_' + self.ccy + '_lag_' + str(self.lag) + '_week' + str(week) + '.csv'
        self.df_train = pd.read_csv(trainname).reset_index(drop = True)
        self.df_eval = pd.read_csv(evalname).reset_index(drop = True)
        self.trainframe = self.df_train.index.values.tolist()
        self.evalframe = self.df_eval.index.values.tolist()
        self.train = self.trainframe[:-self.min_history]
        self.eval = self.evalframe[:-self.min_history]
        self.mode = mode

    def get_features(self,_idx):
        if self.mode == 'train':
            bid = self.df_train['bid price'].values[_idx]
            ask = self.df_train['ask price'].values[_idx]
            feature_span = self.df_train.iloc[_idx,9:].values
        if self.mode == 'eval':
            bid = self.df_eval['bid price'].values[_idx]
            ask = self.df_eval['ask price'].values[_idx]
            feature_span = self.df_eval.iloc[_idx,9:].values
        return bid, ask, feature_span

    def step(self, action):
        assert action in [0, 1, 2], "invalid action"
        self.index += 1
        if self.mode == 'train':
            done = (self.index == len(self.train))
        elif self.mode == 'eval':
            done = (self.index == len(self.eval))

        position = np.zeros(3)
        position[action] = 1

        bid, ask, feature_span = self.get_features(self.index)
        next_bid, next_ask, _ = self.get_features(self.index + 1)
        self.state = np.append(feature_span,position, axis = 0).astype('float32')
        self.price_record = (torch.tensor(bid).to(device),torch.tensor(ask).to(device),
                             torch.tensor(next_bid).to(device), torch.tensor(next_ask).to(device))
        return torch.tensor(self.index).to(device), torch.tensor(self.state).to(device), self.price_record, done

    def reset_eval(self, n):
        if self.mode == 'train':
            to_draw = self.train
        elif self.mode == 'eval':
            to_draw = self.eval
        self.index = to_draw[n]

        position = np.zeros(3)
        action = np.random.choice(3)
        position[action] = 1

        bid, ask, feature_span = self.get_features(self.index)
        next_bid, next_ask, _ = self.get_features(self.index + 1)
        self.state = np.append(feature_span,position, axis = 0).astype('float32')
        self.price_record = (torch.tensor(bid).to(device),torch.tensor(ask).to(device),
                             torch.tensor(next_bid).to(device), torch.tensor(next_ask).to(device))
        return torch.tensor(self.index).to(device), torch.tensor(self.state).to(device), self.price_record

    def reset(self):
        if self.mode == 'train':
            to_draw = self.train
        elif self.mode == 'eval':
            to_draw = self.eval
        n = np.random.choice(len(to_draw))
        self.index = to_draw[n]

        position = np.zeros(3)
        action = np.random.choice(3)
        position[action] = 1

        bid, ask, feature_span = self.get_features(self.index)
        next_bid, next_ask, _ = self.get_features(self.index + 1)
        self.state = np.append(feature_span,position, axis = 0).astype('float32')
        self.price_record = (torch.tensor(bid).to(device),torch.tensor(ask).to(device),
                             torch.tensor(next_bid).to(device), torch.tensor(next_ask).to(device))
        return torch.tensor(self.index).to(device), torch.tensor(self.state).to(device), self.price_record

    def reset_fixed(self, number):
        to_draw = self.eval
        assert(number < len(to_draw))
        n = number
        self.index = to_draw[n]

        position = np.zeros(3)
        action = 1
        position[action] = 1

        bid, ask, feature_span = self.get_features(self.index)
        next_bid, next_ask, _ = self.get_features(self.index + 1)
        self.state = np.append(feature_span,position, axis = 0).astype('float32')
        self.price_record = (torch.tensor(bid).to(device),torch.tensor(ask).to(device),
                             torch.tensor(next_bid).to(device), torch.tensor(next_ask).to(device))
        return torch.tensor(self.index).to(device), torch.tensor(self.state).to(device), self.price_record


### test
if __name__=='__main__':
    nsteps = 5
    np.random.seed(448)

    env = ForexEnv(mode = 'train')
    time, obs, price = env.reset()
    t = 0
    print(time)
    print(obs.shape)
    print(price)

    done = False
    while not done:
        action = np.random.randint(3)
        time,obs, price, done = env.step(action)
        t += 1
        print(time)
        print(obs.shape)
        print(price)
        done = done or t==nsteps


