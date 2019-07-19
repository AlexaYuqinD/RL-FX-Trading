import os
import numpy as np
import pandas as pd
import torch
from pro_data_drl import CreateFeature
np.random.seed(1)
torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Change at each time
week_num = 1
lag = 32
cur = 'AUDUSD'
T = 3633
### Change at each time

_train = None
_eval = None
trainname = './data/train_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
if os.path.exists(trainname) == False:
    CreateFeature(cur, lag, week_num)
_train = pd.read_csv(trainname).reset_index(drop = True)
to_draw_train = np.sort(_train['timestamp'].unique())

evalname = './data/eval_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
if os.path.exists(evalname) == False:
    CreateFeature(cur, lag, week_num)
_eval= pd.read_csv(evalname).reset_index(drop = True)
to_draw_eval = np.sort(_eval['timestamp'].unique())


def draw_train_episode(week_num, lag, cur, min_history):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    n = np.random.randint(to_draw_train.shape[0] - min_history)
    _max = to_draw_train.shape[0]
    _end = min(n+T, _max)
    timeframe = to_draw_train[n:_end]
    train = _train[_train.timestamp.isin(timeframe)]
    target_bid = train['bid price'].values
    target_ask = train['ask price'].values
    feature_span = train.iloc[:,-lag*8*2:].values
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return torch.tensor(target_bid).to(device), torch.tensor(target_ask).to(device), torch.tensor(normalized).to(device)

def draw_eval_episode(week_num, lag, cur, min_history, factor, offset):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    to_draw_eval = np.sort(_eval['timestamp'].unique())
    n = (factor * 3600) % int(to_draw_eval.shape[0]- min_history) + offset
    _max = to_draw_eval.shape[0]
    _end = min(n+T, _max)
    timeframe = to_draw_eval[n:_end]
    eval = _eval[_eval.timestamp.isin(timeframe)]
    target_bid = eval['bid price'].values
    target_ask = eval['ask price'].values
    feature_span = eval.iloc[:,-lag*8*2:].values
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return torch.tensor(target_bid).to(device), torch.tensor(target_ask).to(device), torch.tensor(normalized).to(device)

