import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(1)
from itertools import count

Pad = pd.read_csv('../PadData_v2.csv')
# Default
T = 3617
m = 16
to_draw = np.sort(Pad['timestamp'].unique())
ccy = np.sort(Pad['currency pair'].unique())
min_history = 1000 # min episode length

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
    n = 1
    target_bid, target_ask, other_bid, other_ask = generate_episode(n,cur)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized

def draw_eval_episode(m, cur, min_history, factor, offset):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    n = (factor * 3600) % (int(to_draw.shape[0] * 0.6)) + offset
    target_bid, target_ask, other_bid, other_ask = generate_episode(n,cur)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized
