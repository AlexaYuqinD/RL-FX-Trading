import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(1)
from itertools import count

# Pad = pd.read_csv('PadData_v2.csv')
# Default
T = 3617
m = 16

def generate_episode(week_num, cur, mode, min_history, factor, offset):
    date_list = ['0201','0203','0204','0205',
                 '0206','0207','0208','0210',
                 '0211','0212','0213','0214',
                 '0215','0217','0218','0219',
                 '0220','0221','0222','0224',
                 '0225','0226','0227','0228','0301']
    train_week_1 = date_list[0:4]
    train_week_2 = date_list[4:8]
    train_week_3 = date_list[8:12]
    train_week_4 = date_list[12:16]
    train_week_5 = date_list[16:20]
    eval_week_1 = date_list[4:6]
    eval_week_2 = date_list[8:10]
    eval_week_3 = date_list[12:14]
    eval_week_4 = date_list[16:18]
    eval_week_5 = date_list[20:22]

    if week_num == 1:
        train_week = train_week_1
        eval_week = eval_week_1
    elif week_num == 2:
        train_week = train_week_2
        eval_week = eval_week_2
    elif week_num == 3:
        train_week = train_week_3
        eval_week = eval_week_3
    elif week_num == 4:
        train_week = train_week_4
        eval_week = eval_week_4
    elif week_num == 5:
        train_week = train_week_5
        eval_week = eval_week_5

    Pad = None
    if mode == 'train':
        for train_date in train_week:
            filename = './pad/pad-' + train_date + '.csv'
            tmp = pd.read_csv(filename)
            if Pad is not None:
                Pad = Pad.append(tmp)
            else:
                Pad = tmp
    elif mode == 'eval':
        for eval_date in eval_week:
            filename = './pad/pad-' + eval_date + '.csv'
            tmp = pd.read_csv(filename)
            if Pad is not None:
                Pad = Pad.append(tmp)
            else:
                Pad = tmp

    Pad = Pad.sort_values(by=['currency pair','timestamp'])
    to_draw = np.sort(Pad['timestamp'].unique())
    ccy = np.sort(Pad['currency pair'].unique())
    if mode == 'train':
        n = np.random.randint(to_draw.shape[0] - min_history)
    elif mode == 'eval':
        n = (factor * 3600) % int(to_draw.shape[0]- min_history) + offset
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

def draw_train_episode(week_num, m, cur, min_history):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    # to_draw_train = to_draw[:int(to_draw.shape[0]*0.6)]
    # n = np.random.randint(to_draw_train.shape[0] - min_history)
    target_bid, target_ask, other_bid, other_ask = generate_episode(week_num, cur,'train',min_history,0,0)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized

def draw_eval_episode(week_num, m, cur, min_history, factor, offset):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    target_bid, target_ask, other_bid, other_ask = generate_episode(week_num, cur,'eval', min_history, factor, offset)
    feature_span = get_features(target_bid, target_ask, other_bid, other_ask, m)
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return target_bid, target_ask, normalized

