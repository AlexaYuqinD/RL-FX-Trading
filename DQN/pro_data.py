import numpy as np
import pandas as pd
import datetime

def gen_cols(Pad, cur, lag):
    currency = list(np.sort(Pad['currency pair'].unique()))
    tmp = Pad[Pad['currency pair'] == cur].sort_values(by=['timestamp'])
    for i in range(1,lag+1):
        colname1 = 'bid_lag_' + str(i)
        colname2 = 'ask_lag_' + str(i)
        tmp[colname1] = np.log(tmp['bid price']) - np.log(tmp['bid price'].shift(i))
        tmp[colname2] = np.log(tmp['ask price']) - np.log(tmp['ask price'].shift(i))
    for ccy in currency:
        if ccy == cur:
            pass
        else:
            _tmp = Pad[Pad['currency pair'] == ccy].sort_values(by=['timestamp'])
            mid =  pd.DataFrame(np.mean(np.asarray([_tmp['bid price'].values,_tmp['ask price'].values]), axis=0))
            for i in range(1,lag+1):
                colname3 = ccy + '_lag_' + str(i)
                tmp[colname3] = np.log(mid) - np.log(mid.shift(i))
    tmp['date'] = tmp['timestamp'].astype(str).str[0:10]
    tmp['dow'] = pd.to_datetime(tmp['date']).dt.dayofweek
    tmp['hh'] = tmp['timestamp'].astype(str).str[11:13]
    tmp['mm'] = tmp['timestamp'].astype(str).str[14:16]
    tmp['ss'] = tmp['timestamp'].astype(str).str[17:19]
    tmp['time_1'] = np.sin(np.pi*tmp['dow'].values/7)
    tmp['time_2'] = np.sin(np.pi*tmp['hh'].astype('int64').values/24)
    tmp['time_3'] = np.sin(np.pi*tmp['mm'].astype('int64').values/60)
    tmp['time_4'] = np.sin(np.pi*tmp['ss'].astype('int64').values/60)
    tmp = tmp.drop(['date', 'dow','hh','mm','ss'], axis=1)
    tmp = tmp.reset_index(drop=True)
    tmp = tmp[lag:]
    return tmp

def CreateFeature(cur, lag, week_num):
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

    Pad_train = None
    Pad_eval = None
    for train_date in train_week:
        filename = '../pad/pad-' + train_date + '.csv'
        tmp = pd.read_csv(filename)
        if Pad_train is not None:
            Pad_train = Pad_train.append(tmp)
        else:
            Pad_train = tmp

    final_train = gen_cols(Pad_train,cur,lag)
    trainname = './data/train_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
    final_train.to_csv(trainname,index=False)

    for eval_date in eval_week:
        filename = '../pad/pad-' + eval_date + '.csv'
        tmp = pd.read_csv(filename)
        if Pad_eval is not None:
            Pad_eval = Pad_eval.append(tmp)
        else:
            Pad_eval = tmp
    final_eval = gen_cols(Pad_eval,cur,lag)
    evalname = './data/eval_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
    final_eval.to_csv(evalname,index=False)

if __name__=='__main__':
    CreateFeature('EURUSD', 16, 1)
