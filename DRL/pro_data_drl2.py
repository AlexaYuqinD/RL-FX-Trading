import numpy as np
import pandas as pd

def gen_cols(final, cur, lag):
    currency = list(np.sort(final['currency pair'].unique()))
    tmp = final[final['currency pair'] == cur].sort_values(by=['timestamp']).reset_index(drop = True)
    for i in range(1,lag+1):
        colname1 = 'bid_lag_' + str(i)
        colname2 = 'ask_lag_' + str(i)
        tmp[colname1] = np.log(tmp['bid price']) - np.log(tmp['bid price'].shift(i))
        tmp[colname2] = np.log(tmp['ask price']) - np.log(tmp['ask price'].shift(i))
    for ccy in currency:
        if ccy == cur:
            pass
        else:
            _tmp = final[final['currency pair'] == ccy].sort_values(by=['timestamp']).reset_index(drop = True)
            for i in range(1,lag+1):
                colname3 = ccy + '_bid_lag_' + str(i)
                colname4 = ccy + '_ask_lag_' + str(i)
                tmp[colname3] = np.log(_tmp['bid price']) - np.log(_tmp['bid price'].shift(i))
                tmp[colname4] = np.log(_tmp['ask price']) - np.log(_tmp['ask price'].shift(i))
    tmp = tmp.reset_index(drop=True)
    tmp = tmp[lag:]
    return tmp

def CreateFeature(cur, lag, week_num):
    date_list = ['0201','0203','0204','0205','0206',
                 '0207','0208','0210','0211','0212',
                 '0213','0214','0215','0217','0218',
                 '0219','0220','0221','0222','0224',
                 '0225','0226','0227','0228','0301']
    train_week_1 = date_list[0:5]
    train_week_2 = date_list[5:10]
    train_week_3 = date_list[10:15]
    train_week_4 = date_list[15:20]
    eval_week_1 = date_list[5:7]
    eval_week_2 = date_list[10:12]
    eval_week_3 = date_list[15:17]
    eval_week_4 = date_list[20:22]

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


    final_train = None
    final_eval = None
    for train_date in train_week:
        filename = './final/final-' + train_date + '.csv'
        tmp = pd.read_csv(filename)
        if final_train is not None:
            final_train = final_train.append(tmp)
        else:
            final_train = tmp

    _train = gen_cols(final_train,cur,lag)
    trainname = './data/train_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
    _train.to_csv(trainname,index=False)

    for eval_date in eval_week:
        filename = './final/final-' + eval_date + '.csv'
        tmp = pd.read_csv(filename)
        if final_eval is not None:
            final_eval = final_eval.append(tmp)
        else:
            final_eval = tmp
    _eval = gen_cols(final_eval,cur,lag)
    evalname = './data/eval_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
    _eval.to_csv(evalname,index=False)

if __name__=='__main__':
    CreateFeature('AUDUSD', 32, 2)
