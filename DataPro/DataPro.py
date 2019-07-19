import numpy as np
import pandas as pd
import glob
import os


def rename():
    count = 0
    for filename in glob.glob("fulldata/*.csv"):
        try:
            tmp = pd.read_csv(filename)
            if tmp.shape[0] != 0:
                mm = str(tmp.time[[0]].values[0])[0:2]
                dd = str(tmp.time[[0]].values[0])[3:5]
                newname = 'fulldata/renamed/' + filename[9:28] + mm + dd + '.csv'
                tmp.to_csv(newname,index=False)
                os.remove(filename)
        except:
            pass
        count += 1
        if count % 50 == 0:
            print("changing file #:",count)
    print(count, "files in total")

def gen_list():
    memory = []
    for filename in glob.glob("fulldata/renamed/*.csv"):
        memory.append(filename)
    date = list(set([elem[-8:-4]for elem in memory]))
    cur = list(set([elem[29:35]for elem in memory]))
    return cur, date

def merge_date(cur_list,date_list,lp_list):
    count = 0
    for date in date_list:
        current = None
        for cur in cur_list:
            for lp in lp_list:
                count += 1
                filename = 'fulldata/renamed/' + lp + '-STRM-' +lp[-1] + '-' + cur + '-' + date + '.csv'
                print(filename)
                if os.path.exists(filename):
                    tmp = pd.read_csv(filename, low_memory=False)
                    tmp = tmp.sort_values(by=['time'], ascending=True)
                    tmp['timestamp'] = tmp['time'].astype(str).str[:-4]
                    tmp = tmp.drop_duplicates(['timestamp','currency pair','provider'], 'last')
                    tmp = tmp[tmp['status'] == 'Active']
                    if current is not None:
                        current = current.append(tmp)
                    else:
                        current = tmp
                else:
                    print('no such file')
        current = current[current['bid price'] != 0]
        current = current[current['ask price'] != 0]
        newname = 'fulldata/merge/' + 'merge-' + date + '.csv'
        current.to_csv(newname,index=False)
    print(count, "files in total")

def pad_data():
    for filename in glob.glob("fulldata/merge/*.csv"):
        tmp = pd.read_csv(filename,low_memory=False)
        print('processing', filename)
        pad_col = ['timestamp','currency pair','provider']
        # uniquetime = tmp[tmp['currency pair'].isin(focus_cur)]
        time = np.sort(tmp['timestamp'].unique()).tolist()
        pad = None
        for cur in tmp['currency pair'].unique():
            getlp = tmp[tmp['currency pair'] == cur]
            for lp in getlp['provider'].unique():
                to_pad = getlp[getlp['provider'] == lp]
                to_pad['pad'] = [0]*len(to_pad['provider'])

                cur_lp = pd.DataFrame(columns = pad_col)
                cur_lp['timestamp'] = time
                cur_lp['currency pair'] = [cur]*len(time)
                cur_lp['provider'] = [lp]*len(time)

                cur_lp['_ind'] = cur_lp['timestamp'] + "+" + cur_lp['currency pair'] + "+" + cur_lp['provider']
                to_pad['_ind'] = to_pad['timestamp'] + "+" + to_pad['currency pair'] + "+" + to_pad['provider']
                to_pad = to_pad.drop(columns=['timestamp','currency pair','provider','time'])
                cur_lp = cur_lp.set_index('_ind').join(to_pad.set_index('_ind')).reset_index(drop=True)
                cur_lp[['pad']] = cur_lp[['pad']].fillna(value=1)
                cur_lp = cur_lp.fillna(method='pad')
                print(cur, lp, 'pad ratio', cur_lp[cur_lp['pad']==1].shape[0]/cur_lp.shape[0])
                if pad is not None:
                    pad = pad.append(cur_lp, ignore_index=True)
                else:
                    pad = cur_lp
        new_name = 'fulldata/pad/pad'+filename[20:]
        pad.to_csv(new_name,index=False)

def combine_lp(cur_list,date_list):
    for date in date_list:
        filename = 'fulldata/pad/pad-' + date + '.csv'
        print(filename)
        if os.path.exists(filename):
            all_lp = pd.read_csv(filename, low_memory=False)
            final = None
            for cur in cur_list:
                this_cur = all_lp[all_lp['currency pair'] ==  cur]
                bid_best = this_cur
                bid_best['_ind'] = bid_best['timestamp'] + "+" + bid_best['currency pair']
                bid_best['bid price'] = pd.to_numeric(bid_best['bid price'], errors = 'coerce')
                ask_best = this_cur
                ask_best['_ind'] = ask_best['timestamp'] + "+" + ask_best['currency pair']
                ask_best['ask price'] = pd.to_numeric(ask_best['ask price'], errors = 'coerce')
                bid_best = bid_best.sort_values(by=['bid price'], ascending=True)
                bid_best = bid_best.drop_duplicates(['_ind'], 'last')
                ask_best = ask_best.sort_values(by=['ask price'], ascending=False)
                ask_best = ask_best.drop_duplicates(['_ind'], 'last')
                bid_best = bid_best.rename(index=str, columns={'provider':'bid provider'})
                bid_best = bid_best.rename(index=str, columns={'pad':'bid pad'})
                ask_best = ask_best.rename(index=str, columns={'provider':'ask provider'})
                ask_best = ask_best.rename(index=str, columns={'pad':'ask pad'})
                bid_best = bid_best.drop(columns=['stream', 'ask price','ask volume','guid','tier','status','quote type','currency pair','timestamp'])
                ask_best = ask_best.drop(columns=['stream', 'bid price','bid volume','guid','tier','status','quote type'])
                best = bid_best.set_index('_ind').join(ask_best.set_index('_ind')).reset_index(drop=True)
                best = best.sort_values(by=['currency pair', 'timestamp'], ascending=True)
                if final is not None:
                    final = final.append(best)
                else:
                    final = best
            newname = 'fulldata/final/' + 'final-' + date + '.csv'
            final.to_csv(newname,index=False)
        else:
            print('no such file')

def clean_nan(cur_list):
    for filename in glob.glob("fulldata/archive/*.csv"):
        tmp = pd.read_csv(filename,low_memory=False)
        print('processing', filename)
        tmp = tmp[tmp['bid price'].notnull() & tmp['ask price'].notnull()]

        pad = None
        time = []
        for cur in cur_list:
            current = tmp[tmp['currency pair'] == cur]
            timeframe = np.sort(tmp['timestamp'].unique()).tolist()
            time.append(timeframe)
        base = set(time[0])
        for i in range(1, len(time)):
            base = base.intersection(set(time[i]))
        base = list(base)

        pad = tmp[tmp.timestamp.isin(base)]
        new_name = 'fulldata/final/final'+filename[22:]
        print(pad.isnull().any().any())
        pad.to_csv(new_name,index=False)


if __name__ == '__main__':
    # rename()
    cur = ['AUDUSD', 'USDCAD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDSEK']
    focus_cur = ['AUDUSD', 'EURUSD','USDJPY']
    date = ['0201','0203','0204','0205','0206','0207',
            '0208','0210','0211','0212','0213','0214',
            '0215','0217','0218','0219','0220','0221',
            '0222','0224','0225','0226','0227','0228','0301']
    lp_list = ['LP-1','LP-2','LP-3','LP-4','LP-5']
    # merge_date(cur,date,lp_list)
    # pad_data()
    # combine_lp(cur,date)
    clean_nan(cur)
