import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import re
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics.scorer import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import time
import pickle
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import os
import sys
import glob
import psycopg2 as pg2
from sqlalchemy import create_engine
import configparser
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

config = configparser.ConfigParser()
config.read('../config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']


def time_in_table(table_name, time_stamp):
    '''
    check if time_stamp candle in table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = "SELECT EXISTS(SELECT 1 FROM {} WHERE time='{}');".format(table_name, time_stamp)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data[0][0]

def return_data_table(table_name):
    '''
    get all data from table
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'SELECT * FROM {};'.format(table_name)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def return_data_table_gt_time(table_name, time_stamp):
    '''
    get all data from table
    ex eur_usd_m1, '2017-09-26T15:41:00.000000000Z'
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = "SELECT * FROM {} WHERE time > '{}';".format(table_name, time_stamp)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def get_last_timestamp(table_name):
    '''
    return last timestamp from table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'SELECT time FROM {} ORDER BY time DESC LIMIT 1;'.format(table_name)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data[0][0]

def record_count(table_name):
    '''
    return last timestamp from table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'SELECT COUNT(*) FROM {};'.format(table_name)
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data[0][0]

def data_to_table(table_name, data):
    '''
    insert candles into table_name
    '''
    conn = pg2.connect(dbname='forex')
    conn.autocommit = True
    cur = conn.cursor()
    query = 'INSERT INTO {}(time, volume, close, high, low, open, complete) VALUES (%s, %s, %s, %s, %s, %s, %s)'.format(table_name)
    cur.executemany(query, data)
    cur.close()
    conn.close()

def current_long_short_units():
    client = oandapyV20.API(access_token=access_token)
    r = positions.PositionDetails(accountID=accountID, instrument='EUR_USD')
    client.request(r)
    resp = r.response
    long_units = int(resp['position']['long']['units'])
    short_units = int(resp['position']['short']['units'])
    return long_units+short_units

def account_balance():
    client = oandapyV20.API(access_token=access_token)
    r = accounts.AccountDetails(accountID)
    client.request(r)
    resp = r.response
    nav = int(resp['account']['NAV'])
    return nav

def clean_data(data):
    '''
    take data dump and convert to df
    '''
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    df = pd.DataFrame(data, columns=columns)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['volume'] = df.volume.astype(float)
    df['close'] = df.close.astype(float)
    df['high'] = df.high.astype(float)
    df['low'] = df.low.astype(float)
    df['open'] = df.open.astype(float)
    df.set_index('time', inplace=True)
    df.drop('complete', axis=1, inplace=True)
    return df

def up_down(row):
    '''
    did the instrument move up or down
    '''
    if row >= 0:
        return 1
    elif row < 0:
        return 0
    else:
        None

def add_target(df):
    '''
    target is the next candles direction (up/dow) shifted to the current timestamp
    predicting the next direction
    '''
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['ari_returns'] = (df['close'] / df['close'].shift(1)) - 1
    df['log_returns_shifted'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_label_direction'] = df['log_returns'].apply(up_down)
    df['target_label_direction_shifted'] = df['log_returns_shifted'].apply(up_down)
    return df

def feature_dfs():
    mom_ind = talib.get_function_groups()['Momentum Indicators']
    over_stud = talib.get_function_groups()['Overlap Studies']
    volu_ind = talib.get_function_groups()['Volume Indicators']
    cyc_ind = talib.get_function_groups()['Cycle Indicators']
    vola_ind = talib.get_function_groups()['Volatility Indicators']
    stats_ind = talib.get_function_groups()['Statistic Functions']
    talib_abstract_fun_list = mom_ind + over_stud + volu_ind + cyc_ind + vola_ind + stats_ind
    talib_abstract_fun_list.remove('MAVP')
    no_params_df = pd.DataFrame([])
    only_time_period_df = pd.DataFrame([])
    other_param_df = pd.DataFrame([])
    for fun in talib_abstract_fun_list:
        info = getattr(talib.abstract, fun).info
        data = pd.Series([info['group'], info['name'], info['display_name'], ['{}: {}'.format(key, value) for key, value in info['parameters'].items()], info['output_names']])
        if len(info['parameters']) == 0:
            no_params_df = no_params_df.append(data, ignore_index=True)
        elif 'timeperiod' in info['parameters'] and len(info['parameters']) == 1:
            only_time_period_df = only_time_period_df.append(data, ignore_index=True)
        else:
            other_param_df = other_param_df.append(data, ignore_index=True)
    ind_dfs = [no_params_df, only_time_period_df, other_param_df]
    for ind_df in ind_dfs:
        ind_df.columns = ['Group', 'Name', 'Short Description', 'Parameters', 'Output Names']
    return no_params_df, only_time_period_df, other_param_df

def add_features(df):
    '''
    technical analysis features
    http://mrjbq7.github.io/ta-lib/doc_index.html
    '''
    no_params_df, only_time_period_df, other_param_df = feature_dfs()
    ohlcv = {
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'].astype(float)
        }
    for fun in no_params_df['Name'].values:
        res = getattr(talib.abstract, fun)(ohlcv)
        output = no_params_df[no_params_df['Name']==fun]['Output Names'].values[0]
        if len(output) == 1:
            df[fun+'_'+output[0].upper()] = res
        else:
            for i, val in enumerate(res):
                df[fun+'_'+output[i].upper()] = val
    for fun in only_time_period_df['Name'].values:
        output = only_time_period_df[only_time_period_df['Name']==fun]['Output Names'].values[0]
        for timeperiod in range(5, 55, 10):
            res = getattr(talib.abstract, fun)(ohlcv, timeperiod=timeperiod)
            if len(output) == 1:
                df[fun+'_'+str(timeperiod)+'_'+output[0].upper()] = res
            else:
                for i, val in enumerate(res):
                    df[fun+'_'+str(timeperiod)+'_'+output[i].upper()] = val
    for fun in other_param_df['Name'].values:
        res = getattr(talib.abstract, fun)(ohlcv)
        output = other_param_df[other_param_df['Name']==fun]['Output Names'].values[0]
        if len(output) == 1:
            df[fun+'_'+output[0].upper()] = res
        else:
            for i, val in enumerate(res):
                df[fun+'_'+output[i].upper()] = val
    return df

def split_data_x_y(df):
    '''
    x is only the technical analysis features
    y is only the whether the close of the next candle went up or down
    '''
    drop_columns = ['volume', 'close', 'high', 'low', 'open', 'complete', 'log_returns', 'ari_returns', 'log_returns_shifted', 'target_label_direction', 'target_label_direction_shifted']
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    predict_columns = [i for i in df.columns if i not in drop_columns]
    last_x_ohlcv = df.iloc[-1:][ohlcv]
    last_x_pred = df.iloc[-1:][predict_columns]
    df.dropna(inplace=True)
    y = df['target_label_direction_shifted']
    x = df[predict_columns]
    return x, y, last_x_pred, last_x_ohlcv

def fit_models():
    data = return_data_table('eur_usd_m15')
    df = clean_data(data)
    df = add_target(df)
    df = add_features(df)
    x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
    model = lr = Pipeline([('scale',StandardScaler()), ('clf', LogisticRegression(penalty='l2', C=1))])
    model.fit(x, y)
    pickle.dump(model, open('../picklehistory/live_lr_eur_usd_m15_model.pkl', 'wb'))

def streamer_ohlcv():
    '''
    {'type': 'PRICE',
     'time': '2017-10-02T18:02:44.860705391Z',
     'bids': [{'price': '1.17376', 'liquidity': 10000000}],
     'asks': [{'price': '1.17386', 'liquidity': 10000000}],
     'closeoutBid': '1.17361', 'closeoutAsk': '1.17401',
     'status': 'tradeable',
     'tradeable': True,
     'instrument': 'EUR_USD'}
    '''
    client = API(access_token=access_token, environment="practice")
    s = PricingStream(accountID=accountID, params={"instruments":instruments})
    instruments = "EUR_USD"
    tick=0
    df = pd.DataFrame([])
    df_ohlc = pd.DataFrame()
    df_volume = pd.DataFrame()
    df_cat = pd.DataFrame
    try:
        for data in api.request(s):
            if data['type'] == 'PRICE':
                tick+=1
                print(tick)
                df_ohlc_shape = df_ohlc.shape
                df = df.append(pd.DataFrame({'price': (float(data['bids'][0]['price']) + float(data['asks'][0]['price']))/2, 'volume': 1}, index=pd.to_datetime([data['time']])))
                df_ohlc = df['price'].resample('1T').ohlc()
                df_volume = df['volume'].resample('1T').sum()
                df_cat = pd.concat([df_ohlc, df_volume], axis=1)
                if df_ohlc.shape != df_ohlc_shape:
                    print(df_cat)
    except V20Error as e:
        print(e)

def trade():
    client = oandapyV20.API(access_token=access_token)
    table_name = 'eur_usd_m15'
    model = pickle.load(open('../picklehistory/live_lr_eur_usd_m15_model.pkl', 'rb'))
    #model = Pipeline([('scale',StandardScaler()), ('clf', LogisticRegression(penalty='l2', C=1))])
    count = 0
    while True:
        try:
            count+=1
            print('iter {}'.format(count))
            last_timestamp = get_last_timestamp(table_name)
            params = {'price': 'M', 'granularity': 'M15',
                      'count': 5,
                      'from': last_timestamp,
                      'includeFirst': False,
                      'alignmentTimezone': 'America/New_York'}
            r = instruments.InstrumentsCandles(instrument='EUR_USD',params=params)
            client.request(r)
            resp = r.response
            candle = []
            for can in resp['candles']:
                if can['complete'] == True and time_in_table(table_name, can['time']) == False:
                    candle.append((can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']))
            if candle:
                start = time.time()
                data_to_table(table_name, candle)
                print('added {} candles'.format(len(candle)))
                last_month = int(candle[0][0][5:7])-1
                last_month_timestamp = candle[0][0][:5]+str(last_month).zfill(2)+candle[0][0][7:]
                data = return_data_table_gt_time(table_name, last_month_timestamp)
                df = clean_data(data)
                df = add_target(df)
                df = add_features(df)
                x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
                # model.fit(x, y)
                print('df shape: {} x shape: {} y shape {}'.format(df.shape, x.shape, y.shape))
                print('last x with {}'.format(last_x_ohlcv))
                y_pred = model.predict(last_x_pred)
                units = current_long_short_units()
                print('current long short units {}'.format(units))
                order ={"order": {
                "units": "1000",
                "instrument": "EUR_USD",
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
                }}
                if y_pred == 1 and units == 0:
                    '''
                    buy 1000
                    '''
                    order_units = "100000"
                    order["order"]["units"] = order_units
                    r = orders.OrderCreate(accountID, data=order)
                    client.request(r)
                    print('y_pred == 1, units == 0\n', r.response)
                elif y_pred == 0 and units == 0:
                    '''
                    sell 1000
                    '''
                    order_units = "-100000"
                    order["order"]["units"] = order_units
                    r = orders.OrderCreate(accountID, data=order)
                    client.request(r)
                    print('y_pred == 0, units == 0\n', r.response)
                elif y_pred == 1 and units < 0:
                    '''
                    buy 2000
                    '''
                    order_units = "200000"
                    order["order"]["units"] = order_units
                    r = orders.OrderCreate(accountID, data=order)
                    client.request(r)
                    print('y_pred == 1, units < 0\n', r.response)
                elif y_pred == 0 and units > 0:
                    '''
                    sell 2000
                    '''
                    order_units = "-200000"
                    order["order"]["units"] = order_units
                    r = orders.OrderCreate(accountID, data=order)
                    client.request(r)
                    print('y_pred == 0, units > 0\n', r.response)
                elif y_pred == 1 and units > 0:
                    '''
                    wait
                    '''
                    print('y_pred == 1, units > 0')
                elif y_pred == 0 and units < 0:
                    '''
                    wait
                    '''
                    print('y_pred == 0, units < 0')
        except Exception as e:
                print(e)
        time.sleep(3)


if __name__ == '__main__':
    trade()
