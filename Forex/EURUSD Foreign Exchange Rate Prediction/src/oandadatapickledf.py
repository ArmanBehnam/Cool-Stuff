#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:27:37 2017

@author: applesauce

params
http://developer.oanda.com/rest-live-v20/instrument-ep/

EUR_USD - Euro
GBP_USD - Cable
USD_JPY - Gopher
USD_CHF - Swissie

AUD_USD - Aussie
USD_CAD - Loonie
NZD_USD - Kiwi

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os
import sys
import glob



def get_data():
    accountID = os.environ['oanda_demo_id']
    access_token = os.environ['oanda_demo_api']
    
    client = oandapyV20.API(access_token=access_token)
    
    granularities = ['S5']
    
     #['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3','H4', 'H6', 'H8', 'H12', 'D']
    
    granularities = granularities[::-1]
    
    instru = sys.argv[1]
    
    columns=['time', 'volume', 'close', 'high', 'low', 'open', 'complete']
    
    for gran in granularities:
    
        i=0
        hit_today = False
        df = pd.DataFrame(columns=columns)
        df_next = pd.DataFrame(columns=columns)
        last_timestamp = '2005-01-01T00:00:00.000000000Z'
    
        while not hit_today:
    
            params = {'price': 'M', 'granularity': gran, 'count': 5000,
                      'from': last_timestamp,
                      'includeFirst': False,
                      'alignmentTimezone': 'America/New_York'}
            r = instruments.InstrumentsCandles(instrument=instru,params=params)
            client.request(r)
            resp = r.response
            i+=1
            print(r.status_code, i)
            data = []
            for can in resp['candles']:
                data.append([can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']])
            df_next = pd.DataFrame(data, columns=columns)
            df = df.append(df_next, ignore_index=True)
            last_timestamp = list(df.time)[-1]
            last_month = list(df.time)[-1][:7]
            if last_month == '2017-09':
                hit_today = True
    
        save_name = instru+'_'+gran
        print(save_name, df.shape)
        df.to_pickle('data/'+save_name)
    
def clean_data(file_path_name):
    df = pd.read_pickle(file_path_name)
    df['time'] = pd.to_datetime(df['time'])
    df['volume'] = df.volume.astype(int)
    df['close'] = df.close.astype(float)
    df['high'] = df.high.astype(float)
    df['low'] = df.low.astype(float)
    df['open'] = df.open.astype(float)
    df['complete'] = df.complete.astype(bool)
    df.to_pickle(file_path_name)
    pass


if __name__ == '__main__':
    
    
    
    pass
    
    
    
    
    
