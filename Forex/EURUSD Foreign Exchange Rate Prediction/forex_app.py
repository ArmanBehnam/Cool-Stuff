from flask import Flask, request, render_template
import json
import requests
import socket
from datetime import datetime
import pickle
import time
import pandas as pd
import numpy as np
from src.oandadatapostgres import return_data_table_gt_time

'''
SSH 22
HTTP 80
Custom TCP Rule 8080
RabitMQ
https://www.tradingview.com/widget/
'''

app = Flask(__name__)

@app.route('/liveprediction.html')
def grid():
    columns=['time', 'open', 'high', 'low', 'close', 'volume', 'table_name','y_pred', 'y_pred_proba']
    live_pred = pickle.load(open('picklehistory/live_results_df.pkl', 'rb')).values
    live_feat_imp = pickle.load(open('picklehistory/feature_importance_df.pkl', 'rb')).values
    live_trade = pickle.load(open('picklehistory/eur_usd_m15_prediction_df.pkl', 'rb')).reset_index()
    live_trade['y_pred'] = live_trade['y_pred'].shift(1)
    live_trade['y_pred_proba'] = live_trade['y_pred_proba'].shift(1)
    cum_returns = np.round((np.exp(np.sum(live_trade['y_pred_returns']))-1)*100,2)
    live_trade = live_trade.values
    return render_template('liveprediction.html', live_pred=live_pred, live_feat_imp=live_feat_imp, live_trade=live_trade, cum_returns=cum_returns)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/tables.html')
def tables():
    return render_template('tables.html')

@app.route('/flot.html')
def flot():
    return render_template('flot.html')

@app.route('/morris.html')
def morris():
    return render_template('morris.html')

@app.route('/forms.html')
def forms():
    return render_template('forms.html')

@app.route('/panels-wells.html')
def panelswells():
    return render_template('panels-wells.html')

@app.route('/buttons.html')
def buttons():
    return render_template('buttons.html')

@app.route('/notifications.html')
def notifications():
    return render_template('notifications.html')

@app.route('/typography.html')
def typography():
    return render_template('typography.html')

@app.route('/icons.html')
def icons():
    return render_template('icons.html')

@app.route('/blank.html')
def blank():
    return render_template('blank.html')


@app.route('/login.html')
def login():
    return render_template('login.html')

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)




































    pass
