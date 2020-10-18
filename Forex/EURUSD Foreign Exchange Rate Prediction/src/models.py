
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from datetime import datetime
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics.scorer import make_scorer
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from keras.utils.np_utils import to_categorical
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.optimizers import SGD
# from keras.wrappers.scikit_learn import KerasClassifier
# import theano
import xgboost as xgb
from xgboost import XGBClassifier
import gc
import operator
import time
import pickle
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os
import sys
import glob
import psycopg2 as pg2
from sqlalchemy import create_engine
from oandadatapostgres import *
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})


def get_data(file_name, date_start, date_end):
    '''
    pickled df of candles with open, high, low, close, volume, complete
    '''
    df = pd.read_pickle('../data/'+file_name)
    df.set_index('time', inplace=True)
    df.drop('complete', axis=1, inplace=True)
    #df_columns = list(df.columns)
    df = df.loc[date_start:date_end]
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

def momentum_columns():
    '''
    not used currently
    '''
    mom_cols = []
    for mom_time in [1, 15, 30, 60, 120]:
        col = 'average_log_return_{}_sign'.format(mom_time)
        df[col] = df['log_returns'].rolling(mom_time).mean().apply(up_down) #the sign of the average returns of the last x candles
        mom_cols.append(col)
    return mom_cols

def calc_feature_importance(x, y):
    '''
    http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
    '''
    chi_k = Pipeline([('scale',MinMaxScaler(feature_range=(0.00001, 1))), ('kbest', SelectKBest(chi2, k=10))])
    chi_k.fit(x, y)
    chi_feat_imp = pd.DataFrame(chi_k.steps[1][1].scores_, index=x.columns).sort_values(by=0, ascending=False)
    f_cl_k = Pipeline([('scale',MinMaxScaler(feature_range=(0.00001, 1))), ('kbest', SelectKBest(f_classif, k=10))])
    f_cl_k.fit(x, y)
    f_cl_k_feat_imp = pd.DataFrame(f_cl_k.steps[1][1].scores_, index=x.columns).sort_values(by=0, ascending=False)
    mut_i_c = Pipeline([('scale',MinMaxScaler(feature_range=(0.00001, 1))), ('kbest', SelectKBest(mutual_info_classif, k=10))])
    mut_i_c.fit(x, y)
    mut_i_c_feat_imp = pd.DataFrame(mut_i_c.steps[1][1].scores_, index=x.columns).sort_values(by=0, ascending=False)
    lr = LogisticRegression(C=1, penalty="l2", dual=False).fit(x, y)
    model_lr = SelectFromModel(lr, prefit=True)
    x_new = model_lr.transform(x)
    print(x_new.shape)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
    model_lsvc = SelectFromModel(lsvc, prefit=True)
    x_new = model_lsvc.transform(x)
    print(x_new.shape)
    gbc = GradientBoostingClassifier(max_depth=5, n_estimators=1000)
    gbc.fit(x, y)
    print(gbc.feature_importances_)
    gbc_feat_imp = pd.DataFrame(gbc.feature_importances_, index=x.columns).sort_values(by=0, ascending=False)
    model_gbc = SelectFromModel(gbc, prefit=True)
    x_new = model_gbc.transform(x)
    print(x_new.shape)
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(x, y)
    print(rfc.feature_importances_)

    # lr = LogisticRegression()
    # rfecv = RFECV(estimator=lr, step=1, cv=TimeSeriesSplit(2), scoring='roc_auc', n_jobs=-1)
    # rfecv.fit(x, y)
    # print("Optimal number of features : %d" % rfecv.n_features_)
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()

    return x, y, chi_feat_imp, f_cl_k_feat_imp, mut_i_c_feat_imp, lr, model_lr, lsvc, model_lsvc, gbc, model_gbc, gbc_feat_imp, rfc

def calc_feature_importance_lr(x, y):
    lr = Pipeline([('scale',StandardScaler()), ('clf', LogisticRegression(penalty='l2', C=1))])
    lr.fit(x, y)
    model_lr = SelectFromModel(lr.steps[1][1], prefit=True)
    x_new = model_lr.transform(x)
    print(x_new.shape)
    return x, y, lr, model_lr

# def get_nn(num_inputs=30):
#     '''
#     build keras/tensorflow nn
#     '''
#     model = Sequential()
#     num_neurons_in_layer = 50
#     num_classes = 2
#     model.add(Dense(input_dim=num_inputs,
#                      units=num_neurons_in_layer,
#                      kernel_initializer='uniform',
#                      activation='tanh'))
#     model.add(Dense(input_dim=num_neurons_in_layer,
#                      units=num_neurons_in_layer,
#                      kernel_initializer='uniform',
#                      activation='tanh'))
#     model.add(Dense(input_dim=num_neurons_in_layer,
#                      units=num_neurons_in_layer,
#                      kernel_initializer='uniform',
#                      activation='tanh'))
#     model.add(Dense(input_dim=num_neurons_in_layer,
#                      units=num_classes,
#                      kernel_initializer='uniform',
#                      activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
#     return model

def get_variety_pipes():
    '''
    builds pipelines to cross val and gridsearch
    {'clf': MLPClassifier(activation='logistic', alpha=0.0001, batch_size=500, beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(50, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False), 'clf__activation': 'logistic', 'clf__alpha': 1e-06, 'clf__batch_size': 200,
       'clf__early_stopping': True, 'clf__hidden_layer_sizes': (50, 1),
       'clf__max_iter': 2000, 'clf__shuffle': False, 'pca__n_components': 15}
    '''
    lr = LogisticRegression(penalty='l2', C=1.0)
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    ab = AdaBoostClassifier()
    gb = GradientBoostingClassifier(n_estimators=100)
    # kc = KerasClassifier(build_fn=get_nn, epochs=100, batch_size=500, verbose=0)
    xg = XGBClassifier(max_depth=3, n_estimators=100)
    ml = MLPClassifier(hidden_layer_sizes=(50,1), activation='logistic', learning_rate_init=0.001, batch_size=200, max_iter=5000, early_stopping=True, shuffle=False)
    svc_r = SVC(kernel='rbf', probability=True)
    svc_l = SVC(kernel='linear', probability=True)
    svc_p = SVC(kernel='poly', probability=True)
    svc_s = SVC(kernel='sigmoid', probability=True)
    knc = KNeighborsClassifier(3, n_jobs=-1)
    gpc = GaussianProcessClassifier(n_jobs=-1)
    gnb = GaussianNB()
    qda = QuadraticDiscriminantAnalysis()
    lr_l2_c1 = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)), ('clf', LogisticRegression(penalty='l2', C=1))])
    dt = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)),('clf', DecisionTreeClassifier())])
    rf = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)),('clf', RandomForestClassifier(n_estimators=100))])
    gb = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)),('clf', GradientBoostingClassifier(n_estimators=100))])
    mlp = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)),('clf', MLPClassifier(hidden_layer_sizes=(100,100), activation="logistic", batch_size='auto', max_iter=5000, early_stopping=True))])
    # pca_lr_l1 = Pipeline([('pca', PCA(.99)), ('clf', LogisticRegression(penalty='l1', C=1))])
    # pca_lr_l2 = Pipeline([('pca', PCA(.99)), ('clf', LogisticRegression(penalty='l2', C=1))])
    # lr_l2_c10 = Pipeline([('clf', LogisticRegression(penalty='l2', C=10))])
    # lr_l2_c01 = Pipeline([('clf', LogisticRegression(penalty='l2', C=.01))])
    # lr_mm_scale = Pipeline([('scale',MinMaxScaler()), ('clf', LogisticRegression(penalty='l2', C=1))])
    # lr_ss_scale = Pipeline([('scale',StandardScaler()), ('clf', LogisticRegression())])
    # lr_ss_scale = Pipeline([('scale',StandardScaler()), ('pca', PCA(.99)), ('clf', gb)])
    # classifiers = [lr, gb, ml]
    pipes = {
    'Logistic Regression': lr_l2_c1,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Multilayer Perceptron': mlp
    # 'lr_l2_c10': lr_l2_c10,
    # 'lr_l2_c01': lr_l2_c01
    }
    # for clf in classifiers:
    #     pipes[clf.__class__.__name__] = Pipeline([('scale',MinMaxScaler(feature_range=(0.00001, 1))), ('clf', clf)])
    return pipes

def store_pipe_params():
    pipeline = Pipeline([('scale',StandardScaler()), ('pca', PCA()), ('clf', LogisticRegression())])
    parameters = [
    {
    'clf': [LogisticRegression()],
    'pca__n_components': list(range(2, 207, 5)),
    'clf__C': [1, .1]
    }]
    # 'clf': [XGBClassifier()],
    # 'pca__n_components': [.6, .7, .8, .9, .95],
    # 'kbest__score_func': [chi2, f_classif, mutual_info_classif],
    # 'kbest__k': list(range(10, 60, 5)),
    # 'clf__n_estimators': [100, 200, 500, 1000],
    # 'clf__max_depth': [3,5,8,10]},
    # {
    # 'clf': [MLPClassifier()],
    # 'pca__n_components': [50, 100, 150],
    # 'clf__hidden_layer_sizes': [(50,50), (50,50,50), (100,100), (100,100,100), (150,150), (150,150,150)],
    # 'clf__activation': ['logistic', 'tanh', 'relu'],
    # 'clf__learning_rate_init': [.0001],
    # 'clf__batch_size': [500],
    # 'clf__max_iter': [5000],
    # 'clf__early_stopping': [True]
    # }]
    return pipeline, parameters

def gridsearch_score_returns(y_true, y_pred):
    '''
    gridsearch scoring function that returns cumulative returns
    '''
    prediction_df = pd.DataFrame([])
    prediction_df['log_returns'] = df.loc[y_true.index, 'log_returns']
    prediction_df['y_pred'] = y_pred
    score = (np.exp(np.sum(prediction_df['y_pred'].map({1:1, 0:-1}).shift(1) * prediction_df['log_returns']))-1)*100
    print('{:.2f}%'.format(score))
    return score

def dump_big_gridsearch(n_splits=2):
    '''
    grid search every model and return gridsearch and results
    '''
    table_names = ['eur_usd_m15']#['eur_usd_d', 'eur_usd_h12', 'eur_usd_h6', 'eur_usd_h1', 'eur_usd_m30', 'eur_usd_m15', 'eur_usd_m1']
    start_time_stamps = [datetime(2007,1,1)]#[datetime(2000,1,1), datetime(2000,1,1), datetime(2000,1,1), datetime(2000,1,1), datetime(2006,1,1), datetime(2012,1,1), datetime(2017,5,1)]
    for i in range(len(table_names)):
        table_name = table_names[i].upper()
        from_time = start_time_stamps[i]
        df = get_data(table_name, from_time, datetime(2018,1,1))
        print('got data')
        df = add_target(df)
        print('added targets')
        df = add_features(df)
        print('added features')
        x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
        print('starting grid search')
        score_returns = make_scorer(gridsearch_score_returns, greater_is_better=True)
        pipeline, parameters = store_pipe_params()
        parameters = parameters[0]
        grid_search = GridSearchCV(pipeline,
                                 param_grid=parameters,
                                 verbose=1,
                                 n_jobs=-1,
                                 cv=TimeSeriesSplit(n_splits=n_splits),
                                 scoring='roc_auc')
        grid_search.fit(x, y)
        grid_search_results = pd.DataFrame(grid_search.cv_results_)
        pickle.dump(grid_search, open('../picklehistory/'+table_name+'_grid_object_vlr.pkl', 'wb'))
        pickle.dump(grid_search_results, open('../picklehistory/'+table_name+'_grid_results_vlr.pkl', 'wb'))

def load_gridsearch(file_name):
    '''
    load pickled gridsearched model
    '''
    pick = pickle.load(open(file_name, 'rb'))
    return pick

def get_lr_grids():
    table_names = ['eur_usd_d', 'eur_usd_h12', 'eur_usd_h6', 'eur_usd_h1', 'eur_usd_m30', 'eur_usd_m15', 'eur_usd_m1']
    pipes = {}
    for table in table_names:
        table_upper = table.upper()
        pipes[table+'_lr'] = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)), ('clf', LogisticRegression(penalty='l2', C=1.0))])
    return pipes

def get_nn_grids():
    table_names = ['eur_usd_d', 'eur_usd_h12', 'eur_usd_h6', 'eur_usd_h1', 'eur_usd_m30', 'eur_usd_m15', 'eur_usd_m1']
    pipes = {}
    for table in table_names:
        table_upper = table.upper()
        pipes[table+'_nn'] = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)), ('clf', MLPClassifier(hidden_layer_sizes=(100,100), activation="logistic", batch_size='auto', max_iter=5000, early_stopping=True))])
    return pipes

def get_xg_grids():
    table_names = ['eur_usd_d', 'eur_usd_h12', 'eur_usd_h6', 'eur_usd_h1', 'eur_usd_m30', 'eur_usd_m15', 'eur_usd_m1']
    pipes = {}
    for table in table_names:
        table_upper = table.upper()
    pipes[table+'_xg'] = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)), ('clf', XGBClassifier(n_estimators=500))])

    return pipes

def dump_live_model():
    data = return_data_table('eur_usd_m15')
    df = clean_data(data)
    df = add_target(df)
    df = add_features(df)
    x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
    model = lr = Pipeline([('scale',StandardScaler()), ('pca',PCA(100)), ('clf', LogisticRegression())])
    model.fit(x, y)
    pickle.dump(model, open('../picklehistory/live_lr_eur_usd_m15_model.pkl', 'wb'))

def var_model_one_gran_pipe_cross_val(x, y, df, pipes, n_splits=2):
    '''
    cross validates models and returns prediction results
    '''
    ts = TimeSeriesSplit(n_splits=n_splits)
    prediction_df = pd.DataFrame([])
    for split_index, (train_index, test_index) in enumerate(ts.split(x)):
        print('split index: {}'.format(split_index))
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        prediction_df['y_test_{}'.format(str(split_index))] = y_test.values
        prediction_df['log_returns_{}'.format(str(split_index))] = df['log_returns'][test_index].values
        for key, pipe in pipes.items():
            start = time.time()
            pipe.fit(x_train, y_train)
            y_pred = pipe.predict(x_test)
            y_pred_proba = pipe.predict_proba(x_test)
            prediction_df['{}_{}_pred'.format(key, str(split_index))] = y_pred
            prediction_df['{}_{}_pred_proba'.format(key, str(split_index))] = y_pred_proba[:,1]
            end = time.time()
            print('trained: {} seconds: {:.2f}'.format(key, end-start))
    return prediction_df

def specific_model_gran_pipe_cross_val(x, y, df, pipe_name, pipe, n_splits=2):
    '''
    cross validates models and returns prediction results
    '''
    ts = TimeSeriesSplit(n_splits=n_splits)
    prediction_df = pd.DataFrame([])
    for split_index, (train_index, test_index) in enumerate(ts.split(x)):
        print('split index: {}'.format(split_index))
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        prediction_df['y_test_{}'.format(str(split_index))] = y_test.values
        prediction_df['log_returns_{}'.format(str(split_index))] = df['log_returns'][test_index].values
        start = time.time()
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        y_pred_proba = pipe.predict_proba(x_test)
        prediction_df['{}_{}_pred'.format(pipe_name, str(split_index))] = y_pred
        prediction_df['{}_{}_pred_proba'.format(pipe_name, str(split_index))] = y_pred_proba[:,1]
        end = time.time()
        print('trained: {} seconds: {:.2f}'.format(pipe_name, end-start))
    return prediction_df

def calc_and_print_prediction_returns_pred(prediction_df):
    '''
    calculates and prints the returns with the prediction 0 or 1 to trade
    '''
    pred_cols = [col for col in prediction_df.columns if col[-4:] == 'pred']
    for pred_col in pred_cols:
        sp_ind = re.search('_\d_', pred_col).group(0)[1]
        prediction_df[pred_col+'_returns'] = prediction_df[pred_col].map({1:1, 0:-1}).shift(1) * prediction_df['log_returns_{}'.format(sp_ind)]
        print('{} {:.2f}%'.format(pred_col+'_returns', (np.exp(np.sum(prediction_df[pred_col+'_returns']))-1)*100))
    return prediction_df

def calc_and_print_prediction_returns_proba():
    '''
    use the predict proba (0.0 to 1.0) to trade the distribution and standard deviations
    '''



    pass

def calc_and_print_prediction_stats(prediction_df):
    '''
    calculates and prints the prediction stats
    '''
    pred_cols = [col for col in prediction_df.columns if col[-4:] == 'pred']
    for pred_col in pred_cols:
        sp_ind = re.search('_\d_', pred_col).group(0)[1]
        y_true = prediction_df['y_test_{}'.format(sp_ind)]
        y_pred = prediction_df[pred_col]
        print('\n', pred_col)
        up = sum(y_true==1) / len(y_true) *100
        print('up: {:.2f}%'.format(up))
        print('down: {:.2f}%'.format(100-up))
        print('accuracy: {:.2f}%'.format(accuracy_score(y_true, y_pred)*100))
        y_true = pd.Series(y_true, name='Actual')
        y_pred = pd.Series(y_pred, name='Predicted')
        print('classification report: ')
        print(classification_report(y_true, y_pred))
        print('confusion matrix: ')
        print(pd.crosstab(y_true, y_pred))

def plot_compare_scalers(df):
    '''
    compare sklearn scalers
    '''
    close_prices = df['close'].values.reshape(-1,1)
    sc, mm, ma, rs = StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()
    scalers = [sc, mm, ma, rs]
    fig, axes = plt.subplots(len(scalers)+1, 1, figsize=(10,40))
    for i, ax in enumerate(axes.reshape(-1)):
        if i == 0:
            ax.hist(close_prices, bins=100)
            ax.set_title('No Scaling')
        else:
            scale = scalers[i-1]
            close_prices_scaled = scale.fit_transform(close_prices)
            ax.hist(close_prices_scaled, bins=100)
            ax.set_title(scale.__class__.__name__)
            print('{} min: {:.2f} max: {:.2f}'.format(scale.__class__.__name__, close_prices_scaled.min(), close_prices_scaled.max()))

def plot_prediction_roc(mod_name, prediction_df):
    '''
    plot roc curves
    '''
    proba_cols = [col for col in prediction_df.columns if col[-5:] == 'proba' and col[-12] == '1']
    for pred_col in proba_cols:
        sp_ind = re.search('_\d_', pred_col).group(0)[1]
        y_true =  prediction_df['y_test_{}'.format(sp_ind)]
        y_pred_proba = prediction_df[pred_col]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr) * 100
        plt.plot(fpr, tpr, lw=2, label='{} auc: {:.2f}'.format(mod_name, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc='best')

def plot_prediction_returns(prediction_df):
    '''
    plot returns
    '''
    return_cols = [col for col in prediction_df.columns if col[-12:] == 'pred_returns']
    for return_col in return_cols:
        pred_returns = prediction_df[return_col]
        cum_returns = pred_returns.cumsum().apply(np.exp)-1 #you can add log returns and then transpose back with np.exp
        plt.plot(cum_returns)
    plt.xlabel('time')
    plt.ylabel('percent returns')
    plt.legend(loc='best')

def plot_dim_2(x, y):
    '''
    plot 2 pca features
    '''
    pca = Pipeline([('scale',StandardScaler()), ('pca', PCA(n_components=2))])
    # dim_red = [pca]
    fig, ax = plt.subplots(1, 1)
    # for i, ax in enumerate(axes.reshape(-1)):
    x_new = pca.fit_transform(x)
    x_new_one = x_new[y==1]
    x_new_zero = x_new[y==0]
    ax.scatter(x_new_one[:,0], x_new_one[:,1], c='green', label='up', alpha=.05)
    ax.scatter(x_new_zero[:,0], x_new_zero[:,1], c='orange', label='down', alpha=.05)
    ax.set_xlabel('pca 1')
    ax.set_ylabel('pca 2')
    plt.legend(loc='best')

def plot_line_data(price_series, figtitle):
    '''
    plot timeseries numbers
    '''
    fig, ax = plt.subplots(figsize=(25,10))
    ax.plot(price_series)
    ax.set_title(figtitle)

def plot_a_feature(feature_names, date_start, date_end):
    '''
    plot features
    '''
    df.loc[date_start:date_end].plot(y=feature_names, figsize=(25,10))

def plot_pca_elbow(x):
    '''
    plot pca elbow
    '''
    ss = StandardScaler()
    x_ss = ss.fit_transform(x)
    pca = PCA()
    pca.fit(x_ss)
    plt.figure(1)
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_)
    plt.xlim(0, 30)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')

def plot_pred_proba_hist(plot_title, y_pred_proba):
    '''
    plot predict proba distribution
    '''
    fig, ax = plt.subplots()
    ax.hist(y_pred_proba, bins=100)
    ax.set_title(plot_title)

def plot_return_dist(df):
    '''
    plot predict proba distribution
    '''
    return_types = ['log_returns']
    fig, ax = plt.subplots(1, len(return_types), figsize=(25,10))
    i=0
    # for i, ax in enumerate(axes.reshape(-1)):
    returns = df[return_types[i]].dropna().values.reshape(-1,1)
    ax.hist(returns, bins=5000)
    ax.set_title(return_types[i])
    ax.set_xlim(-.005, .005)
    ax.set_ylabel('frequency')
    ax.set_xlabel('returns')
    plt.show()

def all_steps_simple_feature_importance():
    df = get_data('EUR_USD_M15', datetime(2012,1,1), datetime(2018,1,1))
    print('got data')
    df = add_target(df)
    print('added targets')
    df = add_features(df)
    print('added features')
    x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
    x, y, chi_feat_imp, f_cl_k_feat_imp, mut_i_c_feat_imp, lr, model_lr, lsvc, model_lsvc, gbc, model_gbc, gbc_feat_imp, rfc = calc_feature_importance(x, y)
    return  x, y, chi_feat_imp, f_cl_k_feat_imp, mut_i_c_feat_imp, lr, model_lr, lsvc, model_lsvc, gbc, model_gbc, gbc_feat_imp, rfc

def all_steps_for_models_cross_val():
    df = get_data('EUR_USD_M15', datetime(2007,1,1), datetime(2018,1,1))
    print('got data')
    df = add_target(df)
    print('added targets')
    df = add_features(df)
    print('added features')
    x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
    pipes = get_variety_pipes()
    print('got pipes')
    gran = 'eur_usd_m15'
    prediction_dfs = {}
    for pipe_name, pipe in pipes.items():
        print(pipe)
        prediction_df = specific_model_gran_pipe_cross_val(x, y, df, pipe_name, pipe, n_splits=2)
        prediction_df = calc_and_print_prediction_returns_pred(prediction_df)
        prediction_dfs[pipe_name] = prediction_df
    return prediction_dfs

def all_steps_for_grans_one_model_cross_val():
    table_names = ['eur_usd_d', 'eur_usd_h12', 'eur_usd_h6', 'eur_usd_h1', 'eur_usd_m30', 'eur_usd_m15', 'eur_usd_m1']
    start_time_stamps = [datetime(2000,1,1), datetime(2000,1,1), datetime(2000,1,1), datetime(2000,1,1), datetime(2006,1,1), datetime(2012,1,1), datetime(2017,5,1)]
    pipes_nn = get_nn_grids()
    prediction_dfs = {}
    for i in range(len(table_names)):
        gran = table_names[i]
        df = get_data(gran.upper(), start_time_stamps[i], datetime(2018, 1, 1))
        print('got data')
        df = add_target(df)
        print('added targets')
        df = add_features(df)
        print('added features')
        x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
        pipe_nn = pipes_nn[gran+'_nn']
        print('got pipe')
        prediction_df_nn = specific_model_gran_pipe_cross_val(x, y, df, gran, pipe_nn, n_splits=2)
        prediction_df_nn = calc_and_print_prediction_returns_pred(prediction_df_nn)
        prediction_dfs[gran+'_nn'] = prediction_df_nn
    return prediction_dfs

def for_mods_plot_roc_returns(prediction_dfs):
    for key, value in prediction_dfs.items():
        plot_prediction_roc(key, value)
    plt.show()
    for key, value in prediction_dfs.items():
        plot_prediction_returns(value)
    plt.show()

def for_gran_plot_pca():
    df = get_data('EUR_USD_M15', datetime(2007,1,1), datetime(2018,1,1))
    print('got data')
    df = add_target(df)
    print('added targets')
    df = add_features(df)
    print('added features')
    x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
    plot_dim_2(x, y)
    plt.show()
    plot_pca_elbow(x)
    plt.show()


def live_predict_website():
    '''
    danger, warning, success based on proba distribution
    '''
    count=0
    while True:
        start = time.time()
        table_names = ['eur_usd_d', 'eur_usd_h12', 'eur_usd_h6', 'eur_usd_h1', 'eur_usd_m30', 'eur_usd_m15', 'eur_usd_m1']
        start_time_stamps = ['2000-01-01T00:00:00.000000000Z', '2000-01-01T00:00:00.000000000Z', '2000-01-01T00:00:00.000000000Z', '2000-01-01T00:00:00.000000000Z', '2006-01-01T00:00:00.000000000Z', '2007-01-01T00:00:00.000000000Z', '2017-05-01T00:00:00.000000000Z']
        # grid_search_res = load_gridsearch(grid_pickle)
        model = Pipeline([('scale',StandardScaler()), ('clf', LogisticRegression(penalty='l2', C=1))])
        results_df = pd.DataFrame([])
        feature_importance_df = pd.DataFrame([])
        for i in range(len(table_names)):
            table_name = table_names[i]
            data = return_data_table_gt_time(table_name, start_time_stamps[i])
            print('got data')
            df = clean_data(data)
            df = add_target(df)
            df = add_features(df)
            print('added target and features')
            x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
            model.fit(x, y)
            pickle.dump(model, open('../picklehistory/live_lr_'+table_name+'_model.pkl', 'wb'))
            print('fit model')
            feature_importance_df[table_name] = np.round(np.exp(model.steps[1][1].coef_[0]), 2)
            y_pred = model.predict(last_x_pred)
            y_pred_proba = model.predict_proba(last_x_pred)
            last_x_ohlcv.reset_index(inplace=True)
            last_x_ohlcv['table_name'] = table_name
            last_x_ohlcv['y_pred'] = y_pred
            last_x_ohlcv['y_pred'] = last_x_ohlcv['y_pred'].map({1:'Up', 0:'Down'})
            last_x_ohlcv['y_pred_proba'] = np.round(y_pred_proba[:,1]*100, 2)
            last_x_ohlcv['color'] = last_x_ohlcv['y_pred'].map({'Up':'success', 'Down':'danger'})
            results_df = results_df.append(last_x_ohlcv, ignore_index=True)
        feature_importance_df.index = x.columns
        feature_importance_df.sort_values('eur_usd_m15', ascending=False, inplace=True)
        feature_importance_df = feature_importance_df.iloc[:30]
        feature_importance_df.reset_index(inplace=True)
        pickle.dump(results_df, open('../picklehistory/live_results_df.pkl', 'wb'))
        pickle.dump(feature_importance_df, open('../picklehistory/feature_importance_df.pkl', 'wb'))
        count+=1
        end = time.time()
        print('completed prediction: {} in {:.2f} seconds'.format(count, end-start))

def live_trade_one_gran(instru='EUR_USD', gran='M15'):
    '''
    continuously update table with new candles
    '''
    accountID = os.environ['oanda_demo_id']
    access_token = os.environ['oanda_demo_api']
    client = oandapyV20.API(access_token=access_token)
    table_name = instru.lower()+'_'+gran.lower()
    model = load_gridsearch('../picklehistory/live_lr_'+table_name+'_model.pkl')
    '''
    cross validates models and returns prediction results
    '''
    prediction_df = pd.DataFrame([])
    while True:
        last_timestamp = get_last_timestamp(table_name)
        print('table_name {}: last timestamp: {}'.format(table_name, last_timestamp))
        params = {'price': 'M', 'granularity': gran,
                  'count': 5000,
                  'from': last_timestamp,
                  'includeFirst': False,
                  'alignmentTimezone': 'America/New_York'}
        r = instruments.InstrumentsCandles(instrument=instru,params=params)
        client.request(r)
        resp = r.response
        for can in resp['candles']:
            candle = []
            if can['complete'] == True and time_in_table(table_name, can['time']) == False:
                candle.append((can['time'], can['volume'], can['mid']['c'], can['mid']['h'], can['mid']['l'], can['mid']['o'], can['complete']))
                data_to_table(table_name, candle)
                print('table name: {} added: {}'.format(table_name, candle))
                start = time.time()
                last_month = int(candle[0][0][5:7])-1
                last_month_timestamp = candle[0][0][:5]+str(last_month).zfill(2)+candle[0][0][7:]
                data = return_data_table_gt_time('eur_usd_m15', last_month_timestamp)
                df = clean_data(data)
                df = add_target(df)
                df = add_features(df)
                x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
                y_pred = model.predict(last_x_pred)
                y_pred_proba = model.predict_proba(last_x_pred)
                end = time.time()
                print('last x time: {} prediction: {} proba: {} took: {:.2f} seconds'.format(last_x_ohlcv.index[0], y_pred[0], y_pred_proba[0][1], end-start))
                last_x_ohlcv['y_pred'] = y_pred
                last_x_ohlcv['y_pred_proba'] = y_pred_proba[:,1]
                prediction_df = prediction_df.append(last_x_ohlcv)
                prediction_df = add_target(prediction_df)
                prediction_df['y_pred_returns'] = prediction_df['y_pred'].map({1:1, 0:-1}).shift(1) * prediction_df['log_returns']
                print('{} {:.2f}%'.format('y_pred_returns', (np.exp(np.sum(prediction_df['y_pred_returns']))-1)*100))
                prediction_df = prediction_df[['volume', 'open', 'high', 'low', 'close', 'y_pred', 'y_pred_proba', 'log_returns', 'y_pred_returns', 'target_label_direction_shifted']]
                pickle.dump(prediction_df, open('../picklehistory/'+table_name+'_prediction_df.pkl', 'wb'))
                return_cols = ['y_pred_returns', 'log_returns']
                fig = plt.figure(figsize=(17,8))
                for return_col in return_cols:
                    pred_returns = prediction_df[return_col]
                    cum_returns = pred_returns.cumsum().apply(np.exp)-1 #you can add log returns and then transpose back with np.exp
                    cum_returns.fillna(0, inplace=True)
                    plt.plot(cum_returns)
                plt.legend(loc='best')
                plt.savefig('../static/images/'+table_name+'_returns.png')




if __name__ == '__main__':

    #no_params_df, only_time_period_df, other_param_df = feature_dfs()
    # dump_big_gridsearch()

    #live_predict_website()
    # dump_live_model()
    # prediction_dfs = all_steps_for_grans_one_model_cross_val()
    #
    #
    # prediction_dfs = all_steps_for_models_cross_val()
    # for_mods_plot_roc_returns(prediction_dfs)

    # 
    # df = get_data('EUR_USD_M15', datetime(2012,9,1), datetime(2018,6,1))
    # print('got data')
    # df = add_target(df)
    # print('added target')
    # df = add_features(df)
    # print(df.shape)
    # print('added features')
    # x, y, last_x_pred, last_x_ohlcv = split_data_x_y(df)
    # print(x.shape)
    #
    for_gran_plot_pca()

    # # print(x.shape, y.shape)

    #live_trade_one_gran()
    #live_predict_website()
    #x, y, chi_feat_imp, f_cl_k_feat_imp, mut_i_c_feat_imp, lr, model_lr, lsvc, model_lsvc, gbc, model_gbc, gbc_feat_imp, rfc = all_steps_simple_feature_importance()


    #prediction_df_nn, prediction_df_xg = all_steps_gran_cross_val()
