# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 17:17:17 2016

@author: sicong.yang
"""
import xgboost as xgb
import scipy.io as sio
import numpy as np
import time
import os

from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

from glob import glob
from operator import itemgetter


def read_features(path):
    """
    read features from .mat files by time sequence
    Each .mat file contains a ten minutes clip of Q features, 
    which divided into 10x512 matrix, 
    for one minute a piece of input data.
    :return: features X and class labels y
    """
    X = []
    y = []

    files = sorted(glob(path))
    for f in files:
        content = sio.loadmat(f)
        feat = content['feat']
        # print f, f[-5]
        stat = int(f[-5])

        X.append(feat)
        y.append(stat)

    return X, y

def run_single(train, test, features, target, random_state=1):
    eta = 0.1
    max_depth = 10
    subsample = 0.92
    colsample_bytree = 0.92
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 1000
    early_stopping_rounds = 50
    test_size = 0.2

    kf = KFold(len(train.index), n_folds=int(round(1/test_size, 0)), shuffle=True, random_state=random_state)
    train_index, test_index = list(kf)[0]
    print('Length of train: {}'.format(len(train_index)))
    print('Length of valid: {}'.format(len(test_index)))

    X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
    y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]

    print('Length train:', len(X_train))
    print('Length valid:', len(X_valid))

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features].as_matrix()), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance
    
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
