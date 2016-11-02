# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 16:40:36 2016

@author: sicong.yang
"""
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os

from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

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

    files = os.listdir(path)
    for f in files:
        content = sio.loadmat(path + f)
        feat = content['feat']
        # print f, f[-5]
        stat = int(f[-5])

        X.extend(feat)
        if stat > 0:
            y.extend(np.ones(feat.shape[0]))
        else:
            y.extend(np.zeros(feat.shape[0]))

    return X, y

f = open('submission_gbdt.csv', 'w')
f.write("File,Class\n")
for i in xrange(3):
    
    classifiers = []
    classifier_index = 0
    best_classifier = 0
    best_auc = 0
    path = "../features/train_%d/" % (i + 1)

    print "=" * 20
    print "training GBDT model of Patient %d" % (i + 1)
    print "=" * 20

    X, y = read_features(path)

    # replace NaN with 0
    X = np.array(X)
    X[np.isnan(X)] = 0
    y = np.array(y)
    print "\n= Data are loaded. = \n"
    print "total samples: %d" % y.shape[0]
    print "positive samples: %d" % y[y > 0].shape[0]
    print "positive samples ratio: %.2f" % (float(y[y > 0].shape[0]) / y.shape[0])
    
    test_size = .2
    # shuffle and split training and test sets
    kfolds = KFold(len(X), n_folds=int(round(1/test_size,0)), shuffle=True, random_state=1)
    for train_index, test_index in kfolds:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
        #                                                random_state=0)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        # resapmle positive cases
        X_train_pos = X_train[y_train > 0]
        X_train_resampled = np.concatenate((X_train, np.repeat(X_train_pos, 8, axis=0)))
        y_train_resampled = np.concatenate((y_train, np.ones(X_train_pos.shape[0] * 8)))
        if classifier_index == 0:
            print "\n= Resapmled training dataset. (Only show the first fold) = \n"
            print "training samples: %d" % X_train.shape[0]
            print "positive training samples: %d" % X_train_pos.shape[0]
            print "positive training samples ratio: %.2f" % \
                    (float(X_train_pos.shape[0]) / X_train.shape[0])
            print "resampled training dataset positive ratio: %.2f" % \
                    (float(y_train_resampled[y_train_resampled > 0].shape[0]) / y_train_resampled.shape[0])
            print ""
        
        # TODO: how to use 10-fold to train a model with good generalization
        """
        classifiers.append(RandomForestClassifier(n_estimators=100, max_depth=1,
            min_samples_split=5, random_state=0, max_features=None))
        """
        classifiers.append(GradientBoostingClassifier(n_estimators=100, 
                                                      learning_rate=1.0, 
                                                      max_depth=1, random_state=0))
        
        y_score = classifiers[classifier_index].fit(X_train_resampled, y_train_resampled).predict_proba(X_test)
        score = classifiers[classifier_index].score(X_test, y_test)
        print "Model has been trained for round %d." % classifier_index
    
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_classifier = classifier_index
        plt.plot(fpr,tpr,label='%d. area = %.2f' % (classifier_index, roc_auc))
        classifier_index += 1
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("ROC curve of Patient %d" % (i + 1))
    plt.legend(loc='lower right')
    plt.show()

    print "\nThe best classifier is the %dth classifier" % best_classifier
    print "roc-auc score is", best_auc
    print "\npredicting test dataset."
    path = "../features/test_%d/" % (i + 1)
    filenames = os.listdir(path)
    for j in xrange(len(filenames)):
        filename = '%d_%d.mat' % (i + 1, j + 1)
        if filename in filenames:
            X_predict = sio.loadmat(path + filename)['feat']
            X_predict[np.isnan(X_predict)] = 0
            y_predict = classifiers[best_classifier].predict_proba(X_predict)
            if sum(y_predict[:, 0]) < sum(y_predict[:, 1]):
                f.write(filename + ',1\n')
            else:
                f.write(filename + ',%.8f\n' % np.median(y_predict[:, 1]))
    print "\n"
    
f.close()
print "Done!"
