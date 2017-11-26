# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:25:15 2017

@author: jn107154
"""

import os
os.chdir("/home/jonathan/BayesNets/")
import pandas as pd
import numpy as np
from NaiveBayesNets import KFoldNB #SelectiveNB

np.random.seed(678)
pima = pd.read_csv("/home/jonathan/BayesNets/NaiveBayesNets/data/pima.csv")
class_col_name = 'IsDiabetic'
n = pima.shape[0]
ind = np.random.rand(n) < 0.75
trainpima = pima.loc[ind]
testpima = pima.loc[~ind]

"""
mytest = SelectiveNB(trainpima, class_col_name, xclass = 1, init_col=['Age'])

nbmodel = mytest.Build(trainpima)

testresults = nbmodel.Predict(testpima)
testresults[class_col_name] = testresults.idxmax(axis = 1)
    
accuracy = np.mean(testpima[class_col_name].values == testresults[class_col_name])
print(accuracy)
"""

## k-fold cross validation
kfoldtest = KFoldNB(trainpima, class_col_name, xclass = 1, init_cols=['Age'])

nbmodel = kfoldtest.Build(trainpima)

testresults = nbmodel.Predict(testpima)
testresults[class_col_name] = testresults.idxmax(axis = 1)
    
accuracy = np.mean(testpima[class_col_name].values == testresults[class_col_name])
print(accuracy)



