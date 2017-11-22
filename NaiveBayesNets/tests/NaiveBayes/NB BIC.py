#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:28:15 2017

@author: jonathan
"""

import os
os.chdir("/home/jonathan/BayesNets/NaiveBayes/src")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from NaiveBayes import NaiveBayes
import numpy as np

## quick test ##
#print("starting to train Graph")
df = pd.read_csv("../../TAN/data/pima.csv")
class_col_name = "IsDiabetic"
#df = pd.read_csv("../data/Pima.tr.csv")
#class_col_name = "type"
n = df.shape[0]

ind = np.random.rand(n) < 0.75
traindf = df.loc[ind]
testdf = df.loc[~ind]
traincols = ['NoPregnancies', 'PlasmaGlucose', 'DiastolicBP', 'TricepsSkinThickness',
       '2HourSerumInsulin', 'BMI', 'DiabetesPedigreeFunc', 'Age',
       'IsDiabetic']
nbmodel = NaiveBayes(traindf[traincols], class_col_name = class_col_name, progress_bar=False)
results = nbmodel.Predict(newdf = traindf)
accuracy = (traindf[class_col_name].values == results[class_col_name]).mean()
print(f"TAN accuracy: {round(accuracy, 4)}")


Lik = results[[0,1]]
loglike = []
for name, frame in g:
    s = 1 - frame[name] ## calc deviance from true prob
    slog = np.log(s).sum()
    loglike.append(slog)

deviance = -2 * sum(loglike)
k = traindf.columns.shape[0] - 1 ## -1 for class column
n = traindf.shape[0]


BIC = deviance + k*(np.log(n) - np.log(2*np.pi))
print(f"BIC: {BIC}") ## full model: 59089.440799190968




excluded = None
pred = ['NoPregnancies', 'PlasmaGlucose', 'DiastolicBP', 'TricepsSkinThickness',
       '2HourSerumInsulin', 'BMI', 'DiabetesPedigreeFunc', 'Age']

tests = []
for i in range(8):
    print(excluded)
    traincols = pred + ['IsDiabetic']
    nbmodel = NaiveBayes(traindf[traincols], class_col_name = class_col_name, progress_bar=False)
    
    results = nbmodel.Predict(newdf = testdf)
    accuracy = (testdf[class_col_name].values == results[class_col_name]).mean()
    print(f"TAN accuracy: {round(accuracy, 4)}")
    
    results = nbmodel.Predict(newdf = traindf)
    accuracy = (traindf[class_col_name].values == results[class_col_name]).mean()
    excluded = pred.pop()

    g = results.groupby(by = 'IsDiabetic')
    for name, frame in g:
        s = 1 - frame[name] ## calc deviance from true prob
        slog = np.log(s).sum()
        loglike.append(slog)

    deviance = -2 * sum(loglike)
    k = traindf.columns.shape[0] - 1 ## -1 for class column
    n = traindf.shape[0]    
    BIC = deviance + k*(np.log(n) - np.log(2*np.pi))
    print(f"BIC: {BIC} with {traincols}") ## full model: 59089.440799190968
    tests.append((BIC, accuracy, len(traincols)))



