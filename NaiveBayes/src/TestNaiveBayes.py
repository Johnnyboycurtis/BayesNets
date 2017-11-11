#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:06:30 2017

@author: jonathan
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayes


df = pd.read_csv("../../TAN/data/pima.csv", index_col = 0)
n = df.shape[0]


results = []
for i in tqdm(range(2000)):
    ind = np.random.rand(n) < 0.70
    traindf = df.loc[ind]
    testdf = df.loc[~ind]
    
    nbmodel = NaiveBayes(traindf, 'IsDiabetic')
    
    testresults = nbmodel.Predict(testdf)
    
    accuracy = np.mean(testdf['IsDiabetic'].values == testresults['IsDiabetic'])
    #print(accuracy)
    results.append(accuracy)


res = pd.DataFrame(results, columns = ['accuracy'])
res.hist(bins = 20)

plt.show()

