#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:06:30 2017

@author: jonathan
"""


import sys
sys.path.append("../src/")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayes
import pickle


df = pd.read_csv("../../TAN/data/pima.csv")
col = 'IsDiabetic'
#df = pd.read_csv("../../TAN/data/Pima.tr.csv")
print(df.dtypes)
#col = 'type'
n = df.shape[0]


results = []
for i in tqdm(range(100)):
    ind = np.random.rand(n) < 0.75
    traindf = df.loc[ind]
    testdf = df.loc[~ind]
    
    nbmodel = NaiveBayes(traindf, col)
    
    testresults = nbmodel.Predict(testdf)
    testresults[col] = testresults.idxmax(axis = 1)
    accuracy = np.mean(testdf[col].values == testresults[col])
    #print(accuracy)
    results.append(accuracy)


res = pd.DataFrame(results, columns = ['accuracy'])
res.hist(bins = 20)

#with open("tmp.pickle", "wb+") as myfile:
#    pickle.dump(nbmodel, myfile)

#print('delete tmp.pickle')
plt.show()

