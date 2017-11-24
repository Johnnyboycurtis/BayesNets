"""
@author: Jonathan Navarrete
"""


import sys
sys.path.append("/home/jonathan/BayesNets/")
import pandas as pd
import numpy as np
from NaiveBayesNets import NaiveBayes

#df = pd.read_csv("../../TAN/data/chess.csv")
#col = 'ak'
df = pd.read_csv("/home/jonathan/BayesNets/NaiveBayesNets/data/pima.csv")
col = 'IsDiabetic'
n = df.shape[0]

np.random.seed(124)

ind = np.random.rand(n) < 0.75
traindf = df.loc[ind]
testdf = df.loc[~ind]

nbmodel = NaiveBayes(traindf, col)

testresults = nbmodel.Predict(testdf)
testresults[col] = testresults.idxmax(axis = 1)
accuracy = np.mean(testdf[col].values == testresults[col])
print(accuracy)
