# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:06:19 2017

@author: jn107154
"""
import sys
sys.path.append("L:/APD/BayesNets/BayesNets/NaiveBayes/src/")
import os
os.chdir("L:/APD/BayesNets/BayesNets/NaiveBayes/src/")
from NaiveBayes import NaiveBayes
import pandas as pd
import numpy as np

if __name__ == "__main__":
    #pima = pd.read_csv("../data/pima.csv", dtype='str')
    #class_col_name = "IsDiabetic"
    pima = pd.read_csv("../data/Pima.tr.csv")
    class_col_name = "type"
    #pima = pd.read_csv("../data/chess.csv")
    #class_col_name = "ak"
    n = pima.shape[0]
    ind = np.random.rand(n) < 0.75
    traindf = pima.loc[ind]
    testdf = pima.loc[~ind]
    nbmodel = NaiveBayes(traindf, class_col_name)
    results = nbmodel.Predict(newdf = testdf)
    results[class_col_name] = results.idxmax(axis = 1)
    accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
    print(accuracy)
