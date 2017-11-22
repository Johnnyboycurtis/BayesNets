"""
@author: Jonathan Navarrete
"""

import sys
sys.path.append(r"C:\Users\jn107154\BayesNets\TreeBayes\kde")
#import os
#os.chdir("L:/APD/BayesNets/BayesNets/NaiveBayes/src/")
from TAN import KDETAN
import pandas as pd
import numpy as np

if __name__ == "__main__":
    pima = pd.read_csv("../data/pima.csv")
    class_col_name = "IsDiabetic"
    #pima = pd.read_csv("../data/Pima.tr.csv")
    #class_col_name = "type"
    #pima = pd.read_csv("../data/chess.csv")
    #class_col_name = "ak"
    #pima = pd.read_csv("../data/train.csv") ## digit recognizer data
    #class_col_name = "label"
    n = pima.shape[0]
    ind = np.random.rand(n) < 0.75
    traindf = pima.loc[ind]
    testdf = pima.loc[~ind]
    kdetreenb = KDETAN(traindf, class_col_name, maximum=True, progress_bar=True)
    model = kdetreenb.BuildModel(dataframe = traindf)
    results = model.Predict(newdf = testdf)
    results[class_col_name] = results.idxmax(axis = 1)
    accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
    print(accuracy)
