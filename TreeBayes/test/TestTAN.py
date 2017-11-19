# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:26:20 2017

@author: jn107154
"""

import sys
sys.path.append("../kde/")
import pandas as pd
from TAN import TAN
import numpy as np

## quick test ##
#print("starting to train Graph")
#pima = pd.read_csv("../data/pima.csv", dtype='str')
#class_col_name = "IsDiabetic"
pima = pd.read_csv("../data/Pima.tr.csv")
class_col_name = "type"
#pima = pd.read_csv("../data/chess.csv")
#class_col_name = "ak"
#pima = pd.read_csv("../data/train.csv") ## digit recognizer data
#class_col_name = "label"
n = pima.shape[0]
ind = np.random.rand(n) < 0.75
traindf = pima.loc[ind]
testdf = pima.loc[~ind]
tan = TAN(traindf, class_col_name) ## learns the tree structure
treebayes = tan.BuildModel(traindf) ## build the tree structure
results = treebayes.Predict(newdf = testdf)
accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
print(accuracy)





