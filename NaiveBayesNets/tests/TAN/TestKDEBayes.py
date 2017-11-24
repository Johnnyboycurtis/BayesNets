"""
@author: Jonathan Navarrete
"""


import sys
sys.path.append("/home/jonathan/BayesNets/")
import pandas as pd
from NaiveBayesNets import TAN
import numpy as np

## quick test ##
np.random.seed(678)
#print("starting to train Graph")
#pima = pd.read_csv("../data/pima.csv", dtype='str')
#class_col_name = "IsDiabetic"
pima = pd.read_csv("/home/jonathan/BayesNets/NaiveBayesNets/data/Pima.tr.csv")
class_col_name = "type"
#pima = pd.read_csv("../data/chess.csv")
#class_col_name = "ak"
#pima = pd.read_csv("../data/train.csv") ## digit recognizer data
#class_col_name = "label"
n = pima.shape[0]
ind = np.random.rand(n) < 0.75
traindf = pima.loc[ind]
testdf = pima.loc[~ind]
kdetan = TAN.KDEBayes(traindf, class_col_name) ## learns the tree structure
treebayes = kdetan.BuildModel(traindf) ## build the tree structure
results = treebayes.Predict(newdf = testdf)
accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
print(accuracy)





