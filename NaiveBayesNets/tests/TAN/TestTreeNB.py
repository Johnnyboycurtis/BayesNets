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
pima = pd.read_csv("/home/jonathan/BayesNets/NaiveBayesNets/data/chess.csv")
class_col_name = "ak"
n = pima.shape[0]
ind = np.random.rand(n) < 0.75
traindf = pima.loc[ind]
testdf = pima.loc[~ind]
discretetan = TAN.TreeNB(traindf, class_col_name) ## learns the tree structure
treebayes = discretetan.BuildModel(traindf) ## build the tree structure
results = treebayes.Predict(newdf = testdf)
accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
print(accuracy)





