"""
@author: Jonathan Navarrete
"""


import sys
sys.path.append("C:/Users/jn107154/BayesNets/")
import pandas as pd
from NaiveBayesNets import TAN
import numpy as np

## quick test ##
#np.random.seed(678)
#df = pd.read_csv("C:/Users/jn107154/BayesNets/NaiveBayesNets/data/chess.csv")
#class_col_name = "ak"
df = pd.read_csv("C:/Users/jn107154/BayesNets/NaiveBayesNets/data/balance-scale.csv")
class_col_name = "balance"
n = df.shape[0]
ind = np.random.rand(n) < 0.75
traindf = df.loc[ind]
testdf = df.loc[~ind]
discretetan = TAN.TreeNB(traindf, class_col_name) ## learns the tree structure
treebayes = discretetan.BuildModel(traindf) ## build the tree structure
results = treebayes.Predict(newdf = testdf)
accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
print(accuracy)
