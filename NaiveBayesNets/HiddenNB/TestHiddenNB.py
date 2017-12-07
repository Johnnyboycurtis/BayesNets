"""
@author: Jonathan Navarrete
"""


import sys
#sys.path.append("C:/Users/jn107154/BayesNets/NaiveBayesNets/FAN/treenb")
sys.path.append("C:/Users/jn107154/BayesNets/NaiveBayesNets/HiddenNB")
import pandas as pd
import HiddenNB as h
import numpy as np

## quick test ##
#np.random.seed(678)
#df = pd.read_csv("C:/Users/jn107154/BayesNets/NaiveBayesNets/data/chess.csv")
#class_col_name = "ak"
#df = pd.read_csv("C:/Users/jn107154/BayesNets/NaiveBayesNets/data/balance-scale.csv")
#class_col_name = "balance"
df = pd.read_csv("C:/Users/jn107154/BayesNets/NaiveBayesNets/data/Pima.tr.csv")
class_col_name = "type"
n = df.shape[0]
ind = np.random.rand(n) < 0.75
traindf = df.loc[ind]
testdf = df.loc[~ind]
discretetan = h.KDEBayes(traindf, class_col_name)
#treebayes = discretetan.BuildModel(traindf) ## build the tree structure
#results = treebayes.Predict(newdf = testdf)
#accuracy = np.mean(results[class_col_name] == testdf[class_col_name].values)
#print(accuracy)



#piv = df.pivot_table(values = 'bmi', columns = 'type', index = 'npreg', aggfunc='count')
#piv / 200

        
        

