"""
@author: Jonathan Navarrete
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
#from NaiveBayes import NaiveBayeis
from TAN import TAN


df = pd.read_csv("../../NaiveBayes/data/digits/train.csv")
n = df.shape[0]
ind = np.random.rand(n) < 0.75

traindf = df.loc[ind]
testdf = df.loc[~ind]


## build model
TANmodel = TAN(traindf, 'label', progress_bar=True)
## test model and get predictions
testresults = TANmodel.Predict(testdf, progress_bar=True)
## compare accuracy
accuracy = np.mean(testdf['label'].values == testresults['label'])
print(accuracy)



