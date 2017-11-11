#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:06:30 2017

@author: jonathan
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayes


df = pd.read_csv("../data/digits/train.csv")
n = df.shape[0]
ind = np.random.rand(n) < 0.75

traindf = df.loc[ind]
testdf = df.loc[~ind]


## build model
nbmodel = NaiveBayes(traindf, 'label', progress_bar=True)
## test model and get predictions
testresults = nbmodel.Predict(testdf, progress_bar=True)
## compare accuracy
accuracy = np.mean(testdf['label'].values == testresults['label'])
print(accuracy)



