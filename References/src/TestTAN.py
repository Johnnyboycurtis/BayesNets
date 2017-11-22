# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:26:20 2017

@author: jn107154
"""
import pandas as pd
from TAN import TAN
import numpy as np

## quick test ##
#print("starting to train Graph")
df = pd.read_csv("../data/Pima.tr.csv")
n = df.shape[0]
ind = np.random.rand(n) < 0.7
df['bmi'] = df.bmi.apply(int) ## convert the float to integer
class_col_name = "type"
traindf = df.loc[ind]
testdf = df.loc[~ind]
model = TAN(dataframe = traindf, class_col_name = class_col_name)
#myG  = toDiGraph(model.MST['No'])
#test = find_root(myG, "age")

results = model.Predict(testdf)
accuracy = (testdf.type.values == results.type).mean()
print(f"TAN accuracy: {round(accuracy, 4)}")






## quick test ##
#print("starting to train Graph")
class_col_name = "IsDiabetic"
traindf = pd.read_csv("../data/pima-train.csv")
testdf = pd.read_csv("../data/pima-test.csv")
model = TAN(dataframe = traindf, class_col_name = class_col_name)

results = model.Predict(testdf)
accuracy = (testdf.IsDiabetic.values == results.IsDiabetic).mean()
print(f"TAN accuracy: {round(accuracy, 4)}")




