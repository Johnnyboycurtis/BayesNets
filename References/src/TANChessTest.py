# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:16:10 2017

@author: jn107154
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from TAN import TAN
import numpy as np

## quick test ##
#print("starting to train Graph")
class_col_name = "ak"
#traindf = pd.read_csv("../data/train-chess.csv")
#testdf = pd.read_csv("../data/test-chess.csv")
df = pd.read_csv("../data/chess.csv")
n = df.shape[0]

power = []
for x in tqdm(range(3000)):
    ind = np.random.rand(n) < 0.75
    traindf = df.loc[ind]
    testdf = df.loc[~ind]
    model = TAN(dataframe = traindf, class_col_name = class_col_name)
    
    results = model.Predict(testdf)
    accuracy = (testdf.ak.values == results.ak).mean()
    power.append(accuracy)
    print(f"TAN accuracy: {round(accuracy, 4)}")

answer = sum(power)/len(power)
print(f"final answer: {round(answer,4)}")

res = pd.DataFrame(power, columns = ["accuracy"])
res.hist(bins = 20)
plt.show()

with open("results.txt", "w+") as myfile:
    for line in power:
        myfile.write(f"{line}\n")

