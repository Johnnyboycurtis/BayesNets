"""
@author: Jonathan Navarrete
"""

import sys
sys.path.append("../kde/")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from TAN import TAN
import numpy as np

## quick test ##
#print("starting to train Graph")
df = pd.read_csv("../data/pima.csv")
class_col_name = "IsDiabetic"
#df = pd.read_csv("../data/Pima.tr.csv")
#class_col_name = "type"
n = df.shape[0]

power = []
for x in tqdm(range(1000)):
    ind = np.random.rand(n) < 0.75
    traindf = df.loc[ind]
    testdf = df.loc[~ind]
    tan = TAN(dataframe = traindf, class_col_name = class_col_name, progress_bar=False)
    treebayes = tan.BuildModel(traindf) ## build the tree structure
    results = treebayes.Predict(newdf = testdf)
    accuracy = (testdf[class_col_name].values == results[class_col_name]).mean()
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

