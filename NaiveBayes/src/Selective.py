## Selective Naive Bayes




import os
os.chdir("C:/Users/jn107154/BayesNets/NaiveBayes/src")
from tqdm import tqdm
import pandas as pd
from NaiveBayes import NaiveBayes
import numpy as np



class SelectiveNB():
    def __init__(self, dataframe, class_col_name):
        self.class_col_name = class_col_name
        self.Model = self.Step(dataframe, class_col_name)
    
    def Build(self, dataframe):
        df = dataframe
        traincols = self.Model[1] ## (accuracy, list, length)
        nbmodel = NaiveBayes(DF = df, class_col_name=self.class_col_name, priors=None, progress_bar=False)
        return nbmodel

    def Step(self, dataframe, class_col_name):
        df = dataframe
        n = df.shape[0]
        ind = np.random.rand(n) < 0.75
        traindf = df.loc[ind]
        testdf = df.loc[~ind]

        tests = []
        excluded = None
        pred = traindf.columns.tolist()
        pred.remove(class_col_name)
        m = len(pred)

        for i in range(m):
            traincols = pred + [class_col_name]
            nbmodel = NaiveBayes(traindf[traincols], class_col_name = class_col_name, progress_bar=False)
            results = nbmodel.Predict(newdf = testdf, response=True)
            accuracy = (testdf[class_col_name].values == results[class_col_name]).mean()
            predS = " + \n\t".join(pred)
            print(f"Current Model: {self.class_col_name} ~ {predS}")
            print(f"Accuracy: {round(accuracy, 4)}")
            print("________________________________")
            excluded = pred.pop()
            tests.append((accuracy, list(pred), len(pred)))

        tests.sort(key = lambda x: x[0], reverse=True)
        BestModel = tests[0]
        return BestModel

