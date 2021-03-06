## Selective Naive Bayes

#import os
#os.chdir("C:/Users/jn107154/BayesNets/NaiveBayes/src")
from tqdm import tqdm
import pandas as pd
from .NaiveBayes import NaiveBayes
import numpy as np



class SelectiveNB():
    def __init__(self, dataframe, class_col_name, xclass, init_col=None):
        self.class_col_name = class_col_name
        self.Model = self.Step(dataframe, class_col_name, xclass, init_col)
    
    def Build(self, dataframe):
        df = dataframe
        traincols = self.Model ## list of columns including class_col_name
        nbmodel = NaiveBayes(DF = df[traincols], class_col_name=self.class_col_name, priors=None, progress_bar=False)
        return nbmodel

    def Step(self, dataframe, class_col_name, xclass, init_cols=None):
        df = dataframe
        n = df.shape[0]
        ind = np.random.rand(n) < 0.75
        traindf = df.loc[ind]
        testdf = df.loc[~ind]

        cand = traindf.columns.tolist() ## candidates
        cand.remove(class_col_name)
        
        if init_cols is not None:
            [col for col in init_cols if col not in init_cols]
            pred = init_cols
        else:
            pred = []
        m = len(cand)
        prev_accuracy = 0.01

        for i in range(m):
            next_cols = [x for x in cand if x not in pred] ## make copy
            tests = []
            for col in tqdm(next_cols):
                traincols = [col] + pred  + [class_col_name]
                nbmodel = NaiveBayes(traindf[traincols], class_col_name = class_col_name, progress_bar=False)
                results = nbmodel.Predict(newdf = testdf, response=False)
                testdf['Prediction'] = results.idxmax(axis = 1).values
                yesdf = testdf.loc[testdf[class_col_name] == xclass]
                curr_accuracy = (yesdf['Prediction'] == yesdf[class_col_name].values).mean()
                if curr_accuracy > prev_accuracy:
                    tests.append((curr_accuracy, list(traincols), col))
            if len(tests) > 0:
                tests.sort(key = lambda x: x[0], reverse=True)
                BestModel = tests[0]
                curr_accuracy, curr_pred, curr_col = BestModel
                curr_pred.remove(class_col_name)
                prev_accuracy = curr_accuracy
                pred = curr_pred
                predS = " + \n\t".join(pred)
                print(f"\nCurrent Model: {self.class_col_name} ~ {predS}")
                print(f"Accuracy: {round(prev_accuracy, 4)}")
                print("--------------------------------")
            else:
                print("No Additional Improvements Made\nAlgorithm should STOP!!")
                print(f"Predictors {pred} has accuracy: {round(prev_accuracy, 4)}")
                break
            #ind = np.random.rand(n) < 0.75
            #traindf = df.loc[ind]
            #testdf = df.loc[~ind]
        final = pred + [class_col_name]
        return final







class KFoldNB():
    def __init__(self, dataframe, class_col_name, xclass, k = 5, init_cols=None):
        self.class_col_name = class_col_name
        self.Model = self.Step(dataframe, class_col_name, xclass, k = k, init_cols = init_cols)
    
    def Build(self, dataframe):
        df = dataframe
        traincols = self.Model ## list of columns including class_col_name
        nbmodel = NaiveBayes(DF = df[traincols], class_col_name=self.class_col_name, priors=None, progress_bar=False)
        return nbmodel

    def Step(self, dataframe, class_col_name, xclass, k = 5, init_cols=None,):
        df = dataframe
        list_frames = np.array_split(df, k)
        
        cand = df.columns.tolist() ## candidates
        cand.remove(class_col_name)
        
        if init_cols is not None:
            [col for col in init_cols if col not in init_cols]
            pred = init_cols
        else:
            pred = []
        m = len(cand)
        prev_accuracy = 0.01

        for i in range(m):
            next_cols = [x for x in cand if x not in pred] ## make copy
            tests = []
            for col in tqdm(next_cols):
                traincols = [col] + pred  + [class_col_name]
                accuracy = []
                for j in range(k):
                    train = [xdf for e, xdf in enumerate(list_frames) if j != k]
                    traindf = pd.concat(objs = train, axis = 0)
                    testdf = list_frames[j]
                    nbmodel = NaiveBayes(traindf[traincols], class_col_name = class_col_name, progress_bar=False)
                    results = nbmodel.Predict(newdf = testdf, response=False)
                    testdf['Prediction'] = results.idxmax(axis = 1).values
                    yesdf = testdf.loc[testdf[class_col_name] == xclass]
                    pred_avg = (yesdf['Prediction'] == yesdf[class_col_name].values).mean()
                    accuracy.append(pred_avg)
                    curr_accuracy = np.mean(accuracy)
                if curr_accuracy > prev_accuracy:
                    tests.append((curr_accuracy, list(traincols), col))
            if len(tests) > 0:
                tests.sort(key = lambda x: x[0], reverse=True)
                BestModel = tests[0]
                curr_accuracy, curr_pred, curr_col = BestModel
                curr_pred.remove(class_col_name)
                prev_accuracy = curr_accuracy
                pred = curr_pred
                predS = " + \n\t".join(pred)
                print(f"\nCurrent Model: {self.class_col_name} ~ {predS}")
                print(f"Accuracy: {round(prev_accuracy, 4)}")
                print("--------------------------------")
            else:
                print("No Additional Improvements Made\nAlgorithm should STOP!!")
                print(f"Predictors {pred} has accuracy: {round(prev_accuracy, 4)}")
                break
        final = pred + [class_col_name]
        return final







