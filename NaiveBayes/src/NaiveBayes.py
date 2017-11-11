#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:50:32 2017

@author: Jonathan Navarrete
"""

import numpy as np
from scipy import stats    
import pandas as pd

class NaiveBayes():
  
    def __init__(self, DF, class_col_name, priors = None):
        """
        Naive Bayes Classifier
        DF: Pandas DataFrame
        class_col_name: column name with outcome variable
        priors: dictionary of prior probabilities for each class
        Will not store dataframe as part of results
        """
        self.class_col_name = class_col_name
        self.n = DF.shape[0]
        
        if priors:
            self.priors = priors
        else:
            self.priors = CalcMarginalProbs(DF[class_col_name]) ## returns dictionary of priors
        self.classes = DF[class_col_name].unique().tolist()
        self.CondProbs = self.MarginalProbs(DF, class_col_name)
        
    
    def __repr__(self):
        priors = [f"{i}: {round(j, 4)}" for i,j in self.priors.items()]
        plines = "\t".join(priors)
        col = self.class_col_name
        out = f"""
        Naive Bayes Model for {col}
        Priors:
            {plines}
        """
        return out
    
    def MarginalProbs(self, DF, class_col_name):
        """
        For each class:
            for each column:
                calculate the Marginal Probabilities
        Return everything as a dictionary
        """
        g = DF.groupby(by = class_col_name) ## group df by class
        ## process the following steps for each class
        ClassMats = {} ## dictionary to store MutualInfMatrix for each class
        for klass, frame in g:
            Probs = {} 
            for col, yseries in frame.items():
                p = CalcMarginalProbs(yseries) ## returns dictionary
                Probs[col] = p ## dictionary in dictionary
            ClassMats[klass] = Probs ## dict in dict
        return ClassMats
        
        
    def Predict(self, newdf, logProbs = False):
        """
        Takes a new dataframe of values to predict on
        Then for each covariate used to train Naive Bayes, it will iterate
        through each row and calculate probabilities for each class
        """
        models = self.CondProbs ## dictionary {class: Probs}
        predictions = []
        class_col_name = self.class_col_name
        for index, row in newdf.iterrows():
            results = {}
            newdata = row.to_dict() ## {col: value, col: value, ...}
            for klass, ProbsDict in models.items():
                prior = self.priors[klass]
                logPVals = [np.log(prior)]
                for col, MargProbs in ProbsDict.items():
                    val = newdata[col] ## extract testing data
                    try:
                        p = MargProbs[val]
                    except KeyError:
                        p = 0.0001
                    logp = np.log(p)
                    logPVals.append(logp)
                if logProbs:
                    pval = sum(logPVals)
                    results[klass] = pval
                else:
                    pval = np.exp(sum(logPVals))
                    results[klass] = pval
            predictions.append(results)
        predDF = pd.DataFrame(predictions)
        predDF[class_col_name] = predDF.idxmax(axis = 1).values
        return predDF



def CalcMarginalProbs(yseries):
    vals = yseries.value_counts()
    n = yseries.shape[0]
    apriori = (vals / n).to_dict() ## series to dictionary
    return apriori


    








## end of file ## 