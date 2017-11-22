"""
@author: Jonathan Navarrete
"""

from tqdm import tqdm
import numpy as np
from .Probs2 import Probs
import pandas as pd

class NaiveBayes():
  
    def __init__(self, DF, class_col_name, priors = None, progress_bar=False):
        """
        Naive Bayes Classifier
        DF: Pandas DataFrame
        class_col_name: column name with outcome variable
        priors: dictionary of prior probabilities for each class
        Will not store dataframe as part of results
        Calculates:
            P(Class | data) \propto P(Class)*P(data | Class)
        This model ignores the denominator; 
        Denominator is a scaling factor to be ignored.
        """
        self.class_col_name = class_col_name
        self.n = DF.shape[0]
        
        if priors:
            self.priors = priors
        else:
            self.priors = Probs(DF[class_col_name]) ## returns dictionary of priors
        self.classes = DF[class_col_name].unique().tolist()
        self.CondProbs = self.TrainNB(DF, class_col_name, progress_bar)

    
    def TrainNB(self, DF, class_col_name, progress_bar=False):
        """
        For each class:
            for each column:
                calculate the Marginal Probabilities
        Return everything as a dictionary
        """
        g = DF.groupby(by = class_col_name) ## group df by class
        if progress_bar:
            g = tqdm(g)
        ## process the following steps for each class
        ClassMats = {} ## dictionary to store MutualInfMatrix for each class
        for klass, frame in g:
            frame = frame.drop(labels = class_col_name, axis = 1)
            Densities = {} 
            for col, yseries in frame.items():
                p = Probs(yseries) ## returns dictionary or kde
                Densities[col] = p ## dictionary in dictionary
            ClassMats[klass] = Densities ## dict in dict
        return ClassMats
        

    def Predict(self, newdf, logProbs = False, progress_bar=False, response=False):
        """
        Takes a new dataframe of values to predict on
        Then for each covariate used to train Naive Bayes, it will iterate
        through each row and calculate (un-normalized) probabilities for each class

        If `response = True` then appends a column with predicted class.
        
        Returns posterior probabilities which can then be normalized into
        probabilities that sum to 1. Example:
            P(X = x | Y = y1) = 0.0025
            P(X = x | Y = y2) = 0.001
        can be normalized by denominator (0.0025+0.001) 0.0035 to give
            P*(X = x | Y = y1) = 0.7142857
            P*(X = x | Y = y2) = 0.2857143
        """
        models = self.CondProbs ## dictionary {class: Probs}
        predictions = []
        class_col_name = self.class_col_name
        rows = newdf.iterrows()
        if progress_bar:
            rows = tqdm(rows)
        for index, row in rows:
            results = {}
            newdata = row.to_dict() ## {col: value, col: value, ...}
            for klass, ProbsDict in models.items():
                prior = self.priors.Evaluate(klass)
                logPVals = [np.log(prior)]
                for col, MargProbs in ProbsDict.items():
                    val = newdata[col] ## extract testing data
                    p = MargProbs.Evaluate(val)
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
        if response:
            predDF[class_col_name] = predDF.idxmax(axis = 1).values
        return predDF





## end of file ## 
