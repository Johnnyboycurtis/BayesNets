#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:50:32 2017

@author: Jonathan Navarrete
"""

import numpy as np
from scipy import stats    
from itertools import groupby
import pandas as pd

class NaiveBayes():
  
    def __init__(self, y, X):
        """
        takes in a y as a pandas series -- it is preferred if y values are strings vs integers
        takes in a pandas series or data frame X to serve as inputs
        """
        #X.index = map(str, y)
        self.y = y
        self.X = X
        self.apriori = self.calc_apriori()
        self.classes = np.unique(y)
        self.conditioanlProbs = X.groupby(by = X.index).apply(conditional_prob)
        
    def calc_apriori(self):
        vals = np.unique(self.y)
        n = self.y.shape[0]
        if isinstance(y, pd.DataFrame):
            prop = y.groupby(by = y.columns[0]).apply(lambda s: s.count()) / y.shape[0]
        elif isinstance(y, pd.Series):
            prop = y.groupby(by = y).count() / y.shape[0]
        else:
            prop = np.bincount(y) / (n * 1.0)
        
        apriori = dict(zip(vals, prop))
        
        return apriori
     
        
    def predict(self, newdf):
        models = self.conditioanlProbs
        ans = {}
        apriori = self.apriori
        for m in models.iterrows():
            results = [] ## list of dataframes
            for col in newdf.iteritems():
                params = m[1][col[0]] ## m is a tuble (index, dictionary)
                s = col[1]
                res = s.map(lambda x: NormalPDF(x, params['mu'], params['sd']))
                results.append(res)
            ansDF = pd.concat(results, 1)
            print(m[0])
            print( apriori[m[0]])
            vals = np.log(apriori[m[0]]) + ansDF.apply(lambda x: np.sum(np.log(x)), 1)
            print(vals)
            ans.update({m[0]: vals})         
            
            df = pd.DataFrame(ans)
            df = df.applymap(np.exp)
            denom = df.apply(np.sum, 1)
        return df.div(denom, axis = 'index')
        ## use df.idxmax(axis = 1) to get the label
                
       
def conditional_prob(df):
    def myfun(x):
        if x.dtype == float:              
            """ assumes a normal distribution on the data """        
            mu = np.mean(x)
            sd = np.std(x)
            return {'mu': mu, 'sd': sd}
        else:
            """ 
            if data is discrete (i.e. a factor/levels) then 
            it creates a contingency table
            """
            counts = x.shape[0]
            return counts
    return df.apply(myfun)

    

def NormalPDF(x, mu, sd):
    return stats.norm.pdf(x, mu, sd)

def anotherFun(x, d):
    mu = d['mu']
    sd = d['sd']
    return NormalPDF(x, mu, sd)


    
    
z = np.random.normal(2, 3, (100, 3))

y = np.random.randint(0, 2, 100)

df = pd.DataFrame(z, columns = list('abc'))

#df = pd.read_csv("df.csv", index_col = 0)
#c = NaiveBayes(df.index, df)

df = pd.read_csv("../data/iris.csv", index_col = 0)
y = df['Species']
df = df[[u'Sepal.Length', u'Sepal.Width', u'Petal.Length', u'Petal.Width']]
df.index = y

c = NaiveBayes(y, df)



predictions = c.predict(newdf=df)
print( predictions)
print( "classification")
print( predictions.idxmax(axis = 1))

correctPredictions = predictions.idxmax(axis = 1) == predictions.index
sum(correctPredictions) / 150.0















## end of file ## 