"""
@author: Jonathan Navarrete
"""


import numpy as np
from scipy import stats
import itertools as it

class Probs():

    def __init__(self, useries, vseries=None):
        """
        Class representing probabilities
        Both ulist and vlist have to be lists
        """
        if vseries is not None:
            self.type = 'Bivariate'
            self.jointprobs = Bivariate(useries, vseries) ## order matters! has to come first!
            self.vlist = vseries.tolist() ## save a list of values for use in Mutual Info
            self.vprobs = Univariate(vseries) #self.MarginalProb(vseries)
        else:
            self.type = 'Univariate'
            #self.jointprobs = None
            #self.vprobs = None
        self.ulist = useries.tolist()
        self.uprobs = Univariate(useries)
        

    def __repr__(self):
        if self.type == 'Bivariate':
            return "<KDE: f(u), f(v), f(u,v)>"
        if self.type == 'Univariate':
            return "<KDE: f(u)>"
    


    def CalcMutualInfo(self):
        """
        Calculate Mutual Information statistic
        uprobs: dictionary of probabilities
        vprobs: dictionary of probabilities
        jointprobs: dictionary of probabilities
        """
        uprobs = self.uprobs
        vprobs = self.vprobs
        jointprobs = self.jointprobs
        MI = [] ## collect Mutual Information
        jointkeys = zip(self.ulist, self.vlist)
        for uval, vval in jointkeys:
            uprob = uprobs[uval]
            vprob = vprobs[vval]
            probxy = jointprobs[(uval, vval)]
            I = probxy * np.log(probxy / (uprob * vprob))
            MI.append(I)
        MI = np.sum(MI)
        return MI


    def ConditionalProb(self, u, v):
        """
        Calculate the Conditional Probability
        u,v: values from the distributions of U and V
        V: Parent
        U: Child
        Calculates: P(U = u | V = v) = P(U, V)/P(V)
        """
        vprobs = self.vprobs
        jointprobs = self.jointprobs
        try:
            PV = vprobs.Evaluate(v)
            PUV = jointprobs.Evaluate(u,v)
            condprob = PUV/PV ## P(U,V) / P(V)
        except KeyError:
            #print(f"{u},{v} were not found in joint probs. Returning 0.0001.")
            condprob = 0.0001
        return condprob

    def PredMarginalProb(self, u):
        pval = self.uprobs.Evaluate(u)
        return pval





class Bivariate():
    def __init__(self, xseries, yseries):
        """
        Expects two Pandas Series
        """
        values = np.vstack([xseries, yseries])
        self.probs = stats.gaussian_kde(values)
    
    def __getitem__(self, UV):
        return self.probs(UV)


    def Evaluate(self, u, v):
        KDE = self.probs
        pval = KDE((u,v))
        return pval

        

class Univariate():
    def __init__(self, series):
        """
        Expects a Pandas Series
        """
        self.probs = stats.gaussian_kde(series)
    
    def __getitem__(self, u):
        return self.probs(u)

    def Evaluate(self, u):
        KDE = self.probs
        pval = KDE(u)
        return pval

   
        

"""
import pandas as pd
n = 100
sample1 = pd.Series(np.random.rand(n))
sample2 = pd.Series(np.random.rand(n))
#print(sample1)
#print(sample2)

test = Probs(sample1, sample2)
pval = test.ConditionalProb(0.5, 0.3)
print(pval)

print(test)
print(test.jointprobs)
"""