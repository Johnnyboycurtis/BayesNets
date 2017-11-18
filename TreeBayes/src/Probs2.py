
#import sys
#import pandas as pd
import numpy as np
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
            self.vprobs = Univariate(vseries) #self.MarginalProb(vseries)
        else:
            self.type = 'Univariate'
            #self.jointprobs = None
            #self.vprobs = None
        self.uprobs = Univariate(useries)
        

    def __repr__(self):
        if self.type == 'Bivariate':
            return "<Probs: P(u), P(v), P(u,v)>"
        if self.type == 'Univariate':
            return "<Probs: P(u)>"
    


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
        jointkeys = list(jointprobs.probs.keys())
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
        xlist = xseries.tolist()
        ylist = yseries.tolist()
        self.probs = self.JointProb(xlist, ylist)

    def JointProb(self, xlist, ylist):
        """
        Calculate the Joint Probability Distribution
        Sorts values before doing any calculations
        """
        dat = list(zip(xlist,ylist)) ## need to sort list; don't forget!!
        dat.sort(key=lambda line: line)
        #keyfunc = lambda line: line ## return itself; group by itself
        g = it.groupby(dat, lambda line: line)
        probs = {}
        n = len(xlist) ## assumes len(xlist) == len(ylist)
        for key, val in g:
            vlen = len(list(val))
            probs[key] = vlen/n
        return probs ## returns dictionary

    def __getitem__(self, uv):
        return self.probs.get(uv)

    def Evaluate(self, u, v):
        PMF = self.probs
        notfound = 0.0001 ## should this be np.nan?? not sure, yet..
        pval = PMF.get((u,v), notfound)
        return pval

        

class Univariate():
    def __init__(self, series):
        """
        Expects a Pandas Series
        """
        self.probs = self.CalcPMF(series)
    
    def __getitem__(self, u):
        return self.probs.get(u)

    def CalcPMF(self, series):
        n = series.shape[0]
        #counts = dataframe[class_col_name].value_counts()
        counts = series.value_counts()
        priors = (counts / n).to_dict()
        return priors

    def Evaluate(self, u):
        PMF = self.probs
        notfound = 0.0001
        pval = PMF.get(u, notfound)
        return pval

   
        
    

"""
sample1 = pd.Series(list("aaabbbccccccddddeeeeffhhhhhgggwww"))
sample2 = pd.Series(list("zzzzzzzccwwwccddddeeeeffwwwhhgggwww"))
print(sample1)
print(sample2)

test = Probs(sample1, sample2)
pval = test.ConditionalProb('w', 'z')
print(pval)

print(test)
print(test.jointprobs)
"""