
import numpy as np
import itertools as it

class Probs():
    """Class representing probabilities"""

    def __init__(self, ulist, vlist):
        """Both ulist and vlist have to be lists"""
        self.uprobs = self.MarginalProb(ulist)
        self.vprobs = self.MarginalProb(vlist)
        self.jointprobs = self.PairWiseCondProb(ulist, vlist)
        self.reversed = False ## a flag checking if probs have been reversed

    def __repr__(self):
        return "<Class Probs: P(u), P(v), P(u,v)>"
    
    def __str__(self):
        if self.reversed:
            out = "<P(v), P(u), P(v,u)>"
        else:
            out = "<P(u), P(v), P(u,v)>"
        return out
    
    def __getitem__(self, key):
        return self.probs[key]

    def __reverse__(self):
        """
        Graph module will sometimes reverse u,v --> v,u
        Thus, I need to account for this change by
        reversing positions of probabilities
        """
        uprobs = self.uprobs
        vprobs = self.vprobs
        ## switch
        self.uprobs = vprobs 
        self.vprobs = uprobs
        jointprobs = {}
        for key, val in self.jointprobs.items():
            u,v = key
            newkey = (v,u)
            jointprobs[newkey] = val ## same val; switched key cuz Graph.py!
        self.jointprobs = jointprobs
        self.reversed = True
    
    def keys(self):
        return self.probs.keys()

    def PairWiseCondProb(self, xlist, ylist):
        """
        Calculate the Joint Probability Distribution
        Sorts values before doing any calculations
        """
        dat = list(zip(xlist,ylist)) ## need to sort list; don't forget!!
        dat.sort()
        #keyfunc = lambda line: line ## return itself; group by itself
        g = it.groupby(dat)
        probs = {}
        n = len(xlist) ## assumes len(xlist) == len(ylist)
        for key, val in g:
            vlen = len(list(val))
            probs[key] = vlen/n
        return probs ## returns dictionary
    
    
    
    def MarginalProb(self, datlist):
        """
        Calculate the (Conditional) Marginal Probability
        Sorts values before doing any calculations
        """
        datlist.sort() ## sort values before grouping!
        g = it.groupby(datlist)
        probs = {}
        n = len(datlist)
        for key, val in g:
            vlen = len(list(val))
            probs[key] = vlen / n
        return probs



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
        jointkeys = list(jointprobs.keys())
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
        Calculates: P(V = v | U = u) = P(V,U)/P(U)
        """
        vprobs = self.vprobs
        jointprobs = self.jointprobs
        try:
            PV = vprobs[v]
            PUV = jointprobs[(u,v)]
            condprob = PUV/PV ## P(U,V) / P(V)
        except KeyError:
            condprob = 0
        return condprob
            
            
        
        
        
    


sample1 = list("aaabbbccccccddddeeeeffhhhhhgggwww")
sample2 = list("aaabzzzccwwwccddddeeeeffwwwhhgggwww")

test = Probs(sample1, sample2)
test.ConditionalProb('w', 'z')

print(test)