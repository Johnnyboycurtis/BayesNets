

import itertools as it

class Probs(object):
    """Class representing probabilities"""

    def __init__(self, xlist, ylist=None):
        """Both xlist and ylist have to be lists"""
        if ylist:
            self.probs = self.PairWiseCondProb(xlist, ylist)
        else:
            self.probs = self.MarginalProb(xlist)

    def __repr__(self):
        return "<Dict of Probs>"
    
    def __str__(self):
        return "<Dict of Probs>"
    
    def __getitem__(self, key):
        return self.probs[key]

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


#sample = list("aaabbbccccccddddeeeeffhhhhhgggwww")

#test = Probs(sample)

#print(test)