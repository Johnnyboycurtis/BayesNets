import numpy as np
from scipy import stats

class Probs():

    def __init__(self, yseries, nonparemetric = False):
        """
        Class representing probabilities and kernel density estimation
        Takes a Pandas Series
        """
        dtype = str(yseries.dtype)
        self.dtype = dtype
        if dtype == "float64":
            self.kernel = Density(yseries, nonparemetric)
        else:
            self.probs = PMF(yseries)
        

    def __repr__(self):
        dtype = self.dtype
        if dtype == "float64":
            out =  "<Probs: kde>"
        else:
            out = "<Probs: P(u)>"
        return out
    
    def __str__(self):
        return "<Class Probs: P(u)>"

    
    def Evaluate(self, value):
        dtype = self.dtype
        if dtype == "float64":
            density = self.kernel[value]
        else:
            density = self.probs[value]
        return density
    


class PMF():
    def __init__(self, yseries):
        """
        Probability Mass Function for Discrete data
        """
        self.probs = self.CalcMarginalProbs(yseries.values)

    def __getitem__(self, value):
        notfound = 0.0001
        pval = self.probs.get(value, notfound)
        return pval
    
    def CalcMarginalProbs(self, array):
        n = array.shape[0]
        vals, counts = np.unique(array, return_counts=True)
        prop = counts/n ## proportions
        probs = dict(zip(vals,prop))
        return probs
    







class Density():

    def __init__(self, yseries, nonparemetric=False):
        """
        Density estimation for continuous data
        yseries: Pandas Series or DataFrame
        """
        self.nonparemetric = nonparemetric
        if nonparemetric:
            self.kernel = stats.gaussian_kde(yseries)
        else:
            self.mean = yseries.mean()
            self.sd = yseries.std()
    
    def __getitem__(self, value):
        if self.nonparemetric:
            return self.kernel(value)
        else:
            return self.dnorm(value)

    def dnorm(self, value):
        density = stats.norm.pdf(value, loc = self.mean, scale = self.sd)
        return density


