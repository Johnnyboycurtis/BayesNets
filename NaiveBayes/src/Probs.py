import numpy as np
from scipy import stats

class Probs():
    """Class representing probabilities and kernel density estimation"""

    def __init__(self, yseries, nonparemetric = True):
        """Takes a Pandas Series"""
        dtype = str(yseries.dtype)
        self.dtype = dtype
        if dtype == "float64":
            if nonparemetric:
                self.kernel = stats.gaussian_kde(yseries)
            else:
                self.kernel = self.NormalDensity(yseries)
        else:
            self.probs = self.CalcMarginalProbs(yseries)
        

    def __repr__(self):
        if dtype == "float64":
            out =  "<Probs: Kernel Density Estimator>"
        else:
            out = "<Probs: P(u)>"
        return out
    
    def __str__(self):
        return "<Class Probs: P(u)>"

    def NormalDensity(self, yseries):
        m = yseries.mean()
        sd = yseries.std()
        kernel = lambda x: dnorm(x, m, sd)
        return kernel

    def Evaluate(self, value):
        dtype = self.dtype
        if dtype == "float64":
            density = self.kernel(value) 
        else:
            try:
                density = self.probs[value]
            except KeyError:
                density = 0.0001 ## default value if value not found during training
        return density
    
    def CalcMarginalProbs(self, yseries):
        dtype = str(yseries.dtype)
        vals = yseries.value_counts()
        n = yseries.shape[0]
        probs = (vals / n).to_dict() ## series to dictionary
        return probs


def dnorm(x, mean, sd):
    """
    R-like dnorm function
    """
    density = stats.norm.pdf(x, loc = mean, scale = sd)
    return density