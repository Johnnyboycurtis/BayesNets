import numpy as np
from scipy import stats

class Probs():
    """Class representing probabilities and kernel density estimation"""

    def __init__(self, yseries, nonparemetric = True):
        """Takes a Pandas Series"""
        dtype = str(yseries.dtype)
        self.dtype = dtype
        if dtype == "float64":
            self.nonparemetric = nonparemetric
            if nonparemetric:
                self.kernel = stats.gaussian_kde(yseries)
            else:
                self.kernel = NormalPDF(yseries)
        else:
            self.probs = self.CalcMarginalProbs(yseries)
        

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
            if self.nonparemetric:
                density = self.kernel(value) 
            else:
                self.kernel.dnorm(value)
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

class NormalPDF():
    """
    R-like dnorm function
    """
    def __init__(self, yseries):
        """
        yseries: Pandas Series or DataFrame
        """
        self.mean = yseries.mean()
        self.sd = yseries.std()
    
    def dnorm(self, value):
        density = stats.norm.pdf(value, loc = self.mean, scale = self.sd)
        return density