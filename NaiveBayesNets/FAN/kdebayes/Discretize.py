
import numpy as np
from collections import namedtuple

Bin = namedtuple('Bin', ['a', 'b', 'pval'])

def discretize(data, density = True):
    n = len(data)
    Counts, Breaks = np.histogram(data) ## len(Counts) < len(Breaks)
    Breaks = list(Breaks)
    prev = Breaks.pop(0)
    edges = list(zip(Counts, Breaks))
    results = []
    for count, nexxt in edges:
        if density:
            pval = count/n
        else:
            pval = count
        b = Bin(prev, nexxt, pval)
        results.append(b)
        prev = nexxt
    return results


class Discretize():

    def __init__(self, data):
        """
        Discretization is the process of converting continuous random variables
        to discrete random variables. This method uses a histogram (i.e. np.histogram)
        to bin values.

        This class returns the histogram bin `(a,b)` as discretized values.
        Additionally, you can use the method `density` to get the univariate density value.

        ## test data
        >>> np.random.seed(123)
        >>> x = np.random.rand(1000)

        >>> out = discretize(x)

        >>> test = Discretize(x)

        >>> print(test)

        >>> print(test[0.5])

        >>> print(test[0.1])
        """
        self.length = len(data)
        self.histogram = discretize(data)

    def density(self, key):
        hist = self.histogram
        for a, b, pval in hist:
            if a <= key and key < b:
                #print("found!:",pval)
                return pval
    
    def __getitem__(self, key):
        hist = self.histogram
        match = None
        for a, b, pval in hist:
            if a <= key and key < b:
                match = (a,b)
        if match == None:
            print(f"ERROR with matches!! {key}")
        return match
    


"""
## test data
np.random.seed(123)
x = np.random.rand(1000)

out = discretize(x)

test = Discretize(x)

print(test)

print(test[0.5])

print(test[0.1])
"""



