
import numpy as np
from collections import namedtuple

Bin = namedtuple('Bin', ['a', 'b', 'pval'])

def discretize(data, density = True):
    n = len(data)
    hist = np.histogram(data)
    edges = list(zip(*hist))
    edges.sort(key = lambda line: line[1], reverse=False) ## ascending order
    count, prev = edges.pop(0)
    results = []
    for count, nexxt in edges:
        if density:
            pval = count/n
        else:
            pval = count
        b = Bin(prev, nexxt, pval)
        results.append(b)
        print(b)
        prev = nexxt
    return results


class Discretize():

    def __init__(self, data):
        self.length = len(data)
        self.histogram = discretize(data)
    
    def __getitem__(self, key):
        hist = self.histogram
        for a,b, pval in hist:
            if a <= key and key < b:
                #print("found!:",pval)
                return pval


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



