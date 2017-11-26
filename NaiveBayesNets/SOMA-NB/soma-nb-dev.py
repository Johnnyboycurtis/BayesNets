#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:53:24 2017

@author: jonathan
"""

import os
os.chdir("/home/jonathan/BayesNets/NaiveBayesNets/")
import numpy as np
import pandas as pd
from scipy import stats
import patsy as p


df = pd.read_csv("data/Pima.tr.csv")

col = 'type' ## class column

ydf, xdf = p.dmatrices("type ~ npreg + glu + bp + skin + bmi + age - 1", data = df, return_type='dataframe')

m = ydf.shape[1]
Pearson = {}
for xclass, yseries in ydf.iteritems():
    corr = []
    for xname, xseries in xdf.iteritems():
        rho = yseries.corr(other = xseries, method = 'pearson')
        rho = round(rho, 4)
        corr.append((xclass, xname, rho))
        corr.sort(key = lambda x: x[2], reverse=True) ## sorts in place
    Pearson[xclass] = corr
    

tree = {} ## records (parent, child)

for xclass, corr in Pearson.items():
    first, second = corr[:2]
    tree[xclass] = (first[1], second[1])

print(tree)



## looks like the algorithm can only really work with 2 features: Yes-No
## since we're only interested in predicting YES, we'll use YES as C.

tree = ['bp', 'skin']

newcol = ydf.columns[1]
newdf = pd.concat([xdf, ydf[newcol]], axis = 1)
newdf.drop(labels = tree, inplace = True, axis = 1)






    