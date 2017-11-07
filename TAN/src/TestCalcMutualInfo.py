# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:45:13 2017

@author: jn107154
"""
import os
os.chdir(r"C:\Users\jn107154\BayesNets\TAN\src")
from CalcMutualInfo import *


def testCalcMutualInfo():
    x = list("aaaabbccdddd")
    y = list("222333444555")
    xprob = MarginalProb(x)
    yprob = MarginalProb(y)
    jointprob = testPairwiseCondProb(x,y)
    MI = CalcMutualInfo(xprob, yprob, jointprob)
    return MI


def testMarginalProb(val=None):
    """Test for MarginalProb
    Try: 
        test = testMarginalProb()
        sum(test.values()) ## should equal to 1.0
    """
    if val == None:
        x = list("aaaabbccdddd")
        result = MarginalProb(x)
    else:
        result = MarginalProb(val)
    return result
    
        
def testPairwiseCondProb(x, y):
    #x = list("aaaabbccdddd")
    #y = list("234562445536")
    result = PairWiseCondProb(x,y)
    return result


if __name__ == "__main__":
    #print("calculating mutual information")
    #mi = testCalcMutualInfo()
    #print(mi)
    
    print("starting to train")
    ## the following returns a dictionary
    ## a key for each class
    results = start_train2("small-test.csv", "Class")
    for i, frame in results.items():
        print(f"DataFrame: {i}")
        print(frame)
    
    