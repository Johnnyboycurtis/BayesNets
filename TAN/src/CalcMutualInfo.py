
import os
os.chdir(r"C:\Users\jn107154\BayesNets\TAN\src")
import pandas as pd
import itertools as it
import numpy as np


def start_train(filename, class_col_name, sep = ","):
    df = pd.read_csv(filename, sep = sep)
    g = df.groupby(by = class_col_name)
    ClassMats = {}
    for i, frame in g:
        MutualInfMatrix = [] ## empty list to store pandas data frames
        frame.drop(labels = class_col_name, axis = 1, inplace=True) ## drop class column; gets in the way
        for j, xcol in frame.iteritems():
            MutualInfo = {}
            colnames = []
            for k, ycol in frame.iteritems():
                colnames.append(k)
                if j == k:
                    MutualInfo[k] = 0
                else:
                    xlist = xcol.tolist()
                    ylist = ycol.tolist()
                    xprobs = MarginalProb(xlist)
                    yprobs = MarginalProb(ylist)
                    jointprobs = PairWiseCondProb(xlist, ylist)
                    MI = CalcMutualInfo(xprobs, yprobs, jointprobs)
                    MutualInfo[k] = MI
            #print(MutualInfo)
            #print(f"index = {i} and colnames = {colnames}")
            tmp = pd.DataFrame(data = MutualInfo, index = [j])
            MutualInfMatrix.append(tmp)
        MutualInfMatrix = pd.concat(objs = MutualInfMatrix)
        ClassMats[i] = MutualInfMatrix
    return ClassMats
        


def PairWiseCondProb(xlist, ylist):
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

def MarginalProb(datlist):
    """
    Calculate the (Conditional) Marginal Probability
    Sorts values before doing any calculations
    """
    datlist.sort()
    g = it.groupby(datlist)
    probs = {}
    n = len(datlist)
    for key, val in g:
        vlen = len(list(val))
        probs[key] = vlen / n
    return probs


def CalcMutualInfo(xprobs, yprobs, jointprobs):
    """
    Calculate Mutual Information statistic
    xprobs: dictionary of probabilities
    yprobs: dictionary of probabilities
    jointprobs: dictionary of probabilities
    """
    MI = [] ## collect Mutual Information
    jointkeys = list(jointprobs.keys())
    for xval, yval in jointkeys:
        xprob = xprobs[xval]
        yprob = yprobs[yval]
        probxy = jointprobs[(xval, yval)]
        I = probxy * np.log(probxy / (xprob * yprob))
        #print(f"{probxy}*log({probxy} / ( {xprob}*{yprob})) +")
        MI.append(I)
    MI = np.sum(MI)
    return MI
    




