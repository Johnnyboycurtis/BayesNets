
import os
os.chdir(r"C:\Users\jn107154\BayesNets\TAN\src")
import pandas as pd
import itertools as it
import numpy as np


def start_train(filename, class_col_name, sep = ","):
    df = pd.read_csv(filename, sep = sep) ## first read in the data
    g = df.groupby(by = class_col_name) ## group df by class
    
    ## process the following steps for each class
    ClassMats = {} ## dictionary to store MutualInfMatrix for each class
    for i, frame in g:
        MutualInfMatrix = [] ## empty list to store pandas data frames
        ## drop class column; gets in the way
        frame.drop(labels = class_col_name, axis = 1, inplace=True) 
        
        ## doing things the long way for now...
        for j, xcol in frame.iteritems():
            MutualInfo = {}
            colnames = []
            for k, ycol in frame.iteritems():
                colnames.append(k)
                if j == k:
                    MutualInfo[k] = 0 ## mutual info of self = 0
                else:
                    xlist = xcol.tolist()
                    ylist = ycol.tolist()
                    xprobs = MarginalProb(xlist)
                    yprobs = MarginalProb(ylist)
                    jointprobs = PairWiseCondProb(xlist, ylist)
                    MI = CalcMutualInfo(xprobs, yprobs, jointprobs)
                    MutualInfo[k] = MI
            tmp = pd.DataFrame(data = MutualInfo, index = [j]) ## tmp dataframe
            MutualInfMatrix.append(tmp)
    
        MutualInfMatrix = pd.concat(objs = MutualInfMatrix) ## concat frames
        ClassMats[i] = MutualInfMatrix ## store results for current class
    return ClassMats
        



def start_train2(filename, class_col_name, sep = ","):
    df = pd.read_csv(filename, sep = sep) ## first read in the data
    g = df.groupby(by = class_col_name) ## group df by class
    colnames = df.columns.tolist()
    colnames.pop(colnames.index(class_col_name)) ## drop class column; gets in the way
    ## process the following steps for each class
    ClassMats = {} ## dictionary to store MutualInfMatrix for each class
    for i, frame in g:
        colcombos = it.combinations(colnames, 2) ## will return tuples
        MutualInfo = []
        for x, y in colcombos:
            xlist = frame[x].tolist()
            ylist = frame[y].tolist()
            xprobs = MarginalProb(xlist)
            yprobs = MarginalProb(ylist)
            jointprobs = PairWiseCondProb(xlist, ylist)
            MI = CalcMutualInfo(xprobs, yprobs, jointprobs)
            MutualInfo.append([(x,y), MI])
        MutualInfMatrix = pd.DataFrame(MutualInfo, columns = ['Pairs', "MI"])
        MutualInfMatrix.sort_values(by = "MI", ascending=False, inplace=True)
        ClassMats[i] = MutualInfMatrix ## store results for current class
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
    datlist.sort() ## sort values before grouping!
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
    




