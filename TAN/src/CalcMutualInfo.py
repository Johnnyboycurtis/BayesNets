import pandas as pd
import itertools as it
import numpy as np


def start_train(filename, class_col_name, sep = ","):
    df = pd.read_csv(filename, sep = sep)
    g = df.groupby(by = class_col_name)

    for i, frame in g:
        frame.drop(labels = class_col_name, axis = 1, inplace=True) ## drop class column; gets in the way
        for j, xcol in frame.iteritems():
            for k, ycol in frame.iteritems():
                if j != k:
                    xlist = xcol.tolist()
                    ylist = ycol.tolist()
                    jointprob = PairWiseCondProb(xlist, ylist)
                    MI = CalcMutualInfo(xlist, ylist, jointprob)




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
        print(f"{probxy}*log({probxy} / ( {xprob}*{yprob})) +")
        MI.append(I)
    #MI = np.sum(MI)
    return MI
    




