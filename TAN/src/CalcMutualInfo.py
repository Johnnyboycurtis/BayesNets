import pandas as pd
import itertools as it


def start_train(filename, sep = ",", class_col_name):
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
    """Calculate the Joint Probability Distribution """
    dat = zip(xlist,ylist) ## need to sort list; don't forget!!
    keyfunc = lambda line: line ## return itself; group by itself
    g = it.groupby(dat)
    probs = {}
    n = len(xlist) ## assumes len(xlist) == len(ylist)
    for key, val in g:
        vlen = len(list(val))
        probs[key] = vlen/n
    return probs ## returns dictionary



def CalcMutualInfo(xlist, ylist, jointprob):
    """Calculate Mutual Information statistic"""
    xgroup = it.groupby(xlist, lambda line: line)
    for xi, x in xgroup:









