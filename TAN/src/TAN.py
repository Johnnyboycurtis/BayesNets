# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:28:01 2017

@author: jn107154
"""

#import os
#os.chdir(r"C:\Users\jn107154\BayesNets\TAN\src")
import pandas as pd
import itertools as it
import numpy as np
from Graph import Graph
from SimpleGraphPlot import draw_graph


class TAN(object):
    def __init__(self, dataframe, class_col_name):
        self.dataframe = dataframe
        self.class_col_name = class_col_name
        self.results = self.start_train2() ## a list of size 2
        self.MST = self.BuildMST()
        
    def start_train2(self):
        df = self.dataframe
        class_col_name = self.class_col_name
        g = df.groupby(by = class_col_name) ## group df by class
        colnames = df.columns.tolist()
        colnames.remove(class_col_name) ## remove class column; gets in the way
        ## process the following steps for each class
        ClassMats = {} ## dictionary to store MutualInfMatrix for each class
        for i, frame in g:
            colcombos = it.combinations(colnames, 2) ## will return tuples
            MutualInfo = []
            for x, y in colcombos:
                xlist = frame[x].tolist()
                ylist = frame[y].tolist()
                xprobs = self.MarginalProb(xlist)
                yprobs = self.MarginalProb(ylist)
                jointprobs = self.PairWiseCondProb(xlist, ylist)
                MI = self.CalcMutualInfo(xprobs, yprobs, jointprobs)
                MutualInfo.append([(x,y), MI])
            MutualInfMatrix = pd.DataFrame(MutualInfo, columns = ['Pairs', "MI"])
            MutualInfMatrix.sort_values(by = "MI", ascending=False, inplace=True)
            ClassMats[i] = MutualInfMatrix ## store results for current class
        return [colnames, ClassMats]
    
        
    def PairWiseCondProb(self, xlist, ylist):
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
    
    
    
    def MarginalProb(self, datlist):
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
    
    
    
    def CalcMutualInfo(self, xprobs, yprobs, jointprobs):
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
        
    def BuildMST(self, maximum=True):
        vertices, ClassFrames = self.results
        
        MST = {}
        for i, frame in ClassFrames.items():
            print(f"DataFrame: {i}")
            print(frame)
            graph2 = frame.Pairs.tolist()
            labs = [round(i, 3) for i in frame.MI.tolist()]
            draw_graph(graph2, labels = labs)

            ## Build MST
            g = Graph(vertices, [])  ## number of unique attributes
            for ind,pair,mi in frame.itertuples():
                u,v = pair
                g.addEdge(u,v,mi) 
            maxst = g.KruskalMST(maximum=maximum) ## return Maximum Spanning Tree
            MST[i] = maxst
        return MST
    

if __name__ == "__main__":
    ## quick test ##
    print("starting to train Graph")
    df = pd.read_csv("Pima.tr.csv")
    class_col_name = "type"
    model = TAN(dataframe = df, class_col_name = class_col_name)


