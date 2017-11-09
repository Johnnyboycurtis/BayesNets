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
from Probs import Probs
from Graph import Graph
from SimpleGraphPlot import draw_graph
from Plot import PlotDiGraph, PlotNetwork


class TAN(object):
    def __init__(self, dataframe, class_col_name, maximum=True):
        self.dataframe = dataframe
        self.class_col_name = class_col_name
        self.results = self.Train() ## a list of size 2
        self.MST = self.BuildMST()
        self.colnames = dataframe.columns.tolist().remove(class_col_name)
        
    def Train(self):
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
                xprobs = Probs(xlist) ## will calculate marginal probs
                yprobs = Probs(ylist)
                jointprobs = Probs(xlist, ylist)
                MI = self.CalcMutualInfo(xprobs, yprobs, jointprobs)
                MutualInfo.append([(x,y), MI, xprobs, yprobs, jointprobs])
            MutualInfMatrix = pd.DataFrame(MutualInfo, columns = ['Pairs', "MI", "P(u)","P(v)", "P(u,v)"])
            MutualInfMatrix.sort_values(by = "MI", ascending=False, inplace=True)
            ClassMats[i] = MutualInfMatrix ## store results for current class
        return [colnames, ClassMats]
    
    
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
            print(f"Class: {i}")
            print(frame)
            graph2 = frame.Pairs.tolist()
            labs = [round(i, 3) for i in frame.MI.tolist()]
            PlotNetwork(graph2, labels = labs)

            ## Build MST
            g = Graph(vertices, [])  ## number of unique attributes
            for ind,pair,mi,xprobs,yprobs,jointprobs in frame.itertuples():
                u,v = pair
                g.addEdge(u,v,mi) 
            maxst = g.KruskalMST(maximum = maximum) ## return Maximum Spanning Tree
            ## maxst is a list of tuples((u,v), weight)
            graph = [edges for edges, weights in maxst]
            labs = [round(weight, 4) for edge, weight in maxst]
            PlotDiGraph(graph, labels = labs)
            MST[i] = maxst
        return MST

    def Predict(self, dataframe):
        models = self.MST ## dictionary(class: list of tuples)
        jointprobs = self.jointprobs
        #for key, tree in models.items():
            ## calculate p(V = v | U = u)
             


def toDiGraph(MST):
    """
    Covert to Networkx DiGraph (which is really a tree)
    """
    import networkx as nx
    graph = [edges for edges, weights in MST]
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])
    # create networkx graph
    G=nx.DiGraph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge, weight in MST:
        G.add_edge(edge[0], edge[1], weight=weight)
    return G


def find_root(G,child):
    """
    Function to find and return root of DiGraph (which is really a tree)
    """
    #print(child) ## for testing
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        print(f"found root: {child}")
        return child
    else:  #True if there is a predecessor, False otherwise
        return find_root(G, parent[0])


if __name__ == "__main__":
    ## quick test ##
    print("starting to train Graph")
    df = pd.read_csv("Pima.tr.csv")
    df['bmi'] = df.bmi.apply(int) ## convert the float to integer
    class_col_name = "type"
    model = TAN(dataframe = df, class_col_name = class_col_name)
    myG  = toDiGraph(model.MST['No'])
    test = find_root(myG, "age")
