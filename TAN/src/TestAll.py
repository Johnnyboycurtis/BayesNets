# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:14:47 2017

@author: jn107154
"""

import matplotlib.pyplot as plt
from SimpleGraphPlot import draw_graph
from Graph import Graph
from CalcMutualInfo import *



if __name__ == "__main__":
    
    print("starting to train Graph")
    ## the following returns a dictionary
    ## a key for each class
    ## start_train2 return vertices, dictionary of data frames per class
    #vertices, results = start_train2("pima.csv", "IsDiabetic")
    vertices, results = start_train2("Pima.tr.csv", "type")
    for i, frame in results.items():
        print(f"DataFrame: {i}")
        print(frame)
        graph2 = frame.Pairs.tolist()
        labs = [round(i, 3) for i in frame.MI.tolist()]
        draw_graph(graph2, labels = labs)
        ## Build MST
        n = frame.shape[0]
        g = Graph(vertices, [])  ## number of unique attributes
        for ind,pair,mi in frame.itertuples():
            u,v = pair
            g.addEdge(u,v,mi) 
        minst = g.KruskalMST(maximum=False) ## return Minimum Spanning Tree
        maxst = g.KruskalMST(maximum=True) ## return Maximum Spanning Tree
        
        graph = [(u,v) for u,v,w in maxst]
        labs = [round(w, 4) for u,v,w in maxst]
        draw_graph(graph, labels = labs)


