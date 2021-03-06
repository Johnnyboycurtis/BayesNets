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
from Probs import Probs ## code for calculating joint/marginal probabilities
from Graph import Graph ## Graph module
from SimpleGraphPlot import draw_graph
from Plot import PlotDiGraph, PlotNetwork ## plotting
from tqdm import tqdm

class TAN():
    def __init__(self, dataframe, class_col_name, maximum=True, progress_bar=False):
        self.class_col_name = class_col_name
        self.priors = self.Priors(dataframe, class_col_name)
        colnames = dataframe.columns.tolist()
        colnames.remove(class_col_name)
        self.colnames = colnames
        self.MIresults = self.Train(dataframe, progress_bar = progress_bar) ## a dictionary {class: dataframe}
        self.MST = self.BuildMST()
        self.Roots = self.FindRoot() ## {class: root name}
        self.RootProbs = {class_name: Probs(dataframe[root_name].tolist()) for class_name, root_name in self.Roots.items()}
        self.TreeProbs = self.PruneModel()
    
    def Priors(self, dataframe, class_col_name):
        n = dataframe.shape[0]
        counts = dataframe[class_col_name].value_counts()
        priors = (counts / n).to_dict()
        return priors
        
    def Train(self, df, progress_bar):
        class_col_name = self.class_col_name
        g = df.groupby(by = class_col_name) ## group df by class
        if progress_bar:
            g = tqdm(g)
        colnames = self.colnames
        ## process the following steps for each class
        ClassMats = {} ## dictionary to store MutualInfMatrix for each class
        for i, frame in g:
            colcombos = it.combinations(colnames, 2) ## will return tuples
            MutualInfo = []
            for u, v in tqdm(colcombos):
                ulist = frame[u].tolist()
                vlist = frame[v].tolist()
                probs = Probs(ulist, vlist) ## calculates all probs
                MI = probs.CalcMutualInfo()
                MutualInfo.append([(u,v), MI, probs])
            MutualInfMatrix = pd.DataFrame(MutualInfo, columns = ['Pairs', "MI", "Probs"])
            MutualInfMatrix.sort_values(by = "MI", ascending=False, inplace=True)
            ClassMats[i] = MutualInfMatrix ## store results for current class
        return ClassMats

      
    def BuildMST(self, maximum=True):
        vertices = self.colnames
        ClassFrames = self.MIresults
        
        MST = {}
        for i, frame in ClassFrames.items():
            print(f"\nClass: {i} || Unidirected Graph: ")
            print("--------------------------------")
            #print(frame)
            graph2 = frame.Pairs.tolist()
            labs = [round(i, 3) for i in frame.MI.tolist()]
            #PlotNetwork(graph2, labels = labs)

            ## Build MST
            g = Graph(vertices, [])  ## number of unique attributes
            for ind,pair,mi,probs in frame.itertuples():
                u,v = pair
                g.addEdge(u,v,mi) 
            ## return Maximum Spanning Tree and switched flag (list)
            maxst = g.KruskalMST(maximum = maximum) 
            ## maxst is a list of tuples((u,v), weight)
            graph = [edges for edges, weights, switch in maxst]
            labs = [round(weight, 4) for edge, weight, switch in maxst]
            #PlotDiGraph(graph, labels = labs)
            MST[i] = maxst ## returns list [((u,v), MI, revered-flag), ...]
        return MST

    
    def FindRoot(self):
        Roots = {}
        for klass, vals in self.MST.items():
            edges = [edge for edge, weight, switch in vals]
            parents, children = list(zip(*edges))
            root = [parent for parent in parents if parent not in children]
            if len(root) > 1:
                print(f"WARNING: THERE ARE MULTIPLE ROOTS!! {klass}: {root}")
                print(edges)
            Roots[klass] = root[0] 
        return Roots
            

    def PruneModel(self):
        """
        This needs a lot of work....
        """
        models = self.MST ## dictionary(class: list of tuples)
        modelprobs = self.MIresults ## dictionary {class: dataframe}
        TreeProbs = {}
        for key, tree in models.items():
            edges = []
            switches = []
            for edge, weight, switch in tree:
                if switch:
                    u,v = edge
                    edge = (v,u)
                edges.append(edge)                
                switches.append(switch)
            ## extract class data frame
            df = modelprobs[key]
            df.index = df.Pairs ## make edges the index
            ## extract probabilities
            Seriesprobs = df.loc[edges].Probs
            """
            So when the edge (u,v) gets switched to (v,u),
            I've not yet accounted for this so when I search for it in the
            dataframe, the dataframe just return np.nan instead of breaking!
            """
            classprobs = {}
            stuff = zip(Seriesprobs.iteritems(), switches)
            for Series, truth in stuff:
                i, prob = Series
                if truth:
                    prob.__reverse__()
                classprobs[i] = prob
            TreeProbs[key] = classprobs
        return TreeProbs


    def Predict(self, newdf, log=False, progress_bar=False):
        TreeProbs = self.TreeProbs
        newcols = newdf.columns.tolist()
        RootProbs = self.RootProbs
        if self.class_col_name in newcols:
            newdf = newdf.drop(self.class_col_name, axis = 1)
        results = []
        rows = newdf.iterrows()
        if progress_bar:
            rows = tqdm(rows)
        for i, row in newdf.iterrows():
            sample = row.to_dict()
            class_probs = {}
            for class_name, tree in TreeProbs.items():
                prior = self.priors[class_name]
                root = self.Roots[class_name]
                Prob = RootProbs[class_name]
                pval = Prob.PredMarginalProb(sample[root]) ## get root's new sample
                LogCondProb = 0 + np.log(prior) + np.log(pval)
                for edge, probs in tree.items():
                    u, v = edge ## edge = (u,v)
                    pval = probs.ConditionalProb(sample[u], sample[v])
                    LogCondProb += np.log(pval)
                if log:
                    class_probs[class_name] = LogCondProb
                else:
                    class_probs[class_name] = np.exp(LogCondProb)
            results.append(class_probs)
        dfresult = pd.DataFrame(results)
        dfresult[self.class_col_name] = dfresult.idxmax(axis=1)
        #SProbs = dfresult.sum(axis=1) ## sum of probs
        #dfresult = dfresult.divide(SProbs)
        return dfresult



def toDiGraph(MST):
    """
    Covert to Networkx DiGraph (which is really a tree)
    """
    import networkx as nx
    graph = [edges for edges, weights, switch in MST]
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])
    # create networkx graph
    G=nx.DiGraph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge, weight, switch in MST:
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


