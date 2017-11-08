# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:51:08 2017

@author: Jonathan Navarrete
"""


class Graph():
    """Build a graph"""
    def __init__(self, vertices, edges):
        self.vertices = vertices ## should be a list of column names
        self.edges = edges ## should be a list of tuples
        self.parent = dict()
    
    
    def addEdge(self,u,v,w):
        """
        Add an edge to the graph
        u: vertex1
        v: vertex2
        w: weight
        """
        self.edges.append([u,v,w])
    
    def make_set(self, vertice):
        self.parent[vertice] = vertice
    
    
    # returns first element of set, which includes 'vertice'
    def find_set(self, vertice):
        try:
            if self.parent[vertice] != vertice:
                self.parent[vertice] = self.find_set(self.parent[vertice])
            return self.parent[vertice]
        except KeyError:
            print(self.parent)
    
    # joins two sets: set, which includes 'vertice1' and set, which
    # includes 'vertice2'
    def union(self, u, v):
        ancestor1 = self.find_set(u)
        ancestor2 = self.find_set(v)
        # if u and v are not connected by a path
        if ancestor1 != ancestor2:
            for edge in self.edges:
                self.parent[ancestor1] = ancestor2
    
    
    def KruskalMST(self, maximum=False):
        """Kruskal's Algorithm to build Minimum/Maximum Spanning Tree"""
        mst = [] ## min/max Spanning Tree results
        
        # puts all the vertices in seperate sets
        for vertice in self.vertices:
            self.make_set(vertice)
    
        edges = self.edges
        # sorts edges based on location of weight and min/max
        edges.sort(key = lambda line: line[2], reverse=maximum)
        
        for edge in edges:
            u, v, weight = edge
            # checks if current edge do not close cycle
            if self.find_set(u) != self.find_set(v):
                mst.append(edge)
                self.union(u, v)
        self._printMST(mst)
        return mst
    
    def _printMST(self, mst):
        print("Kruskal MST Results: ")
        print("---------------------")
        for u,v,weight  in mst:
            print(f"{u} -- {v} == {weight}")
