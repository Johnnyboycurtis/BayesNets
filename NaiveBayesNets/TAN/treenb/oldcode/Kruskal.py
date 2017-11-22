# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:18:07 2017

@author: jn107154
"""
## Code provided by:
# http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/
# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected, 
# undirected and weighted graph
 
from collections import defaultdict
 
#Class to represent a graph
class Graph(object):
    """Build a graph"""
    def __init__(self, vertices, edges):
        self.V = vertices # all unique vertices
        self.edges = [] # store results in a list
  
    # function to add an edge to graph
    def addEdge(self,u,v,w):
        """
        Add an edge to the graph
        u: vertex
        v: vertex
        w: weight
        """
        self.graph.append([w, u, v])
        
    parent = dict()
    
    def make_set(self, vertice):
        parent[vertice] = vertice


    # returns first element of set, which includes 'vertice'
    def find(self, vertice):
        if self.parent[vertice] != vertice:
            result = self.find(self.parent[vertice])
        return result
    

    # joins two sets: set, which includes 'vertice1' and set, which
    # includes 'vertice2'
    def union(self, u, v, edges):
        ancestor1 = self.find(u)
        ancestor2 = self.find(v)
        # if u and v are not connected by a path
        if ancestor1 != ancestor2:
            for edge in edges:
                self.parent[ancestor1] = ancestor2


    def kruskal(self, maximum=False):
        mst = set()
        # puts all the vertices in seperate sets
        for vert in self.V:
            self.make_set(vert)
    
        edges = self.edges
        # sorts edges in ascending order
        edges.sort(reverse=maximum)
        for edge in edges:
            weight, u, v = edge
            # checks if current edge do not close cycle
            if self.find(u) != self.find(v):
                mst.add(edge)
                self.union(u, v, edges)
        return mst


if __name__ == "__main__":
    # input graph
    vertices =  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    edges =  [  (2, '1', '2'),
                (2, '1', '3'),
                (2, '2', '3'),
                (1, '2', '6'),
                (1, '3', '4'),
                (5, '4', '6'),
                (4, '6', '7'),
                (7, '4', '5'),
                (6, '7', '5'),
                (1, '4', '10'),
                (2, '5', '10'),
                (8, '5', '8'),
                (2, '5', '9'),
                (3, '8', '9') 
                ]
    
    g = Graph(vertices, edges)
    print(g.kruskal())

"""
mst = kruskal(graph)
print("Minimal Spanning Tree:")
print(mst)
mst_weight = 0
for edge in mst:
    weight, u, v = edge
    mst_weight += weight

print("Cost: ")
print(mst_weight)
"""
    
    
    
    
