# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:51:08 2017

@author: jn107154
"""


class Graph():
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.parent = dict()
    
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
    
    
    def kruskal(self):
        mst = set()
        # puts all the vertices in seperate sets
        for vertice in self.vertices:
            self.make_set(vertice)
    
        edges = self.edges
        # sorts edges in ascending order
        edges.sort()
        for edge in edges:
            weight, u, v = edge
            # checks if current edge do not close cycle
            if self.find_set(u) != self.find_set(v):
                mst.add(edge)
                self.union(u, v)
    
        return mst
"""
# input graph
vert = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
edges = [
            (2, '1', '2'),
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
            (3, '8', '9'),
]

g = Graph(vert, edges)
testmst = g.kruskal()


solution:
    {(1, '2', '6'),
 (1, '3', '4'),
 (1, '4', '10'),
 (2, '1', '2'),
 (2, '1', '3'),
 (2, '5', '10'),
 (2, '5', '9'),
 (3, '8', '9'),
 (4, '6', '7')}
"""