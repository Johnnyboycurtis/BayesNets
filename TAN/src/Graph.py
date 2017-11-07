# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:45:13 2017

@author: jn107154
"""

## Code provided by:
# http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/
# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected, 
# undirected and weighted graph
 
from collections import defaultdict
 
#Class to represent a graph
class Graph:
    """Build a graph"""
    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = [] # store results in a list 
  
    # function to add an edge to graph
    def addEdge(self,u,v,w):
        """
        Add an edge to the graph
        u: vertex
        v: vertex
        w: weight
        """
        self.graph.append([u,v,w])
 
    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        # Attach smaller rank tree under root of 
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
 
        # If ranks are same, then make one as root 
        # and increment its rank by one
        else :
            parent[yroot] = xroot
            rank[xroot] += 1 
    
    # The main function to construct MST using Kruskal's 
        # algorithm
    def KruskalMST(self, Maximum=True):
        """Kruskal's Algorithm to build Minimum/Maximum Spanning Tree"""
        result = [] #This will store the resultant MST
 
        i = 0 # An index variable, used for sorted edges
        e = 0 # An index variable, used for result[]
 
            # Step 1:  Sort all the edges in non-decreasing for Minimum Spanning Tree
            # or decreasing order for Maximum Spanning Tree
                # order of their
                # weight.  If we are not allowed to change the 
                # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,key=lambda item: item[2], reverse=Maximum)
 
        parent = [] ; rank = []
 
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
     
        # Number of edges to be taken is equal to V-1
        while e < self.V -1 :
 
            # Step 2: Pick the smallest edge and increment 
                    # the index for next iteration
            u,v,w =  self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent ,v)
 
            # If including this edge does't cause cycle, 
                        # include it in result and increment the index
                        # of result for next edge
            if x != y:
                e = e + 1    
                result.append([u,v,w])
                self.union(parent, rank, x, y)            
            # Else discard the edge
 
        # print the contents of result[] to display the built MST
        print("Following are the edges in the constructed MST")
        for u,v,weight  in result:
            #print str(u) + " -- " + str(v) + " == " + str(weight)
            print("%d -- %d == %d" % (u,v,weight))
            
        return result

    
