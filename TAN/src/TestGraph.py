# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:20:18 2017

@author: jn107154
"""

from Graph import *


if __name__ == "__main__":
    # Driver code
    vert = [str(i) for i in range(4)]
    #"""
    edges = [('0', '1', 10),
             ('0', '2', 6),
             ('0', '3', 5),
             ('1', '3', 15),
             ('2', '3', 4)]
    """
    edges = [(10, '0', '1'),
             (6, '0', '2'),
             (5, '0', '3'),
             (15, '1', '3'),
             (4, '2', '3')]
    #"""
    g = Graph(vert, edges)
    testmstMIN = g.Kruskal()
    print(testmstMIN)

    testmstMAX = g.Kruskal(maximum=True)
    print(testmstMAX)

    """
    g = Graph(4)  ## [0,1,2,3]
    g.addEdge(0, 1, 10)
    g.addEdge(0, 2, 6)
    g.addEdge(0, 3, 5)
    g.addEdge(1, 3, 15)
    g.addEdge(2, 3, 4)
     
    minst = g.KruskalMST(Maximum=False) ## return Minimum Spanning Tree
    #maxst = g.KruskalMST(Maximum=True) ## return Maximum Spanning Tree
    """
    
    """
    Original results:
        Following are the edges in the constructed MST
        2 -- 3 == 4
        0 -- 3 == 5
        0 -- 1 == 10
    """
    
    ## current results:
    """
    Minimum Spanning Tree
    Following are the edges in the constructed MST
    2 -- 3 == 4
    0 -- 3 == 5
    0 -- 1 == 10
    """
    
    """
    Maximum Spanning Tree
    Following are the edges in the constructed MST
    1 -- 3 == 15
    0 -- 1 == 10
    0 -- 2 == 6
    """
    

"""
    g = Graph(4)  ## [0,1,2,3]
    g.addEdge('0', '1', 10)
    g.addEdge('0', '2', 6)
    g.addEdge('0', '3', 5)
    g.addEdge('1', '3', 15)
    g.addEdge('2', '3', 4)
     
    minst = g.KruskalMST(Maximum=False) ## return Minimum Spanning Tree
    maxst = g.KruskalMST(Maximum=True) ## return Maximum Spanning Tree

"""
