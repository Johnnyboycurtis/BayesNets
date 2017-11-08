# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:14:20 2017

@author: jn107154
"""


import networkx as nx
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.style.use('seaborn-whitegrid')

def PlotNetwork(graph, labels=None, edge_text_pos=0.5):

    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # create networkx graph
    G=nx.Graph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # draw graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)
    graph_pos=nx.shell_layout(G)
    nx.draw_networkx_labels(G, graph_pos)
    if labels is None:
        labels = range(len(graph))
    
    
    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()


def PlotDiGraph(graph, labels=None, edge_text_pos = 0.6):
    """
    Plot Directed Graph
    """
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # create networkx graph
    G=nx.DiGraph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # draw graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)
    graph_pos=nx.shell_layout(G)
    nx.draw_networkx_labels(G, graph_pos)
    if labels is None:
        labels = range(len(graph))
    
    
    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()
