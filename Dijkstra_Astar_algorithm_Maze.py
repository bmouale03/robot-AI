#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:27:01 2025

@author: willy
"""

import os
import time
import numpy as np
import pandas as pd
import networkx as nx

# Create a sample NumPy adjacency matrix
adj_matrix = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0], # 0 = 'A'
    [1,0,1,0,0,1,0,0,0,0,0,0], # 1 = 'B'
    [0,1,0,0,0,0,1,0,0,0,0,0], # 2 = 'C'
    [0,0,0,0,0,0,0,1,0,0,0,0], # 3 = 'D'
    [0,0,0,0,0,0,0,0,1,0,0,0], # 4 = 'E'
    [0,1,0,0,0,0,0,0,0,1,0,0], # 5 = 'F'
    [0,0,1,0,0,0,0,1,0,0,0,0], # 6 = 'G'
    [0,0,0,1,0,0,1,0,0,0,0,1], # 7 = 'H'
    [0,0,0,0,1,0,0,0,0,1,0,0], # 8 = 'I'
    [0,0,0,0,0,1,0,1,0,0,1,0], # 9 = 'J'
    [0,0,0,0,0,0,0,0,0,1,0,1], # 10 = 'K'
    [0,0,0,0,0,0,0,1,0,0,1,0]  # 11 = 'L'
    ])
nodes_names = ['A','B','C','D','E','F','G','H','I','J','K','L']
node_names = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
                  9:'J', 10:'K', 11:'L'}

# Create a graph from the NumPy array
G = nx.Graph(adj_matrix)
G_named = nx.relabel_nodes(G, node_names)
##Print the edges to verify
# print(f"G_edges = {G.edges()}")
# print(f"G_edges = {G_named.edges()}")

def shortest_path_dijkstra(G_named, source_node, target_node):
    """
    Plus court chemin entre 2 noeuds
    
    # shorted path
    chemin = nx.dijkstra_path(G, 4, 6)
    print(chemin)
    
    chemin_named = nx.dijkstra_path(G_named, "E", "G")
    print(f"shortest_path = {chemin_named}")
    
    chemin_named = nx.dijkstra_path(G_named, "L", "J")
    print(f"shortest_path = {chemin_named}") L-H-J or L-K-J
    """
    chemin_named = nx.dijkstra_path(G_named, source_node, target_node)
    
    print(f"shortest_path = {chemin_named}") 
    
    return chemin_named

def look_for_other_shortest_path(G_named, source_node, target_node):
    """
    chemin_autre = None; cpt = 0
    boolean = True
    while boolean:
        chemin_autre = nx.dijkstra_path(G_named, "L", "J")
        boolean = True if chemin_autre == chemin_named else False
        cpt += 1
        print(f"cpt = {cpt} chemin_autre={chemin_autre}") if cpt%500000 == 0 else None
    
    print(f"shortest_path = {chemin_autre}")
    """
    chemin_named = nx.dijkstra_path(G_named, source_node, target_node)
    chemin_autre = None; cpt = 0
    boolean = True
    while boolean:
        chemin_autre = nx.dijkstra_path(G_named, source_node, target_node)
        boolean = True if chemin_autre == chemin_named else False
        cpt += 1
        print(f"cpt = {cpt} chemin_autre={chemin_autre}") if cpt%500000 == 0 else None
    
    print(f"shortest_path: chemin_autre = {chemin_autre}, chemin_named={chemin_named}")

def shortest_path_intermediate_node(G_named, source_node, intertarget_nodes, target_node):
    pass

if __name__ == '__main__':

    ti = time.time()
    
    source_node = "A"
    target_node = "L"
    
    chemin_named = shortest_path_dijkstra(G_named, source_node, target_node)
    
    look_for_other_shortest_path(G_named, source_node, target_node)
    
    print(f"runtime = {time.time() - ti}") 