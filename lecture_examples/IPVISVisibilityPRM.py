# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx

import networkx as nx
def visibilityPRMVisualize(planner, solution, ax = None, nodeSize = 300):
    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    graph = planner.graph
    statsHandler = planner.statsHandler
    collChecker = planner._collisionChecker
    pos = nx.get_node_attributes(graph,'pos')
    color = nx.get_node_attributes(graph,'color')

    collChecker.drawObstacles(ax)
    
    if statsHandler:
        statPos = nx.get_node_attributes(statsHandler.graph,'pos')
        nx.draw_networkx_nodes(statsHandler.graph, pos=statPos, alpha=0.2,node_size=nodeSize)
        nx.draw_networkx_edges(statsHandler.graph, pos=statPos, alpha=0.2,edge_color='y')
        
    # draw graph 
    nx.draw_networkx_nodes(graph, pos, ax = ax, nodelist=list(color.keys()), node_color=list(color.values()), node_size=nodeSize)
    nx.draw_networkx_edges(graph, pos, ax = ax)
    
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    G0=graph.subgraph(Gcc[0])# = largest connected component

    # how largest connected component
    nx.draw_networkx_edges(G0,pos,
                               edge_color='b',
                               width=2.0, ax = ax
                            )
    
    # get nodes based on solution path
    Gsp = nx.subgraph(graph,solution)

    # draw edges based on solution path
    nx.draw_networkx_edges(Gsp,pos,alpha=0.8,edge_color='g',width=5.0, label="Solution Path")
        
    # draw start and goal
    # draw start and goal
    if "start" in graph.nodes(): 
        nx.draw_networkx_nodes(graph,pos,nodelist=["start"],
                                   node_size=nodeSize,
                                   node_color='#00dd00',  ax = ax)
        nx.draw_networkx_labels(graph,pos,labels={"start": "S"},  ax = ax)
    if "goal" in graph.nodes():
        nx.draw_networkx_nodes(graph,pos,nodelist=["goal"],
                                   node_size=nodeSize,
                                   node_color='#dd0000',  ax = ax)
        nx.draw_networkx_labels(graph,pos,labels={"goal": "G"},  ax = ax)

