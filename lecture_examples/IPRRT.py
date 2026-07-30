# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from scipy.spatial import cKDTree
from lecture_examples.IPPRMBase import PRMBase
import numpy as np

import networkx as nx
import random

from lecture_examples.IPPerfMonitor import IPPerfMonitor

# Import wrapper function to track collision checks
from modules.TrackedCollisionChecker import TrackedCollisionChecker

# Import classes to store path length (euclidean distance)
from scipy.spatial.distance import euclidean
from modules.PlannerStats import PlannerStats

class RRTSimple(PRMBase):

    def __init__(self, _collChecker):
        """
        _collChecker: the collision checker interface
        """
        tracked_checker = TrackedCollisionChecker(_collChecker)
        super(RRTSimple, self).__init__(tracked_checker)
        self.graph = nx.Graph()
        self.lastGeneratedNodeNumber = 0

        # Initialize stats tracking
        self.stats = PlannerStats()

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """
        
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
            
        Example:
            config["numberOfGeneratedNodes"] = 500 
            config["testGoalAfterNumberOfNodes"]  = 10
        """
        # 0. reset
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0
        
        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)

        # 2. add start and goal to graph
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=checkedStartList[0])
        self.lastGeneratedNodeNumber +=1
        
        try:
            while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:

                posList = list(nx.get_node_attributes(self.graph,'pos').values())
                kdTree = cKDTree(posList)

                if (self.lastGeneratedNodeNumber % config["testGoalAfterNumberOfNodes"]) == 0:
                    #print "testing goal"
                    result = kdTree.query(checkedGoalList[0],k=1)
                    nearest_pos = self.graph.nodes[result[1]]['pos']
                    
                    if not self._collisionChecker.lineInCollision(nearest_pos, checkedGoalList[0]):
                        self.graph.add_node("goal", pos=checkedGoalList[0])
                        
                        # Calculate distance and add weight to the goal connection
                        dist_to_goal = euclidean(nearest_pos, checkedGoalList[0])
                        self.graph.add_edge(result[1], "goal", weight=dist_to_goal)
                        
                        mapping={0:'start'}
                        self.graph = nx.relabel_nodes(self.graph,mapping)

                        # Record successful stats using the weights
                        self.stats.success = True
                        self.stats.path_length = nx.shortest_path_length(self.graph, "start", "goal", weight="weight")
   
                        return nx.shortest_path(self.graph,"start","goal")


                pos = self._getRandomFreePosition()

                # for every node in graph find nearest neighbours
                result = kdTree.query(pos,k=1)
                if result is None:
                    raise Exception("Something went wrong regarding nearest neighbours")
                
                nearest_pos = self.graph.nodes[result[1]]['pos']
                if not self._collisionChecker.lineInCollision(nearest_pos, pos):
                    self.graph.add_node(self.lastGeneratedNodeNumber, pos=pos)
                    
                    # Calculate distance and add weight to standard tree branches
                    dist_to_node = euclidean(nearest_pos, pos)
                    self.graph.add_edge(result[1], self.lastGeneratedNodeNumber, weight=dist_to_node)
                    
                    self.lastGeneratedNodeNumber +=1
                    
        except Exception as e:
            print(e)
        
        # Explicitly mark as failed if the loop finishes without returning
        self.stats.success = False
        return []



### following code is not yet finalized or corrected
            

# from scipy.spatial import cKDTree
# from IPPRMBase import PRMBase
# import numpy as np

# import networkx as nx
# import random
# class RRT(PRMBase):

#     def __init__(self, _collChecker):
#         """
#         _collChecker: the collision checker interface
#         """
#         super(RRT, self).__init__(_collChecker)
#         self.graph = nx.Graph()
#         self.lastGeneratedNodeNumber = 0


#     def planPath(self, startList, goalList, config):
#         """
        
#         Args:
#             start (array): start position in planning space
#             goal (array) : goal position in planning space
#             config (dict): dictionary with the needed information about the configuration options
            
#         Example:
#             config["numberOfGeneratedNodes"] = 500 
#             config["testGoalAfterNumberOfNodes"]  = 10
#         """
#         # 0. reset
#         self.graph.clear()
#         self.lastGeneratedNodeNumber = 0
        
#         # 1. check start and goal whether collision free (s. BaseClass)
#         checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)
        
#         # 2. add start and goal to graph
#         self.graph.add_node(self.lastGeneratedNodeNumber, pos=checkedStartList[0])
#         self.lastGeneratedNodeNumber +=1
        
#         while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:
                    
#             posList = list(nx.get_node_attributes(self.graph,'pos').values())
#             kdTree = cKDTree(posList)

#             if (self.lastGeneratedNodeNumber % config["testGoalAfterNumberOfNodes"]) == 0:
#                 #print "testing goal"
#                 result = kdTree.query(checkedGoalList[0],k=1)
#                 if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'],checkedGoalList[0]):
#                     self.graph.add_node("goal", pos=checkedGoalList[0])
#                     self.graph.add_edge(result[1],"goal")
#                     mapping={0:'start'}
#                     self.graph = nx.relabel_nodes(self.graph,mapping)

#                     return nx.shortest_path(self.graph,"start","goal")

            

#             pos = self._getRandomFreePosition()
            
#             # for every node in graph find nearest neigbhours
#             result = kdTree.query(pos,k=1)
#             if None:
#                 raise Exception("Something went wrong regarding nearest neighbours")
            
#             start = np.array(self.graph.nodes[result[1]]['pos'])
#             end   = np.array(pos)
#             newPos = 0.5 * (end-start) + start
#             if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'],newPos):
#                 self.graph.add_node(self.lastGeneratedNodeNumber, pos=newPos)
#                 self.graph.add_edge(result[1],self.lastGeneratedNodeNumber)
#                 self.lastGeneratedNodeNumber +=1
            
 