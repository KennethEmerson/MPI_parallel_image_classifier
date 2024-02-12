"""
   this file contains auxilary functions for the training and prediction flow
"""

import math
from operator import itemgetter
import numpy as np

from data_process import TRANSFORM_IMG

def cond_print_init(PRINT_STATUS):
    """sets the conditional print function to print or not print and returns the set function

    Args:
        PRINT_STATUS (bool): if true the messages in the cond print functions will be printed
    """

    def funct(string):
        if PRINT_STATUS:
            print(string,flush= True)
    return funct


def filter_list(source_list,element):
    """filters out an element from a list

    Args:
        source_list (list): the original list to be filtered
        element (_): value to be removed from list

    Returns:
        [list]: the list with the given element value removed
    """

    return [i for i in source_list if i != element]


def pad_list(seq, target_length, padding):
    """pads a list to a certain length with a given padding

    Args:
        seq (list): list to be padded
        target_length (int): final length of the returned length
        padding (_): padding element which will be placed one or more times at end of list until final length

    Returns:
        [list]: padded list of given length
    """
    length = len(seq)
    if length < target_length:
        seq.extend([padding] * (target_length - length))
    return seq


def create_teams(world_size,team_size):
    """creates list of teams in which each team consists of a certain number of node indexes/ranks
       the last team can be less than team size if world_size%team_size != 0

    Args:
        world_size (int): the total number of nodes available to be divided into teams
        team_size (int): max size per team

    Returns:
        [list]: list of teams in which each team is a list of node indexes/ranks
        [list]: list of preprocess nodes (first node in each team)
        [list]: list of predictor nodes 
    """

    list_of_nodes = [*range(2,world_size)]
    nbr_of_teams = math.ceil((world_size-2) / team_size)
    team_list = [list_of_nodes[i * team_size:(i+1) * team_size] for i in range(nbr_of_teams)]
    preprocess_node_list = (list(map(itemgetter(0), team_list)))
    predict_node_list = [i for i in [*range(2,world_size,1)] if i not in preprocess_node_list]
    
    assert len(team_list) > 0
    return team_list,preprocess_node_list,predict_node_list


def get_my_team(team_list,rank):
    """returns the node's team number in the teamlist based on its rank

    Args:
        team_list (list): list of teams in which each team is a list of node indexes/ranks
        rank (int): rank/index of the nod for which to find the team

    Returns:
        [int]: index of the team in which the node resides
    """

    index = -1
    for i in range(len(team_list)):
        if rank in team_list[i]:
            index = i
    return index


