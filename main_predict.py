""" 
    File contains the actual prediction pipeline and the MPI communication.
    The different tasks are distributed in the following way:
        * Node 0 (Source Node) will serve as a streaming source and send the image filenames as input
        * Node 1 will serve as streaming sink and receive the final predictions together with the filenames
        * rest of nodes will be divided into teams that consist of one datapreprocessor and multiple predictor nodes.
    By means of a Line profiler speed test (see readme file) we could establish that the prediction of the image label 
    Takes op 7-8 times as much time as the image loading and preprocessing. Therefore 7 predictor nodes will be provided
    in one team. if a larger amount of nodes are provided extra teams will be created each of max 1 preprocess and 7 
    predictor nodes


    the training can be started with the command: mpiexec -np 10 python3 main_predict.py
    The pipeline works with a minimum of 4 or more nodes.
    (Due to e-mail attachment size restrictions a working model could not be included. 
    Please run main_train.py first to create a saved model)
"""

import math
from mpi4py import MPI
from torch import set_num_threads
from torchvision import transforms

from data_process import *
from neural_net import Neural_Network
from nodes_predict import PredictNode, PreprocessNode, SinkNode, SourceNode
from aux_functions import *

DATASET_ROOT = ""
TRAIN_IMG_DIR = DATASET_ROOT + "train/"
TEST_IMG_DIR = DATASET_ROOT + "test/"

TRAIN_METADATA = "small_metadata.json"
#TRAIN_METADATA = "metadata.json"
TEST_METADATA = "metadata.json"

DATA_LIMIT = 10 

# this is the optimal number of predictor nodes in a team as found through line profiler
# therefore if more processes are provided than 10, (source + sink + one team of 7 predictors and one preprocessor)
# an extra team will be created
PREDICT_PREPROCESS_RATIO = 7

# if True the status updates are printed in the terminal
PRINT_STATUS = True
    
# load the conditional print function  
cond_print = cond_print_init(PRINT_STATUS)

if __name__ == "__main__":

    # Disable Pytorch multithreading
    set_num_threads(1) 
    
    # create MPI communicator
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    
    # minimum 4 nodes are required to create the prediction pipeline
    assert world_size > 4
    rank = comm.Get_rank()
    
    # create teams of 1 preprocessor and multiple predictors up to PREDICT_PREPROCESS_RATIO
    # and create list of preprocessors and predictors
    teams,preprocessors,predictors = create_teams(world_size,PREDICT_PREPROCESS_RATIO + 1)
    
    # initialize Booleans to determine function of node and get team number
    is_source_node = (rank == 0)
    is_sink_node = (rank == 1) 
    is_preprocess_node = (rank in preprocessors)
    is_predict_node = (rank in predictors)
    team = get_my_team(teams,rank)
    
    # create specific communicators to communicate between:
    # - source_node and preprocessor_node of each team
    # - preprocessor_node and eacht predictor_node in team
    # - predictor_nodes and the sink_node
    comm_source_to_prep = comm.Create_group(comm.group.Incl([0] + preprocessors))
    comm_prep_to_predict = comm.Split(color=team)
    comm_predict_to_sink = comm.Create_group(comm.group.Incl([1] + predictors))

#Create Node Objects
    if is_source_node:
        
        cond_print(f"[INFO] number of teams: {len(teams)}")
        cond_print(f"[INFO] list of teams: {teams}")
        cond_print(f"[INFO] preprocessor list: {preprocessors}")
        cond_print(f"[INFO] preprocessor list: {predictors}")
        
        # a warning is given when the last team consists of only a preprocessing node
        if any(len(team)==1 for team in teams):
            cond_print("[WARNING] one team contains only a data preproccesor and will not contribute")

        # create source node object
        node = SourceNode(teams,TRAIN_IMG_DIR,TRAIN_METADATA,data_limit=DATA_LIMIT)
        cond_print("[INFO] Metadata loaded in node 0")

    elif is_preprocess_node:
        # create preprocessor node object
        node = PreprocessNode(TRAIN_IMG_DIR,TRANSFORM_IMG)
    elif is_predict_node:
        # create predictor node object
        node = PredictNode()
    elif is_sink_node:
        # create sink node object
        node = SinkNode()
        
    terminate = False
    cond_print(f"[INFO] node {rank} has started")
    
    while not terminate:
        
        image_filenames = None
        image_tuples = None
        prediction_tuple = None

        # create new batch for all preprocessors based on the 
        # terminate = True will cause source_node to end loop 
        if is_source_node:
            image_filenames, terminate = node.next_batch()
        
        # scatter batch to all preprocessors
        if is_source_node or is_preprocess_node:
            image_filenames = comm_source_to_prep.scatter(image_filenames)
        
        # preprocess the batch
        # terminate = True will cause preprocess_node to end loop 
        if is_preprocess_node:
            image_tuples, terminate = node.preprocess_image(image_filenames)
        
        # scatter preprocesses images to all predictors in group
        if is_preprocess_node or is_predict_node:
            image_tuples = comm_prep_to_predict.scatter(image_tuples)
        
        # predict the image
        # terminate = True will cause predict_node to end loop 
        if is_predict_node:
            prediction_tuple, terminate = node.predict(image_tuples)
        
        # place a dummy prediction as first entry 
        # (this will be the input for the gather from the sink node) 
        if is_sink_node:
            prediction_tuple = ("terminate","terminate")

        #gather the predictions
        if is_predict_node or is_sink_node:
            prediction_tuple = comm_predict_to_sink.gather(prediction_tuple, root=0)
        
        # add predictions to the list in the sink node
        if is_sink_node:
            terminate = node.add_predictions(prediction_tuple)
    
    # print the predictions as gathered by the sink
    if is_sink_node:
        cond_print("\n[INFO] prediction results:")
        cond_print(  "--------------------------")
        for prediction in node.predictions:
            cond_print(prediction)
    
    cond_print(f"[INFO] node {rank} has finished")
