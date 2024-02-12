"""
    This file contains the distributed training pipeline. Because the distributed training requires the nodes to share
    each others gradients to allow for the optimization phase of each node's neural net, it was decided to be unfeasable 
    to create seperate pipelines for preprocessing the images and training the NN (all nodes must remain in sync). 
    Therefore every node is responsible for the preprocessing of its own subset of the data and training it's NN. 
    The first node does however act as a masternode and is ,besides training it's own NN, 
    responsible for dividing the training set and distributing the
    metadata of each subset to the corresponding workernode at the start of the program. 
    The masternode will also collect the training loss and accuracy metrics from all other nodes to allow for saving this 
    metrics and plotting the data at the end of the training.

    The training pipeline will finally, also save the trained model to allow it's use in the prediction pipeline. 

    the training can be started with the command: mpiexec -np 3 python3 main_train.py
    The pipeline works for one or more nodes.
"""
# start with: mpiexec -np 3 python3 main_train.py
# or use more nodes if available

from mpi4py import MPI
import torch
import time

from data_process import *
from neural_net import Neural_Network
from aux_functions import *
from plots import plot_training_results


# PARAMETERS FOR RUNNING THE TRAINING:
#-------------------------------------
DATASET_ROOT = ""
TRAIN_IMG_DIR = DATASET_ROOT + "train/"
TEST_IMG_DIR = DATASET_ROOT + "test/"

TRAIN_METADATA = "small_metadata.json"
#TRAIN_METADATA = "metadata.json"

BATCHES = 5                     # number of batches per epoch   
EPOCHS = 20                     # number of epochs to train

DATA_LIMIT_TRAIN = 100          # value to limit samples loaded/used from original dataset
PRINT_STATUS = True             #if true: status updates will be printed during training
HOT_START = False               #if true: training resumes using the saved model "model.pth"

cond_print = cond_print_init(PRINT_STATUS)


if __name__ == "__main__":

    # Disable Pytorch multithreading
    torch.set_num_threads(1)

    # create MPI communicator 
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    is_masternode = rank==0
    is_worker_node = rank>0
    

    if is_masternode:     
        # load JSON and create X_train(image filenames) and y_train (image labels)
        categories,all_images,all_labels = load_train_metadata(TRAIN_IMG_DIR,TRAIN_METADATA,data_limit=DATA_LIMIT_TRAIN)
        cond_print(f"\n[INFO] categories in actual dataset:{categories} (NN will however use 64500 cat as in orig. dataset)")
        
        # split the training set into parts for each node
        train_data = split_data_equally(all_images,all_labels,world_size)  
    
    else:
        train_data = None
    
    # send the metadata of each training set part to the corresponding worker node
    images, labels = comm.scatter(train_data, root=0)
    
    # create dataset and dataloader in each node
    node_dataset = Custom_Dataset(TRAIN_IMG_DIR,images,labels,TRANSFORM_IMG)
    batch_size = max(len(labels)//BATCHES,1)
    dataloader = Custom_Dataloader(node_dataset,batch_size)
    cond_print(f"[INFO] node {rank}: dataset received")

    # create neural net object in each node
    neural_net = Neural_Network()
    
    if is_masternode:
        # the masternode will load the saved neural net model parameters if requested through "HOTSTART"
        if HOT_START:
            neural_net = Neural_Network.load() 
        
        # the masternode extracts the parameters from it's neural net
        nn_state = neural_net.model.state_dict()
    else:
        nn_state = None

    # the masternode sends its neural net parameters to all worker nodes
    nn_state = comm.bcast(nn_state, root=0)
    
    # the worker nodes copy the masternode neural net parameters into their own network so that all
    # node neural nets are in sync
    if is_worker_node:
        neural_net.model.load_state_dict(nn_state)
    
    
    neural_net.model.train()
    cond_print(f"[INFO] node {rank}: neural net synchronised")

    if is_masternode:
        cond_print("\n[INFO] Training started")
        cond_print("------------------------")
    
    # each node (master and workers) start training own their part of the dataset
    for epoch in range(0,EPOCHS):
        start_time = time.time()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i in range(0,dataloader.batches):
            images, labels = dataloader.get_next_batch()
            batch_loss,batch_correct, batch_total = neural_net.calc_loss_one_batch(images,labels)
            
            train_loss += batch_loss
            train_correct += batch_correct
            train_total += batch_total

            # after each batch every node sends their gradients to all other nodes through allreduce
            # so that the total gradient can be averaged and used for optimization of every node neural network
            # (inspired by: https://pytorch.org/tutorials/intermediate/dist_tuto.html)

            for param in neural_net.get_parameters():
                param.grad.data = comm.allreduce(param.grad.data,MPI.SUM)
                param.grad.data /= float(world_size)
            neural_net.optimize_after_one_batch()
    
        # after each epoch the masternode gathers the training loss and accuracy of each node
        totals = comm.gather([train_loss,train_correct,train_total], root=0)
        
        # after each epoch, the masternode processes the training metrics, saves them and prints an overview
        # for tha epoch
        if is_masternode:
            totals = list((sum(x) for x in zip(*totals)))
            loss = totals[0]
            acc = totals[1]/totals[2]
            neural_net.add_to_log(epoch,loss,acc)
            message = (f"[INFO] Epoch: {epoch} | Duration: {round(time.time() - start_time,1)} |" + 
                       f" train loss: {round(loss,3)} | train acc: {round(100 * acc,2)} %")
            cond_print(message)
    
    # after the training the master node will plot the training results and save the model for further use
    if is_masternode:
        plot_training_results(neural_net.epoch_list,neural_net.loss_list,neural_net.acc_list)
        neural_net.save()