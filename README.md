# MPI_parallel_image_classifier
This repo contains a Message Passing Interface parallel image classifier written with Python, MPI and Pytorch as part of a project for the "Information Retrieval and Data Analytics " course at the Vrije Universiteit Brussel.


## Introduction:

The repo contains an Convolutional neural net image classifier that classifies the images from the [Herbarium 2021 - Half-Earth Challenge](https://www.kaggle.com/c/herbarium-2021-fgvc8/data) dataset (not included in the repo).
Both training and evaluation are parallelized over multiple nodes by means of the [Message Passing Interface](https://mpi4py.readthedocs.io/en/stable/) in order to speed up both scripts.


## How to use:

* The repository is implemented in Python using PyTorch and MPI. A requirements.txt file is available, defining the required dependencies.
* Prior to running the training or evaluation scripts, the dataset and corresponding training and evaluation metadata (containing the labels) should be made available in separate train and test folders. 
* The location of the folders and metadata can be defined in the [main_train.py](./main_train.py) and [main_predict.py](./main_predict.py) files. 
* to run the training script use: `mpiexec -np 3 python3 main_train.py` where 3 is the number of nodes. 
* to run the evaluation script use: `mpiexec -np 10 python3 main_predict.py` where 10 is the number of nodes. 

## implementation details:

* For the training of the network we can use a free to choose number of nodes. The masternode (node 0) will start by loading the metadata from the JSON file and creates two Numpy arrays: one with the image filenames and one with the corresponding labels. The function "load_train_metadata" responsible for the creation of these arrays has the additional option to limit the size of the overall dataset in order to speed up training.

* The image file names and labels are then split up in evenly sized chunks of subsets that are scattered to all the nodes using MPI. each node (masternode and workernodes) will then create their own dataset/dataloader for the received subset.
* after each epoch, all worker nodes will return their gradients to the masternode which can then optimize the master neural net and return the updated weights back to the worker nodes for the next epoch.
* the prediction pipeline is set up with separate preprocessing and prediction nodes and one sink node which collects all predictions.   
* In order to optimize the prediction pipeline a test [test_speed.py](./test_speed.py) was created that chops the classification process into it's different activities in order to estimate the time required for each of these activities using the "line_profiler" library.
* Based on these estimations a viable ratio of preprocessing and prediction nodes can be initialized to optimize the distribution of work.

For further implementation details, please consult the documentation accompanying the code.