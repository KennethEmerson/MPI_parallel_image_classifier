"""
    this file contains the classes for the nodes in the prediction pipeline
"""

import os
from data_process import load_train_metadata
from PIL import Image
from torch import no_grad
from torchvision import transforms

from neural_net import Neural_Network
from aux_functions import *

class SourceNode:
    """ the class for creating a source node object
    """

    def __init__(self,teams,img_dir,metadata,data_limit=None):
        """[summary]

        Args:
            teams (list): list of lists containing the teams with the node indexes
            img_dir (String): the directory containing the images
            metadata (String): the metadata file of the dataset
            data_limit (int, optional): if given the number of images in the dataset is limited. Defaults to None.
        """
        self.images = None
        self.labels = None
        self.img_pointer = 0
        self.teams = teams
        self.batch_end = False

        _,images,labels = load_train_metadata(img_dir,metadata,data_limit)
        self.images = images
        self.labels = labels
    

    def next_batch(self):
        """function will get a new batch of images to be predicted

        Returns:
            [list,bool]: list of image filenames, if boolean is true: request to end the node main code
        """
        terminate = False
        
        # first item = "empty" will be scattered to source_node and is therefore not used
        images_to_send = [["empty"]]
        
        for team in self.teams:
            nbr_of_pred_in_team = len(team)-1
            if not self.batch_end:
                images = (self.images[self.img_pointer:(self.img_pointer + nbr_of_pred_in_team)].tolist())
                images = pad_list(images,nbr_of_pred_in_team,"empty")
                self.img_pointer = self.img_pointer+nbr_of_pred_in_team
            
            # in case the batch has ended the source node will send a "terminate" string through the pipeline
            # which will lead to the termination of all other receiving nodes in the pipeline
            else: 
                images = ['terminate'] * nbr_of_pred_in_team
                terminate = True
            
            images_to_send.append(images) 

        # if pointer has reached end of image_list, next round terminates will be send through pipeline 
        if self.img_pointer >= len(self.images):
            self.batch_end = True
    
        return images_to_send, terminate
    

class PreprocessNode:
    """ the class for creating a preprocess node object 
    """

    def __init__(self,dir,transform_img):
        """initializes the preprocess node object

        Args:
            dir (String): the directory that contains the images
            transform_img (Transform): the Torchvision Transform function to use
        """
        self.dir = dir
        self.transform_img = transform_img
    

    def preprocess_image(self,file_names):
        """preprocess the actual image

        Args:
            file_names (String): the filename of the image

        Returns:
            [tensor,bool]: the image as a tensor, if boolean is true: request to end the node main code
        """
        terminate= False
        
        # first tuple = ("empty","empty") will be scattered to the preprocessor node and is therefore not used
        image_tuples = [("empty","empty")]
        
        
        if all(file_name == "terminate" for file_name in file_names):
            terminate = True
            
        for file_name in file_names:
            
            image_tensor = None
            # propagate the "empty" string
            if file_name == "empty":
                image_tensor = "empty"

            # propagate the "terminate" string to the next nodes
            elif file_name == "terminate":
                image_tensor = "terminate"
                terminate = True
            
            # preprocess the image
            else:
                image=Image.open(os.path.join(self.dir, file_name))
                image_tensor= self.transform_img(image)
            image_tuples.append((file_name,image_tensor))
        return image_tuples, terminate


class PredictNode:
    """the class for creating a predictor node object 
    """
    def __init__(self):
        self.neural_net = Neural_Network.load()
    
    
    def predict(self,image_tuple):
        """predicts a given image

        Args:
            image_tuple (tensor): the image as a tensor

        Returns:
            [(String,int,bool)]: the predicted image name and its predicted label. 
                                 if boolean is true: request to end the node main code
        """
        file_name = image_tuple[0]
        image_tensor = image_tuple[1]
        prediction = None
        terminate = False
        
        # empty and terminate are propagated to the sink node
        if file_name == "empty" and image_tensor == "empty":
            prediction = "empty"
        elif file_name == "terminate" and image_tensor == "terminate":
            prediction = "terminate"
            terminate = True
        else:
            with no_grad():
                prediction = self.neural_net.predict(image_tensor)
        
        return (file_name, prediction), terminate


class SinkNode:
    """ class for creating the sink node object
    """
        
    def __init__(self):
        self.predictions = []
    

    def add_predictions(self,predictions):
        """collects the predictions

        Args:
            predictions (list): list of tuples containing the image name and label

        Returns:
            [Bool]: if boolean is true: request to end the node main code
        """
        terminate = False
        if all(image == ("terminate","terminate") for image in predictions):
            terminate = True
        else:
            predictions = filter_list(predictions,("empty","empty"))
            predictions = filter_list(predictions,("terminate","terminate"))
            self.predictions = self.predictions + predictions
        
        return terminate
        
        


