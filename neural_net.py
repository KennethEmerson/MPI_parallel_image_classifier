"""
    file contains the neural network class for the network used in the project
    Two model networks can be used and are defined as constants:
        resnet18
        a custom CNN which is defined in model.py
"""

import os
from torch import argmax, is_tensor, is_grad_enabled, zeros, FloatTensor, no_grad, save, load
from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable, no_grad
from torchvision import models

from aux_functions import TRANSFORM_IMG
from model import Custom_CNN

NUM_CATEGORY = 64500            #number of categories used in original dataset, must be used
LEARNING_RATE = 0.01            #learning rate of neural net optimizer

# CHANGE UNDERLYING LINE TO SELECT NN MODEL
#MODEL = models.resnet18(num_classes = NUM_CATEGORY)
MODEL = Custom_CNN(num_of_classes= NUM_CATEGORY)


class Neural_Network:
    def __init__(self):
        self.num_classes = NUM_CATEGORY
        self.model = MODEL
        self.criterion = CrossEntropyLoss()
        self.learn_rate = LEARNING_RATE
        self.transform_img = TRANSFORM_IMG
        self.optimizer = Adam(self.model.parameters(),lr = self.learn_rate)
        self.epoch_list = []
        self.loss_list = []
        self.acc_list = []
    

    def reset_log(self):
        """resets the NN log
        """
        self.epoch_list = []
        self.loss_list = []
        self.acc_list = []


    def add_to_log(self,epoch,loss,acc):
        """add the epoch loss and acc to the log

        Args:
            epoch (int): the epcoh correpsonding to the values
            loss (float): the loss for the specific epoch
            acc (float): the accuracy for the specific epoch
        """

        if self.epoch_list:
            self.epoch_list.append(self.epoch_list[-1]+1)    
        else:
            self.epoch_list.append(0)
        self.loss_list.append(loss)
        self.acc_list.append(acc)


    #@profile
    def calc_loss_one_batch(self,images,labels):
        """calculates the gradient for a specific batch and the loss and acc

        Args:
            images (tensor): tensor containing the images 
            labels (tensor): tesnor containing the labels

        Returns:
            [tuple]: tuple containing the loss, number of correct predictions and size of batch
        """

        pred = self.model(images)
        loss = self.criterion(pred,labels)

        batch_loss = loss.detach().item()
        batch_correct = (argmax(pred,1) == labels).sum().item()
        batch_total = labels.size(0)

        self.optimizer.zero_grad()
        loss.backward()
        return batch_loss,batch_correct, batch_total


    def optimize_after_one_batch(self):
        """optimizes the NN after calc the gradient
        """

        self.optimizer.step()
    

    #@profile
    def train_one_epoch(self,dataloader):
        """trains the NN for one epoch

        Args:
            dataloader (Dataloader): the dataloader to use for loading the images
        """
        
        self.model.train()
        for i in range(0,dataloader.batches):
            images, labels = dataloader.get_next_batch()
            self.calc_loss_one_batch(images,labels)
            self.optimize_after_one_batch()
        

    def get_parameters(self):
        """extracts the parameters from the NN

        Returns:
            [generator]: the actual gradient parameters
        """

        return self.model.parameters()


    def validate(self,dataloader):
        """calculates accurracy over a dataset for a trained NN

        Args:
            dataloader (Dataloader): the dataloader to use for loading the images

        Returns:
            [float]: the accuracy of the NN on the given dataset
        """
        self.model.eval()        
        
        val_correct = 0
        val_total = 0
        
        with no_grad():
            for i in range(0,dataloader.batches):
                images, labels = dataloader.get_next_batch()
                pred = self.model(images)
                val_correct += (argmax(pred,1) == labels).sum().item()
                val_total += labels.size(0)
        return val_correct / val_total
    

    @no_grad()
    def predict(self,image):
        """predicts the label of one image

        Args:
            image (tensor): the tensor representation of one image (if a numpy image is given transformation will
                            be performed first)

        Returns:
            [int]: the predicted label
        """
        self.model.eval()
        if is_tensor(image):
                return argmax(self.model(image.unsqueeze(0))).item() # make a singleton batch
        else:
                return argmax(self.model(self.transform_img(image).unsqueeze(0))).item() # make a singleton batch
    

    def save(self):
        """saves the actual model parameters
        """
        
        save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epoch_list': self.epoch_list,
        'loss_list': self.loss_list,
        'acc_list': self.acc_list
        }, 'model.pth')
    

    @classmethod
    def load(cls):
        """loads a saved model

        Returns:
            [Neural_Network]: returns a Neural_Network object of the saved model
        """

        filename = 'model.pth'
        if os.path.isfile(filename):
            checkpoint = load(filename)
            neur_net = Neural_Network()
            neur_net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            neur_net.model.load_state_dict(checkpoint['model_state_dict'])
            neur_net.epoch_list = checkpoint['epoch_list']
            neur_net.loss_list = checkpoint['loss_list']
            neur_net.acc_list = checkpoint['acc_list']
            return neur_net
        else:
            print("[ERROR] model could not be loaded")