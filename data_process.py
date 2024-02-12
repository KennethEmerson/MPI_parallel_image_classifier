"""
    This file contains all the specific functions and classes for loading and manipulating the data
"""

import os
import math
import pandas as pd
import numpy as np
import json as JSON
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms

"""
    the transformation function for the images as required for the standard torchvision
    pretrained neural nets (https://pytorch.org/vision/stable/models.html)
"""
TRANSFORM_IMG = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])


def load_train_metadata(image_dir,metadata,data_limit=None):
    """loads the JSON metadata file for the training data

    Args:
        image_dir (String): the directory in which the metadata file is to be found
        metadata (String): name of JSON-file containing the metadata
        data_limit (int, optional): if not none, only the "data_limit" first number of samples are returned. Defaults to None.

    Returns:
        (int,List,List): returns tuple containing number of categories in whole training set,
                            a list with the image file names and a list with the corresponding labels
    """
    
    train = JSON.load(open(image_dir + metadata,"r", encoding="ISO-8859-1"))
    
    images = pd.DataFrame(train['images'])
    annotate = pd.DataFrame(train['annotations'])
    train_df = images.merge(annotate, on="id")

    if data_limit is not None:
        train_df = train_df[:data_limit]

    num_categories = len(train_df['category_id'].value_counts())
    image_file_names = train_df['file_name'].values
    image_labels = train_df['category_id'].values
    
    return num_categories,image_file_names,image_labels


def load_test_metadata(image_dir,metadata,data_limit=None):
    """loads the JSON metadata file for the test data

    Args:
        image_dir (String): the directory in which the metadata file is to be found
        metadata (String): name of JSON-file containing the metadata
        data_limit (int, optional): if not none, only the "data_limit" first number of samples are returned. Defaults to None.

    Returns:
        [List]: a list with the image file names
    """

    test = JSON.load(open(image_dir + "metadata.json","r", encoding="ISO-8859-1"))
    test_df = pd.DataFrame(test['images'])
    
    if data_limit is not None:
        test_df = test_df[:data_limit]
    image_file_names = test_df['file_name'].values

    return image_file_names


def split_train_val(image_file_names,image_labels,validation_ratio,shuffle=True):
    """splits the list of image names and the list of corresponding labels into a train and validation set

    Args:
        image_file_names (List): list of image filenames
        image_labels (List): list of corresponding labels for each image
        validation_ratio (Float): ratio to determine size of validation set, must be between 0 and 1
        shuffle (bool, optional): if true the original dataset is shuffled first before splitting. Defaults to True.

    Returns:
        [(List,List,List,List)]: returns training image files, training image labels, 
                                 validation image files, validation image labels 
    """

    size_files = image_file_names.size
    size_labels = image_labels.size
    assert size_files == size_labels
    
    indexes = np.array(range(size_files))
    if shuffle:
        np.random.shuffle(indexes)
    val_indexes = indexes[0:math.floor(indexes.size*validation_ratio)]
    train_indexes = indexes[math.floor(indexes.size*validation_ratio):]
    
    train_image_files = np.take(image_file_names,train_indexes)
    train_image_labels = np.take(image_labels,train_indexes)
    
    val_image_files = np.take(image_file_names,val_indexes)
    val_image_labels = np.take(image_labels,val_indexes)
    
    return train_image_files,train_image_labels,val_image_files,val_image_labels


def split_data_equally(image_file_names,image_labels,nbr_of_splits,shuffle=False):
    """splits the list of image names and the list of corresponding labels into equally sized subsets

    Args:
        image_file_names (List): list of image filenames
        image_labels (List): list of corresponding labels for each image
        nbr_of_splits (int): number of subsets to create
        shuffle (bool, optional): if true the dataset is shuffled first before splitting. Defaults to False.

    Returns:
        [type]: [description]
    """
    size_files = image_file_names.size
    size_labels = image_labels.size
    assert size_files == size_labels

    indexes = np.array(range(size_files))
    if shuffle:
        np.random.shuffle(indexes)
    
    split_indexes = np.array_split(indexes,nbr_of_splits)
    image_file_splits = [np.take(image_file_names,x) for x in split_indexes]
    image_label_splits = [np.take(image_labels,x) for x in split_indexes]
    
    return list(zip(image_file_splits, image_label_splits))


class Custom_Dataset(Dataset):
    """Custom Dataset class, inherits from Pytorch Dataset"""
    
    def __init__(self, dir,filenames, labels,transform_img):
        """initialize new custom dataset object

        Args:
            dir (String): root dir where the sample images are found
            filenames (List): list of image filenames
            labels (List): List of corresponding image labels
            transform_img (Transform): The Pytorch Transform object to be used for transforming the images
        """

        self.dir = dir
        self.filenames = filenames
        self.labels = labels
        self.transform = transform_img
    

    def __len__(self):
        """returns the total size of the dataset

        Returns:
            [int]: the total size of the dataset
        """

        return len(self.filenames)
    

    def __getitem__(self,index):
        """returns the transformed image with the correpsonding label, if in dir train, 
           otherwise only returns the transformed image

        Args:
            index (int): index position in image list for image to be returned

        Returns:
            [(Tensor,label)]: Tensor representation of the transformed image, if in test dir returns only tensor
        """

        x = Image.open(os.path.join(self.dir, self.filenames[index]))
        
        if "train" in self.dir:
                return self.transform(x),self.labels[index]
        if "test" in self.dir:
                return self.transform(x)
    

    def getimage(self,index):
        """returns an image from the dataset without transforming (use to show image on screen)

        Args:
            index (int): index position in image list for image to be returned

        Returns:
            [Image]: returns PIL format of image
        """
        image = Image.open(os.path.join(self.dir, self.filenames[index]))
        return image



class Custom_Dataloader:

    def __init__(self,dataset,batch_size):
        """initialize custom dataloader

        Args:
            dataset (Dataset): the dataset to be loaded by the dataloader
            batch_size (int): the size of each batch to be returned
        """
        self.dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last =True)
        self.batches = len(self.dataloader)
    
    #@profile
    def get_next_batch(self):
        return next(iter(self.dataloader))