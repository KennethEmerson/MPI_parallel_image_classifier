"""
    This file contains the NN model for the custom CNN 
"""
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout


# prevents showing of :
# "UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. 
# Please do not use them for anything important until they are released as stable"
import warnings
warnings.filterwarnings("ignore")

class Custom_CNN(Module):   
    def __init__(self,num_of_classes):
        """initialises the NN model

        Args:
            num_of_classes (int): number of label classes
        """
        super(Custom_CNN, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1), 
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2), 
            
            Conv2d(16, 16, kernel_size=3, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)) 

        self.class_layers = Sequential(
            Linear(16*56*56, 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, num_of_classes))
  
    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1,16*56*56)
        x = self.class_layers(x)
        return x