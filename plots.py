"""
    this file contains the plot function used to visualize the training loss and accuracy
"""

import matplotlib.pyplot as plt

def plot_training_results(epoch_list,train_loss_list,train_acc_list,val_loss_list=None,val_acc_list=None):
    """plots the training loss and accuracy

    Args:
        epoch_list (list): list of epoch indexes
        train_loss_list (list): list of loss values over the training epochs
        train_acc_list (list): list of acc values over the training epochs
        val_loss_list (list, optional): list of validation loss values over the training epochs. Defaults to None.
        val_acc_list (list, optional): list of validation acc values over the training epochs. Defaults to None.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(epoch_list,train_loss_list,label="train")
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    if val_loss_list:
        ax1.plot(epoch_list,val_loss_list,label="validation")
    ax1.legend()

    ax2.plot(epoch_list,train_acc_list,label="train")
    ax2.set_title('model accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    if val_acc_list:
        ax2.plot(epoch_list,val_acc_list,label="validation")
    ax2.legend()
    
    plt.show()