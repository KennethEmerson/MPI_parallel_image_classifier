"""
    this is a separate program used to determine the time required for the various steps in the prediction pipeline
"""

import os
from PIL import Image
from torch import unsqueeze, no_grad, argmax
from torchvision import transforms

from neural_net import Neural_Network
from data_process import load_train_metadata
DATA_LIMIT_TRAIN = 100

DATASET_ROOT = ""
TRAIN_IMG_DIR = DATASET_ROOT + "train/"
TRAIN_METADATA = "small_metadata.json"

RESIZE = transforms.Resize(224)
CROP =   transforms.CenterCrop(224)
TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))

@profile
def load_image(dir,file_name):
    return Image.open(os.path.join(dir, file_name))
@profile
def resize(image):
    return RESIZE(image)
@profile
def crop(image):
    return CROP(image)
@profile
def to_tensor(image):
    return TO_TENSOR(image)
@profile
def normalize(tensor):
    return NORMALIZE(tensor)
@profile
def predict(model,tensor):
    model.model.eval()
    with no_grad():
        pred = argmax(model.model(tensor.unsqueeze(0))).item()
        return pred


if __name__ == "__main__":

    neur_net = Neural_Network.load()

    categories,all_images,all_labels = load_train_metadata(TRAIN_IMG_DIR,TRAIN_METADATA,data_limit=DATA_LIMIT_TRAIN)

    for image in all_images:
        image = load_image(TRAIN_IMG_DIR,image)
        image = resize(image)
        image = crop(image)
        img_tensor = to_tensor(image)
        img_tensor = normalize(img_tensor)
        result = predict(neur_net,img_tensor)
        



