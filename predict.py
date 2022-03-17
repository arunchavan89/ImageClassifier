# Imports here
import matplotlib.pyplot as plt

# import pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# image
from PIL import Image
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())

#argparse
import argparse
import random
import os

import json
with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    from torchvision import models
    model = models.vgg11(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], 1024)),
        ('relu1', nn.ReLU()),
        ('drop_out1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(1024, 512)),
        ('relu2', nn.ReLU()),
        ('drop_out2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(512, 102)), #We'll be using this dataset of 102 flower categories
        ('output', nn.LogSoftmax(dim = 1))]))
    
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.epoch=checkpoint['epochs']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.resize((256, 256))
    width, height = image.size
    
    top_left_x = (width - 224)//2
    top_left_y = (height - 224)//2
    bottom_left_x = (width + 224)//2
    bottom_left_y = (height + 224)//2
    
    image = image.crop((top_left_x, top_left_y, bottom_left_x, bottom_left_y))
    np_image = np.array(image)
    
    #normalize the input image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (np_image - mean) / std
    
    #The color channel needs to be first    
    
    return im.transpose(2,0,1) 

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.FloatTensor([image])
    model.eval()
    output = model.forward(image.to(device))
    probs = torch.exp(output)    
    probs, classes = probs.topk(topk)
    
    return probs, classes

if __name__ == "__main__":
    #get user input
    parser = argparse.ArgumentParser(description='Flowers Image Prediction.')
    parser.add_argument('--input', type=str, default='flowers/test/1/image_06743.jpg', help='path to test flower dataset.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='vgg or densenet121')    
    args = parser.parse_args()
    
    print(args.input)
    print(args.checkpoint)
    
    print("Loading Checkpoint.!")
    model = load_checkpoint(args.checkpoint)
    print(model)
    print("Model loaded.!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    probs, classes = predict(args.input, model, topk=5)
    print("probabilty = ", probs)
    print("classes =", classes)
