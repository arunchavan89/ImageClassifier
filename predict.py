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


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    from torchvision import models
    model = models.vgg11(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer_0'])),
        ('relu1', nn.ReLU()),
        ('drop_out1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(checkpoint['hidden_layer_0'], checkpoint['hidden_layer_1'])),
        ('relu2', nn.ReLU()),
        ('drop_out2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(checkpoint['hidden_layer_1'], checkpoint['output_size'])), #We'll be using this dataset of 102 flower categories
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
        
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}    
    classes = classes.to("cpu")
    classes = classes.detach().numpy().tolist()[0]
    categories = [idx_to_class[idx] for idx in classes]
    
    return probs, classes, categories

if __name__ == "__main__":
    #get user input
    parser = argparse.ArgumentParser(description='Flowers Image Prediction.')
    parser.add_argument('--input', type=str, default='flowers/test/1/image_06743.jpg', help='path to test flower dataset.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='path to the checkpoint file')
    parser.add_argument('--gpu', action='store_true', help='Enable/Disable GPU')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='Path to JSON mapping file')
    args = parser.parse_args()
    
    print(args.input)
    print(args.checkpoint)
    
    print("Loading Checkpoint.!")
    model = load_checkpoint(args.checkpoint)
    print(model)
    print("Model loaded.!")
    
    device = 'cuda' if args.gpu else 'cpu'
    print("device =", device)
    model.to(device)
    
    probs, classes, categories = predict(args.input, model, topk=5)
    print("probabilty = ", probs)
    print("classes =", classes)    
    print("categories =", categories)
    
    print("cat_to_name: ", args.cat_to_name)
    with open(args.cat_to_name, 'r') as f:
        flower_to_name = json.load(f)
    for c in categories:
        print (flower_to_name[c])