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

import json
with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)
        
def train(args):
    """Method for training the architecture

        Args: 
            arg

        Attributes:
    """
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # DONE: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    # DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    print("train_loader, test_loader, valid_loader ready.!")
    
    input_size = 0
    if args.arch == "vgg":
        model = models.vgg11(pretrained = True)
        input_size = model.classifier[0].in_features
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained = True)
        input_size = model.classifier.in_features
    else:
        raise Exception("Please write either VGG or densenet121 as name of the model.")
    
    print(args.arch)
    print(input_size)
    
    #Freeze parameters as we don't backprop
    #for param in model.parameters():
        #param.requires_grad = False
    
    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, 1024)),
        ('relu1', nn.ReLU()),
        ('drop_out1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(1024, 512)),
        ('relu2', nn.ReLU()),
        ('drop_out2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(512, 102)), #We'll be using this dataset of 102 flower categories
        ('output', nn.LogSoftmax(dim = 1))]))
    print(model)
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = 5
    steps = 0
    running_loss = 0
    lr=0.001 #learning rate

    #constants
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(epochs):
        model.train()    
        train_loss = 0
    
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
        
            output = model.forward(images)
            loss = criterion(output, labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        else:
            val_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)
                    loss = criterion(output, labels)
                    val_loss += loss.item()
                    p = torch.exp(output)
                    p, top_c = p.topk(1)
                    equals = top_c.squeeze() == labels
                    equals = equals.type(torch.FloatTensor)
                    accuracy += equals.mean()
                else:
                    avg_train_loss = train_loss/len(train_loader)
                    avg_val_loss = val_loss/len(valid_loader)
                    per_accuracy = accuracy/len(valid_loader) * 100
                    print('Epoch: {} ---- Train loss: {:.3f} ---- Val loss: {:.3f} --- Acc: {}'\
                        .format(e+1, avg_train_loss, avg_val_loss, per_accuracy))
    
    print('Model has been trained successfully! Saving the trained model.!')
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': input_size,
                'output_size': 102,
                'epochs': epochs,
                'learning_rate':lr,
                'batch_size': train_loader.batch_size,
                'arch': args.arch,
                'class_to_idx': model.class_to_idx,              
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
                }
    torch.save(checkpoint, 'checkpoint.pth')
    print('Model has been saved successfully!')
    
if __name__ == "__main__":
    #get user input
    parser = argparse.ArgumentParser(description='Flowers Image Classifier.')
    parser.add_argument('--save_dir', type=str, default='./', help='checkpoint save trained model')
    parser.add_argument('--arch', type=str, default='./', help='vgg or densenet121')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory', required=True)
    parser.add_argument('--gpu', type=bool, default=False, help='Enable/Disable GPU')
    args = parser.parse_args()
    
    print("Training started.!")
    train(args)    
    print("Training finished successfully.!")
    
    
    