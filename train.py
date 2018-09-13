import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from PIL import Image
import numpy as np
import time
import argparse

def main():
    starting = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() and parsedArgs.gpu else "cpu")
    parsedArgs = argumentParsing()
    getData(parsedArgs.data_directory)



    #TODO: ADD JSON WITH CAT NAMES
    print(device)
    print(f'Everything took us : {time.time() - starting} seconds')


def argumentParsing():
    parser = argparse.ArgumentParser(description='Hope this works')
    parser.add_argument('data_directory', type=str ,help='Where the data to check be stacked?')
    parser.add_argument('--save_dir', default='.', type=str, help='Where you wanna save the results though?')
    parser.add_argument('--arch', default='densenet121', type=str, help='Which architecture pleases you the most?')
    parser.add_argument('--learning_rate', default='0.001', type=float, help='Learning rate much?')
    parser.add_argument('--hidden_units', default='512', type=int, help='Hidden units much?')
    parser.add_argument('--epochs', default='5', type=int, help='Epochs much?')
    parser.add_argument('--gpu', action='store_true', help='Love using fast GPU?')
    args = parser.parse_args()
    return args

def getData(data_directory):
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Variables for datasets and transforms:
    var_batch_size = 32
    var_shuffle = True

    # Done: Define your transforms for the training, validation, and testing sets
    # data_transforms:

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = valid_transforms = transforms.Compose([transforms.Resize(256),
                                                             transforms.CenterCrop(224),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                                  [0.229, 0.224, 0.225])])

    # Done: Load the datasets with ImageFolder
    # image_datasets:
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Done: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders:

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=var_batch_size, shuffle=var_shuffle)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=var_batch_size, shuffle=var_shuffle)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=var_batch_size, shuffle=var_shuffle)

'''
TODOS:

      1. Everything is controlled by a main() function
DONE  2. Make a function to get all args   getArgs(args)
      3. Load data function (copy)

https://github.com/rajesh-iiith/AIPND-ImageClassifier/blob/master/train.py

https://github.com/Fiboniak/Udacity-AIPND/blob/fiboniak/train.py

https://github.com/jpmassena/AIPND/blob/master/Project/train.py
'''

if __name__ == '__main__':
    main()

