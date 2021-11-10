from get_train_args import get_train_args
in_arg = get_train_args()

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

import seaborn as sb
import os
import time
import numpy as np
import matplotlib.pyplot as plt



data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                       ])

valid_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
                                
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)


import json


if in_arg.device == 'cpu':
    device = 'cpu'
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
densenet121 = models.densenet121(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'densenet': densenet121, 'vgg': vgg16}
model = models[in_arg.arch]

#model = models.densenet121(pretrained=True)
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.4):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    

for param in model.parameters():
    param.requires_grad = False


    
    
    
classifier_dens = Network(input_size = 1024,
                     output_size = 102,
                     hidden_layers = in_arg.hidden_units,
                     drop_p = 0.2)
classifier_vgg = Network(input_size = 25088,
                     output_size = 102,
                     hidden_layers =  in_arg.hidden_units,
                     drop_p = 0.2)

#Update the original classifier
if in_arg.arch == 'vgg':
    model.classifier = classifier_vgg
else:
    model.classifier = classifier_dens

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):

    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = print_every
    model.to(device)

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():

                    for images, labels in validloader:

                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        #calculate our accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                   
                    
if in_arg.device == 'cpu':
    device = 'cpu'
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model = model,
            trainloader = trainloader,
            validloader = validloader,
            epochs = in_arg.epochs,
            print_every = 20,
            criterion = criterion,
            optimizer = optimizer,
            device = device)


#Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch': in_arg.arch, 
              'model': model,
              'learning_rate': in_arg.learning_rate,
              'hidden_units': in_arg.hidden_units,
              'classifier' : model.classifier,
              'epochs': in_arg.epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

filepath = in_arg.save_dir

torch.save(checkpoint, filepath)