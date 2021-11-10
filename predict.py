from get_predict_args import get_predict_args
in_arg = get_predict_args()

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
import json

with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
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

    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model  



fielpath = in_arg.chkpoint
model = load_checkpoint(fielpath)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    # TODO: Process a PIL image for use in a PyTorch model
    process = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    return process(pil_image)
    
    
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk= in_arg.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if in_arg.device == 'cpu':
        device = 'cpu'
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Implement the code to predict the class from an image file
    im = process_image(image_path)
    im.unsqueeze_(0)
    model.to(device).float()
    
    #Put the model into evaluation mode
    model.eval()
    
    #Make the prediction (forward pass)
    with torch.no_grad():
        pred = model.forward(im.to(device))
    
    #Take the exponential of the output (which is in log_softmax)
    ps = torch.exp(pred)
    
    #Take top-k classifications
    top_p, top_class = ps.topk(topk, dim=1)
    
    
    classes = top_class.tolist()[0]
    
    pred_classes = []
    for value in classes:
        pred_classes.append(list(model.class_to_idx)[value]) 

    return top_p, pred_classes

if in_arg.device == 'cpu':
    device = 'cpu'
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


path = in_arg.path_to_image
image_path = path    
# flower_name = image_path.split('/')[2]
#  print(flower_name)

# title_flower = str(cat_to_name[flower_name])
# img = process_image(image_path)
#  print(title_flower)

probs, labels = predict(image_path, model)
probs = probs.tolist()[0]
print(probs)
labels_named = []
for i in range(len(labels)):
        labels[i] = cat_to_name[str(labels[i])]
        labels_named.append(labels[i])
    
print(labels_named)    