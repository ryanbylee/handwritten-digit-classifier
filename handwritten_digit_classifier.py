import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class Digit_Classifier(nn.Module):
  def __init__(self):
    #Handle some under-the-hood PyTorch stuff
    super(Digit_Classifier, self).__init__()
    
    #Now put your layers below in addition to any other member variables you need
    #
    #self.nn_layer1 = torch.nn.Linear(784,400)
    #self.nn_layer2 = torch.nn.Linear(400, 10)
    '''
    self.conv1 = torch.nn.Conv2d(1, 16, 5, 1, 2)
    self.conv2 = torch.nn.Conv2d(16, 32, 5, 1, 2)
    self.output = torch.nn.Linear(32 * 7 * 7, 10)
    '''
    self.conv1 = nn.Sequential(
      nn.Conv2d(
        in_channels=1,              
        out_channels=16,            
        kernel_size=5,              
        stride=1,                   
        padding=2,                  
      ),                              
      nn.ReLU(),                      
      nn.MaxPool2d(kernel_size=2),    
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(16, 32, 5, 1, 2),     
      nn.ReLU(),                      
      nn.MaxPool2d(2),                
    )
        # fully connected layer, output 10 classes
    self.output = nn.Linear(32 * 7 * 7, 10)

  def forward(self, x):
    #Now here you add your forward pass, e.g. how the layers fit together
    #Tips:
    # 1. Don't forget the ReLU layer when needed
    # 2. Consider normalization
    # 3. If you are getting errors look at your dimensions, dimension errors are very easy to make!
    # 4. CNN layers take in rectangular (or square) inputs and give rectangular (or square) outputs. Fully connected layers have input and output that are vectors, when you need to switch between the two consider using a flatten or reshape
    '''
    flattened = torch.reshape(x, (x.shape[0],784))
    sigmoid = torch.nn.Sigmoid()
    outputs_layer1 = self.nn_layer1(flattened)
    outputs = self.nn_layer2(sigmoid(outputs_layer1))
    return outputs
    
    x = self.conv1(x)
    x = nn.ReLU()
    #x = nn.MaxPool2d(2)

    x = self.conv2(x)
    x = nn.ReLU()
    x = nn.MaxPool2d(2)
    '''
    if x.shape == (28, 28):
      x = torch.reshape(x, (1,1,28,28))
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), 32 * 7 * 7)
    output = self.output(x)
    return output

  #Optional: any other member functions that you think would be helpful