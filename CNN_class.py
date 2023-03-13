import torch
import torch.nn as nn
import torch.nn.functional as F

class Digit_Classifier(nn.Module):
  def __init__(self):
    super(Digit_Classifier, self).__init__()

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
    self.output = nn.Linear(32 * 7 * 7, 10)

  def forward(self, x):
    if x.shape == (28, 28):
      x = torch.reshape(x, (1,1,28,28))
    
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    output = self.output(x)

    return output