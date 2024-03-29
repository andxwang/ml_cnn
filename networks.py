import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(torch.tanh(self.conv1(x)))
    x = self.pool(torch.tanh(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = torch.softmax(self.fc3(x), dim=1)
    return x
