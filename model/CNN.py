import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),  # 2D convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 2D max pooling
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 14 * 14, 64),  # Adjust input size based on the output of conv layer
            nn.ReLU(),
            nn.Linear(64, 26)  # Output layer for 26 classes
        )
        
    def forward(self, x):
        x = self.conv(x) 
        x = self.fc(x)
        return x