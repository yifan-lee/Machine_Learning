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
    
    

class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 26)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x