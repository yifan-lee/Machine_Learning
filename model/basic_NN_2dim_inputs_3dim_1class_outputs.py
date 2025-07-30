import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU


class nn_dim3c1(nn.Module):
    def __init__(self, inputDim=3, num_classes=10, dims=[2,2], *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        self.inputDim = inputDim
        for dim in dims:
            layers.append(nn.Linear(inputDim, dim))
            inputDim = dim
        self.hiddenLayers = nn.ModuleList(layers)
        self.outputLayer = nn.Linear(inputDim, num_classes)

    def forward(self, input):
        x = input.view(-1, self.inputDim)
        for layer in self.hiddenLayers:
            x = F.relu(layer(x))
        output = self.outputLayer(x)
        return output
    
    
class nn_dim3c1_dropout(nn.Module):
    def __init__(self, inputDim=3, num_classes=10, dims=[2,2], dropoutRate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        self.inputDim = inputDim
        for dim in dims:
            layers.append(nn.Linear(inputDim, dim))
            layers.append(nn.Dropout(dropoutRate))
            inputDim = dim
        self.hiddenLayers = nn.ModuleList(layers)
        self.outputLayer = nn.Linear(inputDim, num_classes)

    def forward(self, input):
        x = input.view(-1, self.inputDim)
        for layer in self.hiddenLayers:
            if isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                x = F.relu(layer(x))
        output = self.outputLayer(x)
        return output
    
