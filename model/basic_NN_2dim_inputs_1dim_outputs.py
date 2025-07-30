import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU


class NN_dim2(nn.Module):
    def __init__(self, dim1=2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(2, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W2 = nn.Parameter(torch.randn(dim1, 1))
        self.B2 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 2)
        y1 = F.relu(x @ self.W1 + self.B1)
        output = F.relu(y1 @ self.W2 + self.B2)
        return output
    
    
class NN_dim2_layer2(nn.Module):
    def __init__(self, dim1=2,dim2=2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(2, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W2 = nn.Parameter(torch.randn(dim1, dim2))
        self.B2 = nn.Parameter(torch.randn(dim2))
        self.W3 = nn.Parameter(torch.randn(dim2, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 2)
        y1 = F.relu(x @ self.W1 + self.B1)
        y2 = F.relu(y1 @ self.W2 + self.B2)
        output = F.relu(y2 @ self.W3 + self.B3)
        return output


class NN_dim2_flixible_layer(nn.Module):
    def __init__(self, dims=[2,2],*args, **kwargs):
        super().__init__(*args, **kwargs)
        inputDim = 2
        layers = []
        for dim in dims:
            layers.append(nn.Linear(inputDim, dim))
            inputDim = dim
        self.layers = nn.ModuleList(layers)
        self.outputLayer = nn.Linear(inputDim, 1)

    def forward(self, input):
        x = input.view(-1, 2)
        for layer in self.layers:
            x = F.relu(layer(x))
        output = self.outputLayer(x)
        return output
    
    
class NN_dim2_flixible_layer_dropout(nn.Module):
    def __init__(self, dims=[2,2],dropoutRate=0.5,*args, **kwargs):
        super().__init__(*args, **kwargs)
        inputDim = 2
        layers = []
        for dim in dims:
            layers.append(nn.Linear(inputDim, dim))
            layers.append(nn.Dropout(dropoutRate))
            inputDim = dim
        self.layers = nn.ModuleList(layers)
        self.outputLayer = nn.Linear(inputDim, 1)

    def forward(self, input):
        x = input.view(-1, 2)
        for layer in self.layers:
            if layer.__class__.__name__ == 'Dropout':
                x = layer(x)
            else:
                x = F.relu(layer(x))
        output = self.outputLayer(x)
        return output
