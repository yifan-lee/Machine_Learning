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
    
    
class nn_dim3c1_flix(nn.Module):
    def __init__(self, dim1=2, dim2=2, num_classes=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(2, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W2 = nn.Parameter(torch.randn(dim1, dim2))
        self.B2 = nn.Parameter(torch.randn(dim2))
        self.W3 = nn.Parameter(torch.randn(dim2, num_classes))
        self.B3 = nn.Parameter(torch.randn(num_classes))
        self.num_classes = num_classes

    def forward(self, input):
        x = input.view(-1, 2)
        y1 = F.relu(x @ self.W1 + self.B1)
        y2 = F.relu(y1 @ self.W2 + self.B2)
        output = F.softmax(y2 @ self.W3 + self.B3, dim=1)
        return output
