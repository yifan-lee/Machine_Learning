import torch
import torch.nn as nn
import torch.nn.functional as F # activation function ReLU


class BasicNN_baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7),requires_grad=False) # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7),requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(-16.),requires_grad=False)
        
    def forward(self, input):
        hiddenLayerNode1_0 = input*self.w00+self.b00
        hiddenLayerNode1_1 = F.relu(hiddenLayerNode1_0)
        
        hiddenLayerNode2_0 = input*self.w10+self.b10
        hiddenLayerNode2_1 = F.relu(hiddenLayerNode2_0)
        
        hiddenLayerNode3_0 = self.w01*hiddenLayerNode1_1+self.w11*hiddenLayerNode2_1+self.final_bias
        
        output = F.relu(hiddenLayerNode3_0)
        return output
    
    
class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7),requires_grad=True) # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85),requires_grad=True)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=True)
        self.w10 = nn.Parameter(torch.tensor(12.6),requires_grad=True)
        self.b10 = nn.Parameter(torch.tensor(0.),requires_grad=True)
        self.w11 = nn.Parameter(torch.tensor(2.7),requires_grad=True)
        self.final_bias = nn.Parameter(torch.tensor(-16.),requires_grad=True)
        
    def forward(self, input):
        hiddenLayerNode1_0 = input*self.w00+self.b00
        hiddenLayerNode1_1 = F.relu(hiddenLayerNode1_0)
        
        hiddenLayerNode2_0 = input*self.w10+self.b10
        hiddenLayerNode2_1 = F.relu(hiddenLayerNode2_0)
        
        hiddenLayerNode3_0 = self.w01*hiddenLayerNode1_1+self.w11*hiddenLayerNode2_1+self.final_bias
        
        output = F.relu(hiddenLayerNode3_0)
        return output