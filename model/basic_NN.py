import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # activation function ReLU


class BasicNN_baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(
            torch.tensor(1.7), requires_grad=False
        )  # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(-16.0), requires_grad=False)

    def forward(self, input):
        hiddenLayerNode1_0 = input * self.w00 + self.b00
        hiddenLayerNode1_1 = F.relu(hiddenLayerNode1_0)

        hiddenLayerNode2_0 = input * self.w10 + self.b10
        hiddenLayerNode2_1 = F.relu(hiddenLayerNode2_0)

        output = F.relu(
            self.w01 * hiddenLayerNode1_1
            + self.w11 * hiddenLayerNode2_1
            + self.final_bias
        )

        return output


class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.randn(1), requires_grad=True)  # don't optims it
        self.b00 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w01 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w10 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b10 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w11 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.final_bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, input):
        hiddenLayerNode1_0 = input * self.w00 + self.b00
        hiddenLayerNode1_1 = F.relu(hiddenLayerNode1_0)

        hiddenLayerNode2_0 = input * self.w10 + self.b10
        hiddenLayerNode2_1 = F.relu(hiddenLayerNode2_0)

        output = (
            self.w01 * hiddenLayerNode1_1
            + self.w11 * hiddenLayerNode2_1
            + self.final_bias
        )

        return output


class SimplifiedNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, 2))
        self.B1 = nn.Parameter(torch.randn(2))
        self.W2 = nn.Parameter(torch.randn(2, 1))
        self.B2 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        output = (y1 @ self.W2 + self.B2)
        return output
    
class BetterNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = 8
        self.W1 = nn.Parameter(torch.randn(1, hidden_dim))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        self.B1 = nn.Parameter(torch.randn(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, 1))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.B2 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        output = (y1 @ self.W2 + self.B2)
        return output



class flixNN(nn.Module):
    def __init__(self, dim1=16,dim2=16,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W2 = nn.Parameter(torch.randn(dim1, dim2))
        self.B2 = nn.Parameter(torch.randn(dim2))
        self.W3 = nn.Parameter(torch.randn(dim2, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        y2 = F.relu(y1 @ self.W2 + self.B2)
        output = (y2 @ self.W3 + self.B3)
        # output = (y2 @ self.W3 + self.B3)
        return output
    
class flixNN2(nn.Module):
    def __init__(self, dim1=2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, dim1))
        self.B1 = nn.Parameter(torch.randn(dim1))
        self.W3 = nn.Parameter(torch.randn(dim1, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        output = (y1 @ self.W3 + self.B3)
        return output
    
    