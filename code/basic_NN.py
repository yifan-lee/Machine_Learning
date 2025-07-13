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

        hiddenLayerNode3_0 = (
            self.w01 * hiddenLayerNode1_1
            + self.w11 * hiddenLayerNode2_1
            + self.final_bias
        )

        output = F.relu(hiddenLayerNode3_0)
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

        hiddenLayerNode3_0 = (
            self.w01 * hiddenLayerNode1_1
            + self.w11 * hiddenLayerNode2_1
            + self.final_bias
        )

        output = F.relu(hiddenLayerNode3_0)
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
        output = F.relu(y1 @ self.W2 + self.B2)
        return output


class BetterNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.randn(1, 16))
        self.B1 = nn.Parameter(torch.randn(16))
        self.W2 = nn.Parameter(torch.randn(16, 16))
        self.B2 = nn.Parameter(torch.randn(16))
        self.W3 = nn.Parameter(torch.randn(16, 1))
        self.B3 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input.view(-1, 1)
        y1 = F.relu(x @ self.W1 + self.B1)
        y2 = F.relu(y1 @ self.W2 + self.B2)
        output = F.relu(y2 @ self.W3 + self.B3)
        return output
