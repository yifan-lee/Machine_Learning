import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, featureDim=1, hiddenDim=128, outputDim=1):
        super().__init__()
        self.rnn = nn.RNN(featureDim, hiddenDim, batch_first=True)
        self.fc = nn.Linear(hiddenDim, outputDim)
    def forward(self, x):
        _, hidden = self.rnn(x)       # out shape: [batch, seq, hidden]
        lastHidden = hidden[-1]        # 取最后一步的hidden
        return self.fc(lastHidden).squeeze()
    
    
class RNN_TS(nn.Module):
    def __init__(self, featureDim=1, hiddenDim=128, outputDim=1):
        super().__init__()
        self.rnn = nn.RNN(featureDim, hiddenDim, batch_first=True)
        self.fc = nn.Linear(hiddenDim, outputDim)
    def forward(self, x):
        _, hidden = self.rnn(x)       # out shape: [batch, seq, hidden]
        lastHidden = hidden[-1]        # 取最后一步的hidden
        return self.fc(lastHidden).squeeze()