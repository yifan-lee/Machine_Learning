import torch

def evaluate(model, x, y, criterion):
    yPred = model(x)
    loss = criterion(yPred, y)
    return loss