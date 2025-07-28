import torch

def train(model, x, y, optimizer, criterion, epochs=20):
    for epoch in range(epochs):
        yPred = model(x)
        loss = criterion(yPred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model