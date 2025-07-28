import torch

def train(model, dataloader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        for X, y in dataloader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model