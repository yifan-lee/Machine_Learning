import torch
from torch.utils.data import DataLoader, TensorDataset

def train(model, x, y=None, optimizer=None, criterion=None, epochs=20, patience=20, device='mps'):
    if y is None:
        dataLoader = x
    else:
        dataLoader = DataLoader(
            TensorDataset(x, y),
            batch_size=32,
            shuffle=True
        )
    best_loss = float("inf")
    counter = 0
    
    model.train()
    for epoch in range(epochs):
        for x, y in dataLoader:
            x = x.to(device)
            y = y.to(device)
            yPred = model(x)
            optimizer.zero_grad()
            loss = criterion(yPred, y)
            loss.backward()
            optimizer.step()
            
        # early stopping
        current_loss = loss.item()
        if current_loss < best_loss - 1e-6:  # small threshold to avoid float issues
            best_loss = current_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return model