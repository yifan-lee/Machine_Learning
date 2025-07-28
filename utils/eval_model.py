import torch

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            out = model(X)
            total_loss += criterion(out, y).item()
    return total_loss / len(dataloader)