import torch

def train(model, x, y, optimizer, criterion, epochs=20, patience=20):
    best_loss = float("inf")
    counter = 0
    for epoch in range(epochs):
        yPred = model(x)
        loss = criterion(yPred, y)
        optimizer.zero_grad()
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