import torch

def evaluate(model, x, y, criterion):
    yPred = model(x)
    loss = criterion(yPred, y)
    return loss



def evaluate_CNN(model, x, y=None, criterion=None, device='mps'):
    if y is None:
        dataLoader = x
    else:
        dataLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=32,
            shuffle=False
        )
        
    model.eval()
    totalCount = 0
    correctCount = 0
    with torch.no_grad():
        for x, y in dataLoader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correctCount += (pred == y).sum().item()
            totalCount += y.size(0)
    return correctCount / totalCount if totalCount > 0 else 0