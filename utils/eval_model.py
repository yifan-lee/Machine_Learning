import torch

def evaluate(model, x, y, criterion, device='mps'):
    x = x.to(device)
    y = y.to(device)
    model.to(device)
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
    model.to(device)
    model.eval()
    totalCount = 0
    correctCount = 0
    wrongIndexes = []
    with torch.no_grad():
        for batchIndex, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correctCount += (pred == y).sum().item()
            totalCount += y.size(0)
            for i in range(len(pred)):
                if pred[i] != y[i]:
                    wrongIndex = batchIndex * dataLoader.batch_size + i
                    wrongIndexes.append((wrongIndex, pred[i].item(), y[i].item()))
    correctPercent = correctCount / totalCount if totalCount > 0 else 0
    return correctPercent, wrongIndexes


def evaluate_RNN(model, x, y=None, criterion=None, device='mps'):
    if y is None:
        dataLoader = x
    else:
        dataLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=32,
            shuffle=False
        )
    model.to(device)
    model.eval()
    totalCount = 0
    correctCount = 0
    wrongIndexes = []
    with torch.no_grad():
        for batchIndex, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = (torch.sigmoid(output) > 0.5).float()
            correctCount += (pred == y).sum().item()
            totalCount += y.size(0)
            for i in range(len(pred)):
                if pred[i] != y[i]:
                    wrongIndex = batchIndex * dataLoader.batch_size + i
                    wrongIndexes.append((wrongIndex, pred[i].item(), y[i].item()))
    correctPercent = correctCount / totalCount if totalCount > 0 else 0
    return correctPercent, wrongIndexes