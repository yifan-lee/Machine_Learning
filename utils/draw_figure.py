import matplotlib.pyplot as plt
import random

def prepare_data_for_draw_CNN_incorrect_predictions(testDataset, wrongIndexes, n=5):
    sampled_wrong = random.sample(wrongIndexes, min(5, len(wrongIndexes)))
    xAll = []
    yAll = []
    preddictAll = []
    for idx in sampled_wrong:
        x, y = testDataset[idx[0]]
        pred = idx[1]
        xAll.append(x)
        yAll.append(y)
        preddictAll.append(pred)
    dataForFigure = {
        'x': xAll,
        'y': yAll,
        'predict': preddictAll
    }
    return dataForFigure

def draw_CNN_incorrect_predictions(data, figurePath,fileName):
    """
    Draws figures for the wrong predictions of the model on the test dataset.
    
    Args:
        model: The trained model to evaluate.
        testDataset: The dataset containing test images and labels.
        device: The device to run the model on (e.g., 'mps', 'cpu').
    """
    xAll = data['x']
    yAll = data['y']
    predictAll = data['predict']
    
    for i in range(len(xAll)):
        x = xAll[i]
        y = yAll[i]
        predict = predictAll[i]
        plt.figure()
        plt.imshow(x.squeeze(), cmap='gray')
        plt.title(f"True: {y}, Pred: {predict}")
        plt.axis('off')
        plt.savefig(f'{figurePath}/{fileName}_{i}.png')
        plt.close()
    
    return 0
    
