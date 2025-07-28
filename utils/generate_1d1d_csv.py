import torch
import numpy as np
import pandas as pd


def generate_1d1d_csv(train_size=1000, test_size=200):
    x_train = torch.linspace(0, 1, train_size)
    y_noice = torch.randn(train_size) * 0.1
    y_train = torch.sin(2 * np.pi * x_train) + 0.5 * torch.cos(5 * np.pi * x_train) + 0.3 * x_train**2 + y_noice
    
    x_test = torch.linspace(0.05, 0.95, test_size)
    y_noice_test = torch.randn(test_size) * 0.1
    y_test = torch.sin(2 * np.pi * x_test) + 0.5 * torch.cos(5 * np.pi * x_test) + 0.3 * x_test**2 + y_noice_test
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }
    
    