import torch
import numpy as np
import pandas as pd

from generate_1d1d_csv import generate_1d1d_csv

seed = 615
torch.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":
    simulationData = generate_1d1d_csv(train_size=1000, test_size=200)
    trainDf = pd.DataFrame({'x': simulationData['x_train'].numpy(), 'y': simulationData['y_train'].numpy()})
    testDf = pd.DataFrame({'x': simulationData['x_test'].numpy(), 'y': simulationData['y_test'].numpy()})
    
    trainDf.to_csv('./data/train_1d1d.csv', index=False)
    testDf.to_csv('./data/test_1d1d.csv', index=False)
    print("CSV train_1d1d.csv 和 test_1d1d.csv")


# x_train = torch.linspace(0, 1, 1000)
# noise_train = torch.randn(1000) * 0.1
# y_train = (torch.sin(np.pi * x_train) > 0.5).float() + noise_train  # 二元分类目标

# x_test = torch.linspace(0.05, 0.95, 20)
# noise_test = torch.randn(20) * 0.1
# y_test = (torch.sin(np.pi * x_test) > 0.5).float() + noise_test

# dat_1dim = {
#     'x_train':x_train,
#     'y_train':y_train,
#     'x_test':x_test,
#     'y_test':y_test
# }

# x_train = torch.rand(1000, 2)
# y_train = (((torch.sin(np.pi * x_train[:, 0]) + torch.cos(np.pi * x_train[:, 1])) > 1.0).float() + torch.randn(1000) * 0.1)

# x_test = torch.rand(200, 2)
# y_test = (((torch.sin(np.pi * x_test[:, 0]) + torch.cos(np.pi * x_test[:, 1])) > 1.0).float() + torch.randn(200) * 0.1)

# dat_2dim = {
#     'x_train':x_train,
#     'y_train':y_train,
#     'x_test':x_test,
#     'y_test':y_test
# }
