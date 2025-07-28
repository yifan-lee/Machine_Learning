import pandas as pd


def combine_train_test_data(train, test):
    trainDF = pd.DataFrame(train.numpy())
    trainDF['labels'] = 'train'
    
    testDF = pd.DataFrame(test.numpy())
    testDF['labels'] = 'test'

    combinedDF = pd.concat([trainDF, testDF], axis=0)
    return combinedDF