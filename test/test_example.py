import math
from ..mlops import digits
import os

def test_create_split_100():
    '''
    generate data via preprocess function
    '''
    dateSetSize = 100
    testSplit = 0.20
    valSplit = 0.10

    expectedTestSize = round(dateSetSize * testSplit)
    expectedValSize = round(dateSetSize * valSplit)

    data, target = digits.preprocess(dateSetSize)
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, testSplit, valSplit)

    trianSize = len(X_train_s)
    testSize = len(X_test_s)
    valSize = len(X_val_s)

    assert trianSize + testSize + valSize == dateSetSize
    assert testSize == expectedTestSize
    assert valSize == expectedValSize
    assert trianSize == dateSetSize - (expectedTestSize + expectedValSize)

def test_create_split_9():
    '''
    generate data via preprocess function
    '''
    dateSetSize = 9
    testSplit = 0.20
    valSplit = 0.10

    expectedTestSize = round(dateSetSize * testSplit)
    expectedValSize = round(dateSetSize * valSplit)

    data, target = digits.preprocess(dateSetSize)
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, testSplit, valSplit)

    trianSize = len(X_train_s)
    testSize = len(X_test_s)
    valSize = len(X_val_s)

    assert trianSize + testSize + valSize == dateSetSize
    assert testSize == expectedTestSize
    assert valSize == expectedValSize
    assert trianSize == dateSetSize - (expectedTestSize + expectedValSize)
    

def test_model_writing():
    '''
    generate data via preprocess function
    '''
    data, target = digits.preprocess()
    
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'
    metricFile = 'model/metric_gama'

    '''
    Run classification experiment
    '''
    digits.trainAll(gamaSet, X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)

    assert os.path.isfile(filenameGama.format(gamaSet[0]))


def test_small_data_overfit_checking():
    '''
    generate data via preprocess function
    '''
    data, target = digits.preprocess(50)
    
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'
    metricFile = 'model/metric_gama'

    '''
    check train metric for classification experiment
    '''
    acc, f1 = digits.train(gamaSet[0], X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)
    
    assert acc > 0.99
    assert f1 > 0.99