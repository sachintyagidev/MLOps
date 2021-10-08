import math
from ..mlops import digits
import os
import pickle
import numpy as np

def test_labelset_check():
    dateSetSize = 100
    testSplit = 0.20
    valSplit = 0.10

    expectedTestSize = round(dateSetSize * testSplit)
    expectedValSize = round(dateSetSize * valSplit)

    data, target = digits.preprocess(dateSetSize)
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, testSplit, valSplit)

    x_trianSize = len(X_train_s)
    y_trianSize = len(y_train_s)

    x_testSize = len(X_test_s)
    y_testSize = len(y_test_s)

    x_valSize = len(X_val_s)
    y_valSize = len(y_val_s)

    assert x_trianSize == y_trianSize
    assert x_testSize == y_testSize
    assert x_valSize == y_valSize

def test_create_split_100_lable():
    dateSetSize = 100
    testSplit = 0.20
    valSplit = 0.10

    expectedTestSize = round(dateSetSize * testSplit)
    expectedValSize = round(dateSetSize * valSplit)

    data, target = digits.preprocess(dateSetSize)
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, testSplit, valSplit)

    trianSize = len(y_train_s)
    testSize = len(y_test_s)
    valSize = len(y_val_s)

    assert trianSize + testSize + valSize == dateSetSize
    assert testSize == expectedTestSize
    assert valSize == expectedValSize
    assert trianSize == dateSetSize - (expectedTestSize + expectedValSize)

def test_create_split_9_lable():
    dateSetSize = 9
    testSplit = 0.20
    valSplit = 0.10

    expectedTestSize = round(dateSetSize * testSplit)
    expectedValSize = round(dateSetSize * valSplit)

    data, target = digits.preprocess(dateSetSize)
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, testSplit, valSplit)

    trianSize = len(y_train_s)
    testSize = len(y_test_s)
    valSize = len(y_val_s)

    assert trianSize + testSize + valSize == dateSetSize
    assert testSize == expectedTestSize
    assert valSize == expectedValSize
    assert trianSize == dateSetSize - (expectedTestSize + expectedValSize)

def test_create_split_100():
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

def test_model_not_corrupted():
    '''
    model saved in "test_model_writing" test case will be validated
    '''

    data, target = digits.preprocess()
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'

    loaded_model = pickle.load(open(filenameGama.format(gamaSet[0]), 'rb'))

    predicted = loaded_model.predict(X_test_s)

    assert len(predicted) == len(X_test_s)

def test_model_deterministic():
    '''
    model saved in "test_model_writing" test case will be validated
    '''

    data, target = digits.preprocess()
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'

    loaded_model = pickle.load(open(filenameGama.format(gamaSet[0]), 'rb'))

    predicted_1 = loaded_model.predict(X_test_s)

    predicted_2 = loaded_model.predict(X_test_s)

    assert np.array_equal(predicted_1, predicted_2)

def test_train_test_dimensionality():

    data, target = digits.preprocess()
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    trainDim = len(X_train_s[0])
    testDim = len(X_test_s[0])
    valDim = len(X_val_s[0])

    assert trainDim == testDim == valDim