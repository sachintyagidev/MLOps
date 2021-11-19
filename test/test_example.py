import math
from ..mlops import digits
import os
import pickle
import numpy as np

'''
def test_labelset_check():
    dateSetSize = 100
    testSplit = 0.20
    valSplit = 0.10

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

    digits.trainAll(gamaSet, X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)

    assert os.path.isfile(filenameGama.format(gamaSet[0]))


def test_small_data_overfit_checking():
    data, target = digits.preprocess(50)
    
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'
    metricFile = 'model/metric_gama'

    acc, f1 = digits.train(gamaSet[0], X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)
    
    assert acc > 0.99
    assert f1 > 0.99

def test_model_not_corrupted():
    
    data, target = digits.preprocess()
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'

    loaded_model = pickle.load(open(filenameGama.format(gamaSet[0]), 'rb'))

    predicted = loaded_model.predict(X_test_s)

    assert len(predicted) == len(X_test_s)

def test_model_deterministic():
    
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


def test_DT_acc():
    data, target = digits.preprocess()
    
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-3, -2)]
    filenameGama = 'model/model_{}.sav'
    metricFile = 'model/metric_gama'

    acc, f1 = digits.train(gamaSet[0], X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)

    accDT, f1DT = digits.train(None, X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile, digits.Classifier.DecisionTree)

    print(accDT, f1DT)
    print(acc, f1)

    assert accDT > 0.79
    assert f1DT > 0.79

    assert acc > 0.99
    assert f1 > 0.99

def test_model_multiple_train():
    data, target = digits.preprocess()
    
    
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

    gamaSet = [10 ** exponent for exponent in range(-7, -1)]
    filenameGama = 'model/model_svm_{}.sav'
    metricFile = 'model/metric_gama'

    digits.trainAll(gamaSet, X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)

    print('Best Gama Value: ' + str(digits.searchBestModel(metricFile)))
    
    depthSet = [2, 4, 6, 8, 10, 12, 14, 16]
    filenameGama = 'model/model_DT_{}.sav'
    metricFile = 'model/metric_max_depth'

    digits.trainAll(depthSet, X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile, digits.Classifier.DecisionTree)
    print('Best Depth Value: ' + str(digits.searchBestModel(metricFile)))
'''

loaded_svm_model = pickle.load(open('./model/model_svm_0.0001.sav', 'rb'))
loaded_dt_model = pickle.load(open('./model/model_DT_14.sav', 'rb'))
data, target = digits.preprocess()
min_acc_req = 0.87
#X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = digits.create_splits(data, target, 0.15, 0.15)

def test_digit_correct_0():
    result = np.where(target == 0)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)

    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==0
    assert predictedDT==0
    

def test_digit_correct_1():
    result = np.where(target == 1)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==1
    assert predictedDT==1

def test_digit_correct_2():
    result = np.where(target == 2)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==2
    assert predictedDT==2

def test_digit_correct_3():
    result = np.where(target == 3)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==3
    assert predictedDT==3

def test_digit_correct_4():
    result = np.where(target == 4)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==4
    assert predictedDT==4

def test_digit_correct_5():
    result = np.where(target == 5)
    image = data[result[0][1]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==5
    assert predictedDT==5

def test_digit_correct_6():
    result = np.where(target == 6)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==6
    assert predictedDT==6

def test_digit_correct_7():
    result = np.where(target == 7)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==7
    assert predictedDT==7

def test_digit_correct_8():
    result = np.where(target == 8)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==8
    assert predictedDT==8

def test_digit_correct_9():
    result = np.where(target == 9)
    image = data[result[0][0]]

    image = np.array(image).reshape(1, -1)
    
    predictedSVM = loaded_svm_model.predict(image)
    predictedDT = loaded_dt_model.predict(image)
    accSVM, f1 = digits.test(loaded_svm_model, data[result[0]], target[result[0]])
    accDT, f1 = digits.test(loaded_dt_model, data[result[0]], target[result[0]])

    assert accSVM > min_acc_req
    assert accDT > min_acc_req
    assert predictedSVM==9
    assert predictedDT==9