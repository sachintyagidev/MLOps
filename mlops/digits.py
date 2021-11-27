import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
from sklearn.metrics import f1_score
import pickle
from sklearn import tree
from enum import Enum
import base64
from statistics import mean
from sklearn.neural_network import MLPClassifier

class Classifier(Enum):
    SVM = 1
    DecisionTree = 2


digits = datasets.load_digits()
images = digits.images
target = digits.target

n_samples = len(images)
data = images.reshape((n_samples, -1))

def preprocess(size = 0):
    digits = datasets.load_digits()

    if size > 0:
        images = digits.images[0:size]
        target = digits.target[0:size]
    else:
        images = digits.images
        target = digits.target

    n_samples = len(images)
    data = images.reshape((n_samples, -1))
    #print(n_samples)

    return data, target

def create_splits(data, target, test_size, val_size):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(val_size / (1 - test_size)))

    #print('Train size: ' + str(len(X_train)) + ', ' + 'Test size: ' + str(len(X_test)) + ', ' + "Val size: " + str(len(X_val)) + '\n')

    return X_train, X_test, y_train, y_test, X_val, y_val

def test(clf, X_val, y_val):
    predicted = clf.predict(X_val)
    acc = clf.score(X_val, y_val)
    f1 = f1_score(y_val, predicted, average='macro')

    return acc, f1

result = {}

def trainExam(hyperParameter, X_train, y_train, X_val, y_val, X_test_s, y_test_s, clfType = Classifier.SVM):
    clf = None
    if clfType == Classifier.SVM:
        clf = svm.SVC(gamma=hyperParameter)
    elif clfType == Classifier.DecisionTree:
        clf = tree.DecisionTreeClassifier(max_depth=hyperParameter)

    clfMLP = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    mlpTest = clfMLP.score(X_test_s, y_test_s)
    mlpTrain = clfMLP.score(X_train, y_train)
    mlpVal = clfMLP.score(X_val, y_val)

    print('MPL Result : Train : ' + str(mlpTrain) + ' Test : ' + str(mlpTest) + ' Validation : ' + str(mlpVal))

    filenameGama = 'modelExam/model_{}.sav'

    clf.fit(X_train, y_train)
    val_acc, val_f1 = test(clf, X_val, y_val)
    train_acc, train_f1 = test(clf, X_train, y_train)
    test_acc, test_f1 = test(clf, X_test_s, y_test_s)
    
    pickle.dump(clf, open(filenameGama.format(hyperParameter), 'wb'))
    return val_acc, val_f1, train_acc, train_f1, test_acc, test_f1

def trainHyperparameter(gama):
    valAcc = []
    testAcc = []
    trainAcc = []

    print('Hyperparameter : ' + str(gama))
    for i in range(0,3):
        testSplit = 0.15
        valSplit = 0.15
        data, target = preprocess()
            
        X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = create_splits(data, target, testSplit, valSplit)

        val_acc, val_f1, train_acc, train_f1, test_acc, test_f1 = trainExam(gama, X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
        print('Run ' + str(i + 1) + ", Train : " + str(train_acc) + ", Test : " + str(test_acc) + ", Validation : " + str(val_acc))
        valAcc.append(val_acc)
        testAcc.append(test_acc)
        trainAcc.append(train_acc)
        val_acc, val_f1, train_acc, train_f1, test_acc, test_f1 = 0,0,0,0,0,0

    print('Mean :' + 'Train : ' + str(mean(trainAcc)) + ", Test : " + str(mean(testAcc)) + ", Validation : " + str(mean(valAcc))+ '\n')

def mainExam():

    gamaSet = [10 ** exponent for exponent in range(-4, -1)]
    result['Hyperparameter'] = gamaSet
    for gama in gamaSet:
        trainHyperparameter(gama)

    print('Done')
mainExam()