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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Classifier(Enum):
    SVM = 1
    DecisionTree = 2


digits = datasets.load_digits()
images = digits.images
target = digits.target

n_samples = len(images)
data = images.reshape((n_samples, -1))

def preprocess(percent = 0):
    digits = datasets.load_digits()
    totalImage = len(digits.images)

    if percent > 0:
        size = int(totalImage * percent / 100)
        images = digits.images[0:size]
        target = digits.target[0:size]
    else:
        images = digits.images
        target = digits.target

    n_samples = len(images)
    data = images.reshape((n_samples, -1))

    return data, target

def create_splits(data, target, test_size, val_size):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(val_size / (1 - test_size)))
    return X_train, X_test, y_train, y_test, X_val, y_val

def test(clf, X_val, y_val):
    predicted = clf.predict(X_val)
    acc = clf.score(X_val, y_val)
    f1 = f1_score(y_val, predicted, average='macro')

    return acc, f1

macroF1 = []
bestModel = []
filenameGama = 'modelAss11/model_{}_{}.sav'
testSplit = 0.10
valSplit = 0.10

def train(hyperParameter, datasetSize, X_train, y_train, X_val, y_val, X_test_s, y_test_s, clfType = Classifier.SVM):
    clf = None
    if clfType == Classifier.SVM:
        clf = svm.SVC(gamma=hyperParameter)
    elif clfType == Classifier.DecisionTree:
        clf = tree.DecisionTreeClassifier(max_depth=hyperParameter)

    clf.fit(X_train, y_train)
    val_acc, val_f1 = test(clf, X_val, y_val)
    train_acc, train_f1 = test(clf, X_train, y_train)
    test_acc, test_f1 = test(clf, X_test_s, y_test_s)
    
    pickle.dump(clf, open(filenameGama.format(datasetSize, hyperParameter), 'wb'))
    return val_acc, val_f1, train_acc, train_f1, test_acc, test_f1

def trainHyperparameter(datasize):
    data, target = preprocess(datasize)

    testF1 = []
    gamaSet = [10 ** exponent for exponent in range(-6, -1)]
    bestF1 = 0
    bestGama = None
    for gama in gamaSet:
        X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = create_splits(data, target, testSplit, valSplit)

        val_acc, val_f1, train_acc, train_f1, test_acc, test_f1 = train(gama, datasize, X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)
        if bestF1 < test_f1:
            bestF1 = test_f1
            bestGama = gama
        print('\t Gama : ' + str(gama) + "\t Train : " + str(train_f1) + "\t Test : " + str(test_f1) + "\t Validation : " + str(val_f1))
        testF1.append(test_f1)
        val_acc, val_f1, train_acc, train_f1, test_acc, test_f1 = 0,0,0,0,0,0
    
    print('\t Dataset size :' + str(datasize) + '\t macro f1 : ' + str(mean(testF1)))
    macroF1.append(mean(testF1))
    bestModel.append(bestGama)

def mainAss11():

    for size in range(10, 110, 10):
        print('Dataset size:' + str(size))
        trainHyperparameter(size)
    
    plt.plot(range(10, 110, 10), macroF1)
    plt.xlabel("Dataset Size")
    plt.ylabel("macro f1")
    plt.show()

    data, target = preprocess()
    X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = create_splits(data, target, testSplit, valSplit)

    for size in range(1, 10, 1):
        idx1 = size
        size1 = 10*(size+1)

        idx2 = size - 1
        size2 = size*10

        loadedModel1 = pickle.load(open(filenameGama.format(size1, bestModel[idx1]), 'rb'))
        loadedModel2 = pickle.load(open(filenameGama.format(size2, bestModel[idx2]), 'rb'))

        acc_1, f1_1 = test(loadedModel1, X_test_s, y_test_s)
        acc_2, f1_2 = test(loadedModel2, X_test_s, y_test_s)

        predicted1 = loadedModel1.predict(X_test_s)
        predicted2 = loadedModel2.predict(X_test_s)

        conf1 = confusion_matrix(y_test_s, predicted1)
        conf2 = confusion_matrix(y_test_s, predicted2)

        print('\nModel :' + str(size1) + ' and ' + str(size2))
        print('F1 : ' + str(f1_1) + ' and ' + str(f1_2))

        ax1 = sns.heatmap(conf1, annot=True, cmap='Blues')
        ax1.set_title('Confusion Matrix of model size : {} \n\n'.format(size1));
        #plt.show()

        ax2 = sns.heatmap(conf2, annot=True, cmap='Blues')
        ax2.set_title('Confusion Matrix of model size : {} \n\n'.format(size2));
        plt.show()
        #print('Confusion Matrix :')
        #print(conf1)
        #print('\n')
        #print(conf2)

        #pickle.dump(clf, open(, 'wb'))
    print('Done')

mainAss11()