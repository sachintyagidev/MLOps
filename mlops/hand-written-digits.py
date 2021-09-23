import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
from sklearn.metrics import f1_score
import pickle
import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument('--testSize', help='Test Set Size', default=0.15, type=float)
parser.add_argument('--valSize', help='Validation Size', default=0.176, type=float)
parser.add_argument('--modelLoc', help='Model Save Location', default='model')
parser.add_argument('--metricLoc', help='Metric Save Location', default='model')
parser.add_argument('--gamaSet', help='Gamma List to Test', default=[10 ** exponent for exponent in range(-7, 0)], nargs='+', type=float)

args=parser.parse_args()

gamaSet = args.gamaSet
test_size = float(args.testSize)
val_size = float(args.valSize)
filenameGama = args.modelLoc + '/model_{}.sav'
metricFile = args.metricLoc + '/metric_gama'

def preprocess():
    digits = datasets.load_digits()

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    return data, digits.target

def create_splits(data, target, test_size, val_size):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False)

    print('Train size: ' + str(len(X_train)) + ', ' + 'Test size: ' + str(len(X_test)) + ', ' + "Val size: " + str(len(X_val)) + '\n')

    return X_train, X_test, y_train, y_test, X_val, y_val

def train(gama, X_train, y_train, X_val, y_val, filenameGama, metricFile):
    clf = svm.SVC(gamma=gama)
    clf.fit(X_train, y_train)

    acc, f1 = test(clf, X_val, y_val)

    validate(clf, gama, acc, f1)

def test(clf, X_val, y_val):
    predicted = clf.predict(X_val)
    acc = clf.score(X_val, y_val)
    f1 = f1_score(y_val, predicted, average='macro')

    return acc, f1

def validate(clf, gama, acc, f1):
    pickle.dump(clf, open(filenameGama.format(gama), 'wb'))
    file1 = open(metricFile, "a")
    file1.write('{}, {}, {}\n'.format(gama, acc, f1))
    file1.close()

    print('Gamma :' + str(gama))

    print('Result on validation set')
    print('Accuracy : ' + str(acc) + ', F1 Score :' + str(f1) + '\n')
    

def trainAll(gamaSet):
    for gama in gamaSet:
        train(gama, X_train_s, y_train_s, X_val_s, y_val_s, filenameGama, metricFile)

def searchBestModel(metricFile):
    bestGamma = None
    bestAcc = 0.0
    bestF1 = 0.0

    file1 = open(metricFile, 'r')
    Lines = file1.readlines()
 
    for line in Lines:
        x = line.split(",")
        acc= float(x[1])
        f1= float(x[2])
        if bestAcc < acc and bestF1 < f1:
            bestAcc = acc
            bestF1 = f1
            bestGamma = float(x[0])
    
    return bestGamma


def report(filenameGama, bestParameter):
    # load the model from disk
    loaded_model = pickle.load(open(filenameGama.format(bestParameter), 'rb'))

    predicted = loaded_model.predict(X_test_s)
    print('Result on Test set')
    print('Accuracy : ' + str(loaded_model.score(X_test_s, y_test_s)) + ', F1 Score :' + str(f1_score(y_test_s, predicted, average='macro')) + '\n')

    predicted = loaded_model.predict(X_train_s)
    print('Result on Train set')
    print('Accuracy : ' + str(loaded_model.score(X_train_s, y_train_s)) + ', F1 Score :' + str(f1_score(y_train_s, predicted, average='macro')) + '\n')


data, target = preprocess()

X_train_s, X_test_s, y_train_s, y_test_s, X_val_s, y_val_s = create_splits(data, target, test_size, val_size)

trainAll(gamaSet)

bestParameter = searchBestModel(metricFile)

print('Best gama : ' + str(bestParameter) + '\n')

report(filenameGama, bestParameter)