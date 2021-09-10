import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
from sklearn.metrics import f1_score

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gamaSet = [0.0001 ,0.001 , 0.01, 0.1, 1.0]

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=.15, shuffle=False)

X_train, X_val, y_train, y_val = train_test_split(data, digits.target, test_size=0.15, shuffle=False)

print('Train size: ' + str(len(X_train)) + ', ' + 'Test size: ' + str(len(X_test)) + ', ' + "Val size: " + str(len(X_val)) + '\n')

bestGamma = None
bestAcc = 0
bestF1 = 0
bestCLF = None

for gama in gamaSet:
    clf = svm.SVC(gamma=gama)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_val)
    acc = clf.score(X_val, y_val)
    f1 = f1_score(y_val, predicted, average='macro')

    if bestAcc < acc and bestF1 < f1:
        bestAcc = acc
        bestF1 = f1
        bestGamma = gama
        bestCLF = clf

    print('Gamma :' + str(gama))

    print('Result on validation set')
    print('Accuracy : ' + str(acc) + ', F1 Score :' + str(f1) + '\n')


print('\nBest Gamma is : ' + str(bestGamma) + ' with accuracy of ' + str(bestAcc) + ' and F1 score of ' + str(bestF1) + '\n')

predicted = bestCLF.predict(X_test)
print('Result on Test set')
print('Accuracy : ' + str(bestCLF.score(X_test, y_test)) + ', F1 Score :' + str(f1_score(y_test, predicted, average='macro')) + '\n')

predicted = bestCLF.predict(X_train)
print('Result on Train set')
print('Accuracy : ' + str(bestCLF.score(X_train, y_train)) + ', F1 Score :' + str(f1_score(y_train, predicted, average='macro')) + '\n')
