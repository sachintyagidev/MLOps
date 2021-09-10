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

currSize = digits.images[0].shape
print('Current Image size :' + str(currSize))

print('Current data set size :' + str(digits.images.shape))

resizeSet = [(4,4) ,(8,8) ,(16,16)]
testSizeSet = [0.5 ,0.3 , 0.2]
gamaSet = [0.0001 ,0.001 , 0.01, 0.1, 1.0]

for size in resizeSet:
    new_digits = np.array(list
                            (map
                            (lambda img: resize(
                                            img.reshape(currSize),
                                            size,
                                            mode='constant'),
                digits.images)))

    print('')
    print('Changed data set size :' + str(new_digits.shape))

    # flatten the images
    data = new_digits.reshape((n_samples, -1))

    for gama in gamaSet:
        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=gama)

        for testSize in testSizeSet:
            # Split data into 50% train and 50% test subsets
            X_train, X_test, y_train, y_test = train_test_split(
                data, digits.target, test_size=testSize, shuffle=False)

            # Learn the digits on the train subset
            clf.fit(X_train, y_train)

            # Predict the value of the digit on the test subset
            predicted = clf.predict(X_test)

            print('Train Set size : ' + str(testSize) + ', Gamma :' + str(gama) +', Accuracy : ' + str(clf.score(X_test, y_test)) + ', F1 Score :' + str(f1_score(y_test, predicted, average='macro')))