from operator import truediv
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np
import math


def knn(k: int, dataset: tuple, classification: bool):
    '''
    Applies KNN classification to the given dataset. 
    Will split the dataset into training/test sets.

    Inputs:
        k: integer representing how many neighbors to apply 
            KNN on
        dataset: generated dataset from sklearn.datasets
        classification: whether or not this is a classification
            problem
    
    Outputs:
        X_train: the part of the dataset used to train the model
        X_test: the part of the dataset used to test the model
        y_train: the corresponding classifications of the training data
        y_test: the corresponding classifications of the test data
        classified_test: the classified test data using the model
        classified_train: the classified training data using the model
    '''
    # Generated data points
    X = dataset[0]
    X = StandardScaler().fit_transform(X)
    # Classification of the data point at corresponding index
    y = dataset[1]
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size=.2,
                                                        random_state=42)

    # Classifies test data based on model
    classified_test = []
    for i in range(X_test.shape[0]):
         # Finds distance between given point and all points
        distances = []
        for j in range(X_train.shape[0]):
            distances.append(euclideandist(X_train[j, 0], X_train[j, 1], X_test[i, 0], X_test[i, 1]))

        # Determines the indices of the k nearest neighbors
        nearest_neighbors = []
        for i in range(k):
            minelementidx = distances.index(min(distances))
            nearest_neighbors.append(minelementidx)
            distances[minelementidx] = math.inf

        # Different results for classification vs. regression problem
        if classification:
            prediction = round(knn_classify(nearest_neighbors, y_train))
        else:
            prediction = knn_classify(nearest_neighbors, y_train)
        classified_test.append(prediction)
    
    # Classifies training data based on model
    classified_train = []
    for i in range(X_train.shape[0]):
         # Finds distance between given point and all points
        distances = []
        for j in range(X_train.shape[0]):
            distances.append(euclideandist(X_train[j, 0], X_train[j, 1], X_train[i, 0], X_train[i, 1]))

        # Determines the indices of the k nearest neighbors
        nearest_neighbors = []
        for i in range(k):
            minelementidx = distances.index(min(distances))
            nearest_neighbors.append(minelementidx)
            distances[minelementidx] = math.inf

        # Different results for classification vs. regression problem
        if classification:
            prediction = round(knn_classify(nearest_neighbors, y_train))
        else:
            prediction = knn_classify(nearest_neighbors, y_train)
        classified_train.append(prediction)

    return X_train, X_test, y_train, y_test, classified_test, classified_train

def euclideandist(x1, y1, x2, y2):
    '''
    Returns the 2-norm, or Euclidean distance between two points

    Inputs:
        x1: x coordinate of the first point
        y1: y coordinate of the first point
        x2: x coordinate of the second point
        y2: y coordinate of the second point

    Outputs:
        The Euclidean distance between the two points
    '''
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def knn_classify(nearest_neighbors, y_train):
    '''
    Returns the classifications of the points

    Inputs:
        nearest_neighbors: the indices of the k-nearest neighbors 
            of the points we want to classify
        y_train: the classified dataset
    '''
    values = [y_train[idx] for idx in nearest_neighbors]
    return mean(values)

def plotAll(X_train, X_test, y_train, y_test, classified_test, cm_bright, k, fname):
    '''
    Plots all the data
    '''
    # Visualize the datasets
    figure = plt.figure(figsize=(27, 9))
    ax = plt.subplot(1, 3, 1)
    ax.scatter(X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                cmap=cm_bright,
                s=200,
                edgecolors='k',
                alpha=0.5
                )
    plt.title(f'Training Set', fontsize=18)

    ax = plt.subplot(1, 3, 2)
    ax.scatter(X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                s=200,
                edgecolors='k',
                alpha=0.5
                )
    plt.title(f'Test Set Actual Classifications', fontsize=18)

    ax = plt.subplot(1, 3, 3)
    ax.scatter(X_test[:, 0],
                X_test[:, 1],
                c=classified_test,
                cmap=cm_bright,
                s=200,
                edgecolors='k',
                alpha=0.5
                )

    plt.title(f'Test Set Predicted Classifications with KNN Classification, k = {k}', fontsize=18)
    figure.savefig(fname=fname)

def run_classification(cm_bright, linearly_separable, moons):
    '''
    Runs K-Nearest Neighbors classification
    '''
    # Make model for all K and classifies the training and test datasets with both implemented model and given KNNClassifier
    K = np.arange(1, 160)
    knnlinseperror_train = []
    pyliblinseperror_train = []
    knnlinseperror_test = []
    pyliblinseperror_test = []
    knnmoonserror_train = []
    pylibmoonserror_train = []
    knnmoonserror_test = []
    pylibmoonserror_test = []
    for i in K:
        # Implemented KNN
        X_train, X_test, y_train, y_test, classified_test, classified_train = knn(i, linearly_separable, True)
        if (i % 5 == 0) or (i == 1):
            plotAll(X_train, X_test, y_train, y_test, classified_test, cm_bright, i, f'classlinsep{i}')
        knnlinseperror_train.append(1 - accuracy_score(y_train, classified_train))
        knnlinseperror_test.append(1 - accuracy_score(y_test, classified_test))

        # kNN classifier
        clf = KNeighborsClassifier(i)
        # Build model
        clf.fit(X_train, y_train)
        pyliblinseperror_train.append(1 - accuracy_score(y_train, clf.predict(X_train)))
        pyliblinseperror_test.append(1 - accuracy_score(y_test, clf.predict(X_test)))

        # Implemented KNN
        X_train, X_test, y_train, y_test, classified_test, classified_train = knn(i, moons, True)
        if (i % 5 == 0) or (i == 1):
            plotAll(X_train, X_test, y_train, y_test, classified_test, cm_bright, i, f'classmoons{i}')
        knnmoonserror_train.append(1 - accuracy_score(y_train, classified_train))
        knnmoonserror_test.append(1 - accuracy_score(y_test, classified_test))

        # kNN classifier
        clf = KNeighborsClassifier(i)
        # Build model
        clf.fit(X_train, y_train)
        pylibmoonserror_train.append(1 - accuracy_score(y_train, clf.predict(X_train)))
        pylibmoonserror_test.append(1 - accuracy_score(y_test, clf.predict(X_test)))

    # Plots results
    fig, ax = plt.subplots(2, 2, figsize=(27, 27))
    ax[0, 0].invert_xaxis()
    ax[0, 1].invert_xaxis()
    ax[1, 0].invert_xaxis()
    ax[1, 1].invert_xaxis()
    ax = plt.subplot(2, 2, 1)
    plt.plot(K, knnlinseperror_train, label='KNN Implmentation Training Error')
    plt.plot(K, knnlinseperror_test, label='KNN Implementation Test Error')
    plt.plot(K, pyliblinseperror_train, label='KNeighborsClassifier Training Error')
    plt.plot(K, pyliblinseperror_test, label='KNeighborsClassifier Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('Classification Accuracy for Linearly Separable Dataset: KNN Implemented vs. KNeighborsClassifier')

    ax = plt.subplot(2, 2, 2)
    plt.plot(K[:20], knnlinseperror_train[:20], label='KNN Implmentation Training Error')
    plt.plot(K[:20], knnlinseperror_test[:20], label='KNN Implementation Test Error')
    plt.plot(K[:20], pyliblinseperror_train[:20], label='KNeighborsClassifier Training Error')
    plt.plot(K[:20], pyliblinseperror_test[:20], label='KNeighborsClassifier Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('Classification Accuracy for Linearly Separable Dataset (first 20 data points): KNN Implemented vs. KNeighborsClassifier')

    ax = plt.subplot(2, 2, 3)
    plt.plot(K, knnmoonserror_train, label='KNN Implmentation Training Error')
    plt.plot(K, knnmoonserror_test, label='KNN Implementation Test Error')
    plt.plot(K, pylibmoonserror_train, label='KNeighborsClassifier Training Error')
    plt.plot(K, pylibmoonserror_test, label='KNeighborsClassifier Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('Classification Accuracy for Make Moons Dataset: KNN Implemented vs. KNeighborsClassifier')

    ax = plt.subplot(2, 2, 4)
    plt.plot(K[:20], knnmoonserror_train[:20], label='KNN Implmentation Training Error')
    plt.plot(K[:20], knnmoonserror_test[:20], label='KNN Implementation Test Error')
    plt.plot(K[:20], pylibmoonserror_train[:20], label='KNeighborsClassifier Training Error')
    plt.plot(K[:20], pylibmoonserror_test[:20], label='KNeighborsClassifier Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('Classification Accuracy for Make Moons Dataset (first 20 data points): KNN Implemented vs. KNeighborsClassifier')

    fig.savefig(fname='classificationaccuracies')

def run_regression(cm_bright, linearly_separable, moons):
    '''
    Runs K-Nearest Neighbors regression
    '''
    # Make model for all K and classifies the training and test datasets with both implemented model and given KNNClassifier
    K = np.arange(1, 160)
    knnlinseperror_train = []
    pyliblinseperror_train = []
    knnlinseperror_test = []
    pyliblinseperror_test = []
    knnmoonserror_train = []
    pylibmoonserror_train = []
    knnmoonserror_test = []
    pylibmoonserror_test = []
    for i in K:
        # Implemented KNN
        X_train, X_test, y_train, y_test, classified_test, classified_train = knn(i, linearly_separable, False)
        if (i % 5 == 0) or (i == 1):
            plotAll(X_train, X_test, y_train, y_test, classified_test, cm_bright, i, f'reglinsep{i}')
        knnlinseperror_train.append(mse(y_train, classified_train))
        knnlinseperror_test.append(mse(y_test, classified_test))

        # kNN classifier
        clf = KNeighborsRegressor(i)
        # Build model
        clf.fit(X_train, y_train)
        pyliblinseperror_train.append(mse(y_train, clf.predict(X_train)))
        pyliblinseperror_test.append(mse(y_test, clf.predict(X_test)))

        # Implemented KNN
        X_train, X_test, y_train, y_test, classified_test, classified_train = knn(i, moons, False)
        if (i % 5 == 0) or (i == 1):
            plotAll(X_train, X_test, y_train, y_test, classified_test, cm_bright, i, f'regmoons{i}')
        knnmoonserror_train.append(mse(y_train, classified_train))
        knnmoonserror_test.append(mse(y_test, classified_test))

        # kNN classifier
        clf = KNeighborsRegressor(i)
        # Build model
        clf.fit(X_train, y_train)
        pylibmoonserror_train.append(mse(y_train, clf.predict(X_train)))
        pylibmoonserror_test.append(mse(y_test, clf.predict(X_test)))

    # Plots results
    fig, ax = plt.subplots(2, 2, figsize=(27, 27))
    ax[0, 0].invert_xaxis()
    ax[0, 1].invert_xaxis()
    ax[1, 0].invert_xaxis()
    ax[1, 1].invert_xaxis()
    ax = plt.subplot(2, 2, 1)
    plt.plot(K, knnlinseperror_train, label='KNN Implmentation Training Error')
    plt.plot(K, knnlinseperror_test, label='KNN Implementation Test Error')
    plt.plot(K, pyliblinseperror_train, label='KNeighborsRegressor Training Error')
    plt.plot(K, pyliblinseperror_test, label='KNeighborsRegressor Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('KNN Regression Accuracy for Linearly Separable Dataset: KNN Implemented vs. KNeighborsRegressor')

    ax = plt.subplot(2, 2, 2)
    plt.plot(K[:20], knnlinseperror_train[:20], label='KNN Implmentation Training Error')
    plt.plot(K[:20], knnlinseperror_test[:20], label='KNN Implementation Test Error')
    plt.plot(K[:20], pyliblinseperror_train[:20], label='KNeighborsRegressor Training Error')
    plt.plot(K[:20], pyliblinseperror_test[:20], label='KNeighborsRegressor Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('KNN Regression Accuracy for Linearly Separable Dataset (first 20 data points): KNN Implemented vs. KNeighborsRegressor')

    ax = plt.subplot(2, 2, 3)
    plt.plot(K, knnmoonserror_train, label='KNN Implmentation Training Error')
    plt.plot(K, knnmoonserror_test, label='KNN Implementation Test Error')
    plt.plot(K, pylibmoonserror_train, label='KNeighborsRegressor Training Error')
    plt.plot(K, pylibmoonserror_test, label='KNeighborsRegressor Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('KNN Regression Accuracy for Make Moons Dataset: KNN Implemented vs. KNeighborsRegressor')

    ax = plt.subplot(2, 2, 4)
    plt.plot(K[:20], knnmoonserror_train[:20], label='KNN Implmentation Training Error')
    plt.plot(K[:20], knnmoonserror_test[:20], label='KNN Implementation Test Error')
    plt.plot(K[:20], pylibmoonserror_train[:20], label='KNeighborsRegressor Training Error')
    plt.plot(K[:20], pylibmoonserror_test[:20], label='KNeighborsRegressor Test Error')
    plt.xlabel('Model Complexity in terms of k')
    plt.ylabel('Model Error')
    plt.legend()
    plt.title('KNN Regression Accuracy for Make Moons Dataset (first 20 data points): KNN Implemented vs. KNeighborsRegressor')

    fig.savefig(fname='regressionaccuracies')

def mse(ds1, ds2):
    '''
    Computes the MSE of the two datasets (ds1, ds2)
    '''
    return (np.square(ds1 - ds2)).mean()


# Necessary initializations, generates datasets
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples=200)
linearly_separable = (X, y)
moons = make_moons(noise=0.3, random_state=0, n_samples=200)

run_classification(cm_bright, linearly_separable, moons)
run_regression(cm_bright, linearly_separable, moons)