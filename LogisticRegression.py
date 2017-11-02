#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

global_data_name = "Not yet defined"


def get_data_name():
    return global_data_name


# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc=2):
    global global_data_name
    global_data_name = data_name
    if data_name == 'synthetic-easy':
        print('Creating easy synthetic labeled dataset')
        X, y = datasets.make_classification(
            n_features=2, n_redundant=0, n_informative=2, n_classes=nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print('Creating medium synthetic labeled dataset')
        X, y = datasets.make_classification(
            n_features=2, n_redundant=0, n_informative=2, n_classes=nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print('Creating hard easy synthetic labeled dataset')
        X, y = datasets.make_classification(
            n_features=2, n_redundant=0, n_informative=2, n_classes=nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print('Creating two moons dataset')
        X, y = datasets.make_moons(noise=0.3, random_state=0)
    elif data_name == 'circles':
        print('Creating two circles dataset')
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print('Loading iris dataset')
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print('Loading digits dataset')
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print('Loading breast cancer dataset')
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print('Cannot find the requested data_name')
        assert False
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test


# compare the prediction with grount-truth, evaluate the score
def myscore(y, y_gt):
    assert len(y) == len(y_gt)
    return np.sum(y == y_gt) / float(len(y))


# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):
    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm,
                edgecolors='k', marker='x', linewidth=3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.title(s=get_data_name())
    # plt.show()
    plt.savefig('public/draw_data.png')


####################################################
# binary label classification
# train the weight vector w
def my_train_binary(X_train, y_train):
    print('Start training ...')
    # fake code, only return a random vector
    np.random.seed(100)
    nfeatures = X_train.shape[1]
    w = np.random.rand(nfeatures + 1)
    # w = [-1,0,0]
    print('Finished training.')
    return w


# predict labels using the logistic regression model on any input set X
def my_predict_binary(X, w):
    # here is a fake implementation, you should replace it
    assert len(w) == X.shape[1] + 1
    w_vec = np.reshape(w, (-1, 1))
    x_extended = np.hstack([X, np.ones([X.shape[0], 1])])
    y_pred = np.ravel(np.sign(np.dot(x_extended, w_vec)))
    return convert_pred2gt_binary(y_pred)


# convert -1/1 to 0/1
def convert_pred2gt_binary(y_pred):
    return np.maximum(np.zeros(y_pred.shape), y_pred)


# convert 0/1 to -1/1
def convert_gt2pred_binary(y_gt):
    y = 2 * (y_gt - 0.5)
    return y


# draw results on 2D plan for binary classification
# use it for debugging
def draw_result_binary(X_train, X_test, y_train, y_test, w):
    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(
            x_min, x_max, h), np.arange(
            y_min, y_max, h))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.figure(1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                cmap=cm_bright, edgecolors='k', label='Training Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                edgecolors='k', marker='x', linewidth=3, label='Test Data')
    tmpX = np.c_[xx.ravel(), yy.ravel()]
    Z = my_predict_binary(tmpX, w)
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    y_predict = my_predict_binary(X_test, w)
    score = myscore(y_predict, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' %
                                            score).lstrip('0'), size=15, horizontalalignment='right')
    plt.title(s=get_data_name())
    # plt.show()
    plt.savefig('public/draw_result_binary.png')


def my_train_multi(X_train, y_train):
    print('Start training ...')
    # fake code, only return a random vector
    np.random.seed(100)
    nfeatures = X_train.shape[1]
    w = np.random.rand(nfeatures + 1)
    # w = [-1,0,0]
    print('Finished training.')
    return w


def my_predict_multi(X, w):
    return np.zeros([X.shape[0], 1])


def main():
    x_train, x_test, y_train, y_test = acquire_data('synthetic-easy')
    number_of_features = x_train.shape[1]
    number_of_training_data = x_train.shape[0]
    number_of_test_data = x_test.shape[0]
    y = np.append(y_train, y_test)
    number_of_classes = len(np.unique(y))
    draw_data(x_train, x_test, y_train, y_test, number_of_classes)
    if number_of_classes == 2:
        w_opt = my_train_binary(x_train, y_train)
        draw_result_binary(x_train, x_test, y_train, y_test, w_opt)
    else:
        w_opt = my_train_multi(x_train, y_train)
    if number_of_classes == 2:
        y_train_predict = my_predict_binary(x_train, w_opt)
        y_test_predict = my_predict_binary(x_test, w_opt)
    else:
        y_train_predict = my_predict_multi(x_train, w_opt)
        y_test_predict = my_predict_multi(x_test, w_opt)
    train_score = myscore(y_train_predict, y_train)
    test_score = myscore(y_test_predict, y_test)
    print('Training Score:', train_score)
    print('Test Score:', test_score)


if __name__ == "__main__":
    main()
