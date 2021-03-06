#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

logging.basicConfig(filename='public/log', level=logging.DEBUG)
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


def acquire_data(data_name, number_of_classes_for_synthetic_data_set=2):
    global global_data_name
    global_data_name = data_name
    if data_name == 'synthetic-easy':
        logging.info('Creating easy synthetic labeled data set')
        x, y = datasets.make_classification(
            n_features=2, n_redundant=0, n_informative=2, n_classes=number_of_classes_for_synthetic_data_set,
            random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        x += 0 * rng.uniform(size=x.shape)
    elif data_name == 'synthetic-medium':
        logging.info('Creating medium synthetic labeled data set')
        x, y = datasets.make_classification(
            n_features=2, n_redundant=0, n_informative=2, n_classes=number_of_classes_for_synthetic_data_set,
            random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        x += 3 * rng.uniform(size=x.shape)
    elif data_name == 'synthetic-hard':
        logging.info('Creating hard easy synthetic labeled data set')
        x, y = datasets.make_classification(
            n_features=2, n_redundant=0, n_informative=2, n_classes=number_of_classes_for_synthetic_data_set,
            random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        x += 5 * rng.uniform(size=x.shape)
    elif data_name == 'moons':
        logging.info('Creating two moons data set')
        x, y = datasets.make_moons(noise=0.3, random_state=0)
    elif data_name == 'circles':
        logging.info('Creating two circles data set')
        x, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        logging.info('Loading iris data set')
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
    elif data_name == 'digits':
        logging.info('Loading digits data set')
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        logging.info('Loading breast cancer data set')
        bcancer = datasets.load_breast_cancer()
        x = bcancer.data
        y = bcancer.target
    else:
        logging.info('Cannot find the requested data_name')
        assert False
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=42)
    return x_train, x_test, y_train, y_test


def my_score(y, y_gt):
    assert len(y) == len(y_gt)
    return np.sum(y == y_gt) / float(len(y))


def draw_data(x_train, x_test, y_train, y_test, number_of_classes):
    h = .02
    x = np.vstack([x_train, x_test])
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5

    cm = plt.cm.jet
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train,
                cmap=cm, edgecolors='k', label='Training Data')
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm,
                edgecolors='k', marker='x', linewidth=3, label='Test Data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.title(s=get_data_name())
    # plt.show()
    plt.savefig('public/draw_data_' + get_data_name() + '.png')


def my_train_binary(x_train, y_train):
    print('Start training ...')
    number_of_features = x_train.shape[1]
    w_vector = np.zeros(number_of_features + 1)
    x_extended = np.hstack([x_train, np.ones([x_train.shape[0], 1])])
    w_t_x = np.dot(x_extended, w_vector)
    my_sigmoid = np.divide(1, np.add(1, np.exp(-w_t_x)))  # sigmoid (w_t_x)
    y_n = my_sigmoid.reshape(-1)
    loss2 = -(np.dot(y_train, np.log(y_n)) + np.dot(np.subtract(1, y_train), np.log(np.subtract(1, y_n))))
    d_loss = loss2
    eta = 1.01
    while d_loss > 0.05 or d_loss < 0:
        gradient_e = np.dot(np.subtract(y_n, y_train), x_extended)
        w_vector = np.subtract(w_vector, np.multiply(eta, gradient_e))
        w_t_x = np.dot(x_extended, w_vector)
        my_sigmoid = np.divide(1, np.add(1, np.exp(-w_t_x)))
        y_n = my_sigmoid.reshape(-1)
        y_n[y_n < 0.05] = .05
        y_n[y_n >= 1] = .95
        loss1 = loss2
        loss2 = -(np.dot(y_train, np.log(y_n)) + np.dot(np.subtract(1, y_train), np.log(np.subtract(1, y_n))))

        if abs(loss1 > loss2) < d_loss:
            eta = 1.01
        else:
            eta = -0.5
        d_loss = abs(loss1 - loss2)
    print('Finished training.')
    logging.debug(w_vector)
    return w_vector


def my_predict_binary(x, w):
    # fixme TODO
    assert len(w) == x.shape[1] + 1
    w_vec = np.reshape(w, (-1, 1))
    x_extended = np.hstack([x, np.ones([x.shape[0], 1])])
    y_pred = np.ravel(np.sign(np.dot(x_extended, w_vec)))
    return convert_negative_one_one_to_zero_one(y_pred)


# convert -1/1 to 0/1
def convert_negative_one_one_to_zero_one(y_pred):
    return np.maximum(np.zeros(y_pred.shape), y_pred)


# convert 0/1 to -1/1
def convert_zero_one_to_negative_one_one(y_gt):
    y = 2 * (y_gt - 0.5)
    return y


def draw_result_binary(x_train, x_test, y_train, y_test, w):
    h = .02
    x = np.vstack([x_train, x_test])
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(
            x_min, x_max, h), np.arange(
            y_min, y_max, h))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.figure(1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train,
                cmap=cm_bright, edgecolors='k', label='Training Data')
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
                edgecolors='k', marker='x', linewidth=3, label='Test Data')
    tmp_x = np.c_[xx.ravel(), yy.ravel()]
    z = my_predict_binary(tmp_x, w)
    z = z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, z, cmap=plt.cm.RdBu, alpha=.4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    y_predict = my_predict_binary(x_test, w)
    score = my_score(y_predict, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' %
                                            score).lstrip('0'), size=15, horizontalalignment='right')
    plt.title(s=get_data_name())
    # plt.show()
    plt.savefig('public/draw_result_binary_' + get_data_name() + '.png')


def my_train_multi(x_train, y_train):
    logging.info('Start training ...')
    # fixme TODO
    np.random.seed(100)
    number_of_features = x_train.shape[1]
    w = np.random.rand(number_of_features + 1)
    logging.info('Finished training.')
    return w


def my_predict_multi(x, w):
    # fixme TODO
    return np.zeros([x.shape[0], 1])


def do_it_with(data_set):
    x_train, x_test, y_train, y_test = acquire_data(data_set)
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
    train_score = my_score(y_train_predict, y_train)
    test_score = my_score(y_test_predict, y_test)
    logging.info("Training Score: ")
    logging.info(train_score)
    logging.info("Test Score: ")
    logging.info(test_score)
    logging.info('Good bye')


def main():
    # TODO zero out variables after done with do it?
    do_it_with('synthetic-easy')
    do_it_with('synthetic-medium')
    do_it_with('synthetic-hard')
    do_it_with('moons')
    do_it_with('circles')


if __name__ == "__main__":
    main()
