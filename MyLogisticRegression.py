import logging

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='public/my_log.log', level=logging.DEBUG)
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

def do_it_with(data_set):
    x_train, x_test, y_train, y_test = acquire_data(data_name=data_set)
    y = np.append(y_train, y_test)
    model = LogisticRegression()
    y_raveled = np.ravel(y_train)
    model = model.fit(x_train, y_raveled)
    logging.debug("here is the model score for " + data_set + ": ")
    logging.debug(model.score(x_test, np.ravel(y_test)))


def main():
    do_it_with('synthetic-easy')
    do_it_with('synthetic-medium')
    do_it_with('synthetic-hard')


if __name__ == "__main__":
    main()
