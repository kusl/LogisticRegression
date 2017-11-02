import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split


def sigmoid(X):
    '''Compute the sigmoid function '''
    # d = zeros(shape=(X.shape))
    den = 1.0 + numpy.e ** (-1.0 * X)
    d = 1.0 / den
    return d


# computes cost given predicted and actual values
def compute_cost(theta, X, y):
    # number of training examples
    m = X.shape[0]
    theta = numpy.reshape(theta, (len(theta), 1))
    # y = reshape(y,(len(y),1))
    J = (1. / m) * (
        - numpy.transpose(y).dot(numpy.log(sigmoid(X.dot(theta)))) - numpy.transpose(1 - y).dot(
            numpy.log(1 - sigmoid(X.dot(theta)))))
    grad = numpy.transpose((1. / m) * numpy.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    # optimize.fmin expects a single value, so cannot return grad
    return J[0][0]  # ,grad


def compute_grad(theta, x, y):
    # print theta.shape
    theta.shape = (1, 3)
    grad = numpy.zeros(3)
    h = sigmoid(x.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sum_delta = delta.T.dot(x[:, i])
        grad[i] = (1.0 / h) * sum_delta * - 1
    theta.shape = (3,)
    print(grad)
    return grad


def main():
    x, y = datasets.make_classification(
        n_features=2, n_redundant=0, n_informative=2, n_classes=2, random_state=1, n_clusters_per_class=1)
    rng = numpy.random.RandomState(2)
    x += 0 * rng.uniform(size=x.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=42)
    print("Here comes x 기차")
    compute_grad(theta=numpy.random.RandomState(2), x=x_train, y=y_train)


if __name__ == "__main__":
    main()
