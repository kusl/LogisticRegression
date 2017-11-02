import numpy

def sigmoid(X):
    '''Compute the sigmoid function '''
    # d = zeros(shape=(X.shape))
    den = 1.0 + numpy.e ** (-1.0 * X)
    d = 1.0 / den
    return d


def compute_cost(theta, X, y):  # computes cost given predicted and actual values
    m = X.shape[0]  # number of training examples
    theta = numpy.reshape(theta, (len(theta), 1))
    # y = reshape(y,(len(y),1))
    J = (1. / m) * (
        - numpy.transpose(y).dot(numpy.log(sigmoid(X.dot(theta)))) - numpy.transpose(1 - y).dot(
            numpy.log(1 - sigmoid(X.dot(theta)))))
    grad = numpy.transpose((1. / m) * numpy.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    # optimize.fmin expects a single value, so cannot return grad
    return J[0][0]  # ,grad


def compute_grad(theta, X, y):
    # print theta.shape
    theta.shape = (1, 3)
    grad = numpy.zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / h) * sumdelta * - 1
    theta.shape = (3,)
    print(grad)
    return grad
