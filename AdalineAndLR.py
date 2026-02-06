import numpy as np

"""
Intro to Machine Learning Assignment 1
Encompasses the solution to Task 1 and the function
implementation for Task 4.
Students: Jackie Javier, e.t.c.
Built from code provided by the textbook.
"""
class AdalineGD:
    """ADAptive LInear NEuron classifer.

    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random
        weight initialization.

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting. Bias scalar b is absorbed.
    losses_ : list
        Mean squared error loss function values in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ------------
        X : {array-like}, shape = {n_examples, n_features}
            Training vectors, where n_examples
            is the number of examples and
            n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : Instance of AdalineGD

        """

        # Absorbing the bias. hstack will stack the original with
        # a column of ones.
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,
                              scale=0.01, size=X_bias.shape[1])
        #self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X_bias)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X_bias.T.dot(errors) / X_bias.shape[0]
            #self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) #+ self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.where(self.activation(
            self.net_input(X_bias)) >= 0.5, 1, 0)


class LogisticRegressionGD:
    """Gradient descent-based logisitc regression classifier.

    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random
        weight initialization.

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting. Bias scalar b is absorbed.
    losses_ : list
        Mean squared error loss function values in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X : {array-like}, shape = {n_examples, n_features}
            Training vectors, where n_examples
            is the number of examples and
            n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        ------------
        self : Instance of LogisticRegressionGD

        """

        # Absorbing the bias.
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,
                              scale=0.01, size=X_bias.shape[1])
        #self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X_bias)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X_bias.T.dot(errors) / X_bias.shape[0]
            #self.b_ += self.eta * 2.0 * errors.mean()
            loss = ((-y.dot(np.log(output))) - ((1-y).dot(np.log(1-output)))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) #+ self.b_

    def activation(self, z):
        """Compute logisitc sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.where(self.activation
                        (self.net_input(X_bias)) >= 0.5, 1, 0)
