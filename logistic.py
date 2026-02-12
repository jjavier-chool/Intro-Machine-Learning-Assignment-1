import numpy as np
class LogisticRegressionGD:
    def __init__(self, eta=0.05, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w_ = np.zeros(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self.w_ += self.eta * X.T.dot(errors)
            loss = -np.mean(y * np.log(output + 1e-8) +
                            (1 - y) * np.log(1 - output + 1e-8))
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

