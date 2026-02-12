import numpy as np

class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # Add bias column (column of ones)
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Initialize weights INCLUDING bias weight
        self.w_ = np.zeros(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = net_input
            errors = y - output
            self.w_ += self.eta * X.T.dot(errors)

            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.where(self.net_input(X) >= 0.0, 1, -1)
