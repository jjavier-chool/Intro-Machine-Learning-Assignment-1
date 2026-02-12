import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from adaline import AdalineGD
from logistic import LogisticRegressionGD

iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(iris_url, header=None)
iris = iris[iris[4].isin(['Iris-setosa', 'Iris-versicolor'])]
X_iris = iris.iloc[:, 0:4].values
y_iris = np.where(iris.iloc[:, 4] == 'Iris-setosa', 0, 1)

wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine = pd.read_csv(wine_url, header=None)
wine = wine[wine[0].isin([1, 2])]
X_wine = wine.iloc[:, 1:].values
y_wine = np.where(wine[0]==1, 0, 1)

def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

X_iris = standardize(X_iris)
X_wine = standardize(X_wine)
eta = 0.01
epochs = 100
ada_iris = AdalineGD(eta=eta, n_iter=epochs)
ada_iris.fit(X_iris, y_iris)

lr_iris = LogisticRegressionGD(eta=eta, n_iter=epochs)
lr_iris.fit(X_iris, y_iris)
ada_wine = AdalineGD(eta=eta, n_iter=epochs)
ada_wine.fit(X_wine, y_wine)
lr_wine = LogisticRegressionGD(eta=eta, n_iter=epochs)
lr_wine.fit(X_wine, y_wine)

plt.figure()
plt.plot(range(1, epochs+1), ada_iris.losses_, label='Iris')
plt.plot(range(1, epochs+1), ada_wine.losses_, label='Wine')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Adaline Loss Convergence")
plt.legend()
plt.savefig("task2_ada.png")
plt.close()
plt.figure()
plt.plot(range(1, epochs+1), lr_iris.losses_, label='Iris')
plt.plot(range(1, epochs+1), lr_wine.losses_, label='Wine')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Loss Convergence")
plt.legend()
plt.savefig("task2_lr.png")
plt.close()
plt.figure()
plt.plot(range(1, epochs+1), ada_iris.losses_, label='Adaline (Iris)')
plt.plot(range(1, epochs+1), lr_iris.losses_, label='Logistic (Iris)')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adaline vs Logistic Regression (Iris)")
plt.legend()
plt.savefig("task2_compare.png")
plt.close()
