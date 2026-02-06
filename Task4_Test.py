import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AdalineAndLR import LogisticRegressionGD

"""
Intro to Machine Learning Assignment 1
Encompasses the testing of the solution to Task 4.
Students: Jackie Javier, e.t.c.
"""
# Preparing wine data
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                 header = None)
y = wine.iloc[0:129, 0].values
y = np.where(y == 1, 0, 1)
X = wine.iloc[0:129, 1:14].values
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Epochs and learning rate
test_n = 60
test_eta = 0.01

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))

#do GD here, plot time cost vs convergence speed

#do SGD here, plot time cost vs convergence speed

#do batch SGD here, plot time cost cs convergence speed

plt.tight_layout()
plt.show()
plt.savefig("task4.png")
