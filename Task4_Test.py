import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AdalineAndLR import LogisticRegressionGD

"""
Intro to Machine Learning Assignment 1
Encompasses the testing of the solution to Task 4.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
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

# GD
start = time.time()
gd = LogisticRegressionGD(n_iter=test_n, eta=test_eta).fit(X_std, y)
end = time.time()
t_gd = end - start

# SGD
start = time.time()
sgd = LogisticRegressionGD(n_iter=test_n, eta=test_eta).fit_SGD(X_std, y)
end = time.time()
t_sgd = end - start

# Mini-batch SGD
start = time.time()
mb = LogisticRegressionGD(n_iter=test_n, eta=test_eta).fit_mini_batch_SGD(X_std, y, 32)
end = time.time()
t_mb = end - start

# Figures
plt.figure(figsize=(6, 4))
plt.bar(['GD', 'SGD', 'Mini-batch SGD'], [t_gd, t_sgd, t_mb])
plt.ylabel('Training time (seconds)')
plt.title('Training Time Comparison')
plt.tight_layout()
plt.show()
plt.savefig("task4_time.png")

plt.figure(figsize=(6, 4))
plt.plot(gd.losses_, label='GD')
plt.plot(sgd.losses_, label='SGD')
plt.plot(mb.losses_, label='Mini-batch SGD')
plt.xlabel('Epochs')
plt.ylabel('log Loss')
plt.title('LogisitcRegression Loss Convergence Comparison')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("task4_compare.png")
