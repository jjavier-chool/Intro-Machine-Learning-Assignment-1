# Intro-Machine-Learning-Assignment-1
The first assignment for CS429. Initial implementations of AdalineGD and LogisticRegressionGD are taken from the textbook.

## TODO:
-Task 1 modifications to AdalineAndLR.py
-Ensure Task2_Test.py is correct
-Task 3
-Task 4 modification to AdalineAndLR.py + new test file

## How to Run
For Task 2, run:
```
python Task2_Test.py
```
Because the data links given in the book return a 404, make sure that the data files are in the correct location (/data).

It is probably necessary to downgrade numpy or test in a separate environment with numpy 1.0 due to some jank between the book's code and numpy 2.0.

Environment method I used:
```console
python3 -m venv adaline-env
source adaline-env/bin/activate
pip install "numpy<2" pandas matplotlib
```
Deactive with:
```console
deactivate
```

Reactivate with:
```console
source adaline-env/bin/activate
```
