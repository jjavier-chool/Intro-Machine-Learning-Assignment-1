# Intro-Machine-Learning-Assignment-1
The first assignment for CS429. Initial implementations of AdalineGD and LogisticRegressionGD are taken from the textbook.

## TODO:
- Task 3 fixing (everything wants to be virginica...)
- Task 4 modification to AdalineAndLR.py + new test file
- Report on Overleaf

## How to Run
For Tasks 2-4, run:
```
python Task2_Test.py
```
```
python Task3_Test.py
```
```
python Task4_Test.py
```
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
