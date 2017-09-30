## Foreword
This repository is my attempt to re-do all the assignments in the [Machine Learning course](https://www.coursera.org/learn/machine-learning/) on [Coursera](https://coursera.com) using Python language. The data is therefore mostly taken from this course and is in `./data` folder

There are many reasons for doing this:
* In the course, you always have a very detailed instruction for each assignment with "already written" functions. Now I'm trying to code it from scratch to have a better understanding of Machine Learning.

* To enhance my Python skills: learning Python is important to move to advanced topics after finishing the course, and this is a good starting. It also offers me a chance to learn about so-called scientific Python packages: [`numpy`](http://www.numpy.org/), [`matplotlib`](http://matplotlib.org/) and [`scikit-learn`](http://scikit-learn.org/stable/).

## Contents

### Linear Regression

* **univariate.py**: 1 feature input X (the output Y is continuous value), use built-in Linear Regression model from `scikit-learn` package, plot input X and output line with `matplotlib`.

* **multiple.py**: multiple features input X (the output Y is continuous value), plot input X and output hyperplane in 3D with `matplotlib`, use built-in Linear Regression model.

### Logistic Regression

* **classification.py**: 2 features input X (the output Y is discrete value 0/1), **the data is almost Linearly Separable**, plot scatter input X and decision boundary, apply Feature Normalization, no use built-in model (self written Gradient Descent).

* **classification2.py**: 2 features input X (the output Y is discrete value 0/1), **the data is not Linearly Separable**, plot scatter input X and decision boundary, apply Feature Normalization and Feature Mapping.

* **classification-packed.py**: same as classification2.py, but use built-in Logistic Regression model from `scikit-learn` package.

_Notes: **classification2.py** and **classification-packed.py** offer two different ways to plot scatter data and non linear decision boundary._

### Multi-class Classification

* **one-vs-all.py**: 400 features input X (the output Y is discrete value 1-10), use self written Logistic Regression model to classify, plot a grid of grayscale image of digits.

* **one-vs-all-packed.py**: Same as **one-vs-all.py**, but use built-in Logistic Regression model from `scikit-learn`.
