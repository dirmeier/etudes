# Deep Single Cell

Deep networks for learning pathogen infection from single cell features in Python.

## Introduction

In his iconic, nay epic, book [*Algorithms on Strings, Trees and Sequences*](http://www.cambridge.org/gb/academic/subjects/computer-science/algorithmics-complexity-computer-algebra-and-computational-g/algorithms-strings-trees-and-sequences-computer-science-and-computational-biology?format=HB&isbn=9780521585194) (1997),
Dan Gusfield considers MSA to be one of the most important topics, if not the holy grail, in computational biology (if you don't have a copy get one asap). In the early 21. century the grail has (imho) been passed to analysis of large-scale RNA interference screens.
This repository contains a benchmark if glorious deep nets might be worth considering for analysis of a large-scale RNAi screening data set of single cell and pathogen features.
The benchmark is done against a random forest and a logistic model.

## Installation

You can install all dependencies using:

```bash
  pip install -r requirements.txt
```

That should let you run the notebook.

## Benchmark

**Disclaimer: I am not really into deep learning so how I trained the model might be sub-optimal**. The network is created like this:

* 3 hidden layers with 20/30 nodes each, 20% dropout and rectifier activation functions,
* sigmoid activation for output layer,
* stochastic gradient descent with 1% learning rate, 1e-4 learning rate decay, 0.9 Nesterov momentum for optimization,
* cross-entropy as loss function.

The data consists of a set of image based single cell features of dimension `(1m x 18)` as predictors and a binary label if a cell is infected with a pathogen or not as a response.
The benchmark is done against

* a random forest consisting of 100 trees, each having `sqrt(18)` depth,
* a logistic model with ridge penalty.

It looks as if there is no clear winner and all perform equally well with the deep-net having a slight advantage.
Given that the net is non-parametric and computationally more expensive, I'll probably stick with the logit model.
I guess the setting/architecture/etc. of the neural net is just not optimal here.

<div align="center">
<p><b>Accuracy of neural net, random forest and logistic model.</b></p>
<img src="https://rawgit.com/dirmeier/machine-learning-notebooks/develop/single-cell-deep-net/_fig/training.png" alt="Drawing" width="100%" />
</div>

## Author

* Simon Dirmeier <a href="simon.dirmeier@web.de">simon.dirmeier@web.de</a>
