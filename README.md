
# Probabilistic modelling notebooks


[![Project Status](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Travis](https://travis-ci.org/dirmeier/probabilistic-modelling-notebooks.svg?branch=master)](https://travis-ci.org/dirmeier/probabilistic-modelling-notebooks)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dirmeier/probabilistic-modelling-notebooks/master)

A collection of Jupyter notebooks on Probabilistic Models.

## Introduction

The repository hosts some `jupyter notebooks` on probabilistic models in Python using  `Edward`, `PyMC3`, `GPy` and `PyStan`. The first few notebooks follow [Rasmussen and Williams (2006)](http://www.gaussianprocess.org/gpml/) and introduce simple Gaussian process models. The next couple of notebooks will be on other models such as splines, hierarchical models or generative models.

**I do not take warranty for the correctness or completeness of these documents.**

## Notebooks

- [01 Intro to Bayesian linear regression](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/tree/master/01-bayesian_regression.ipynb) introduces the concept of *Bayesian inference* using a linear regression example and how we *move* from putting distributions on parameters to putting distributions on functions.
- [02 Gaussian process regression](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/02-gaussian_process_regression.ipynb) introduces non-parametric Bayesian regression.
- [03 Gaussian process classification](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/03-gaussian_process_classification.ipynb) extends Gaussian process regression to classification scenarios. 

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier@web.de</a>
