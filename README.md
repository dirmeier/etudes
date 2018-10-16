
# Probabilistic modelling notebooks


[![Project Status](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Travis](https://travis-ci.org/dirmeier/probabilistic-modelling-notebooks.svg?branch=master)](https://travis-ci.org/dirmeier/probabilistic-modelling-notebooks)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dirmeier/probabilistic-modelling-notebooks/master)

A collection of Jupyter notebooks on Probabilistic Models.

## Introduction

The repository hosts some `jupyter notebooks` on probabilistic models, such as

* hierarchical models,
* (non-parametric) Bayesian models,
* graphical models.

The notebooks use probabilistic languages/packages such as `Edward`, `PyMC3`, `GPy`, `Stan` and `Greta`, or are entirely from scratch. All of them are only serving as a personal collection.

**I do not take warranty for the correctness or completeness of these documents.**

### Notebooks

Bayesian regression:

- [01 Intro to Bayesian linear regression](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/tree/master/01-bayesian_regression.ipynb) introduces the concept of *Bayesian inference* using a linear regression example and how we *move* from putting distributions on parameters to putting distributions on functions.
- [01a Bayesian Lasso](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/01a-bayesian_lasso.ipynb) explains how the frequentist LASSO can be interpreted as conditional Laplace prior on the regression coefficients.
- [02 Gaussian process regression](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/02-gaussian_process_regression.ipynb) introduces non-parametric Bayesian regression.
- [03 Gaussian process classification](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/03-gaussian_process_classification.ipynb) *extends* Gaussian process regression to classification scenarios.
- [04 Student-t process regression](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/04-t_process_regression.ipynb) uses a *t-process* instead of a GP for regression.

Graphical models:

- [Bayesian filtering](https://github.com/dirmeier/probabilistic-modelling-notebooks/tree/master/bayesian-filtering.ipynb) introduces recursive estimation using the Kalman and particle filters.
- [Gaussian Graphical Models](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/gaussian_graphical_models.ipynb) implements the *graphical LASSO* for sparse precision matrix estimation.
- [Pairwise binary Markov random fields](https://github.com/dirmeier/probabilistic-modelling-notebooks/blob/master/pb-mrf.ipynb) shows the application of undirected graphical models to expression data.
- [Gaussian mixture model](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-programming-notebooks/blob/master/gaussian_mixture_model.ipynb) implements a Gaussian mixture model using Expectation Maximization, and using Variational inference and Monte Carlo estimation.


## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier@web.de</a>
