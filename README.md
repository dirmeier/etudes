
# Gaussian process notebooks


[![Project Status](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Travis](https://travis-ci.org/dirmeier/gaussian-process-notebooks.svg?branch=master)](https://travis-ci.org/dirmeier/gaussian-process-notebooks)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dirmeier/gaussian-process-notebooks/master)

 A collection of Jupyter notebooks on Gaussian processes.

## Introduction

The repository hosts some `jupyter notebooks` on Gaussian processes in Python using `GPy` and `scipy`. The first few introductory chapters follow [Rasmussen and Williams (2006)](http://www.gaussianprocess.org/gpml/). The rest are from recent publications on Gaussian processes, i. e.

- scaling to big data problems,
- distributed computation,
- Gaussian process latent variable models,
- Gaussian process regression networks,
- Student-t processes, 
- variational inference and sampling,
- mixtures of Gaussian processes,
- kernel approximations,
- sparse Gaussian processes.

**I do not take warranty for the correctness or completeness of these documents.**

## Notebooks

- [01 Intro to Bayesian linear regression](https://nbviewer.jupyter.org/github/dirmeier/gaussian-process-notebooks/tree/master/01-bayesian_regression.ipynb) introduces the concept of *Bayesian inference* using a linear regression example and how we *move* from putting distributions on parameters to putting distributions on functions.
- [02 Gaussian process regression](https://nbviewer.jupyter.org/github/dirmeier/gaussian-process-notebooks/blob/master/02-gaussian_process_regression.ipynb) introduces non-parametric Bayesian regression.
- [03 Gaussian process classification](https://nbviewer.jupyter.org/github/dirmeier/gaussian-process-notebooks/blob/master/03-gaussian_process_classification.ipynb) extends Gaussian process regression to classification scenarios. 

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier@web.de</a>
