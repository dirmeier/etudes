# Études

[![Project Status](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Travis](https://travis-ci.org/dirmeier/etudes.svg?branch=master)](https://travis-ci.org/dirmeier/etudes)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dirmeier/etudes/master)

:notes: A collection of numerical études.

## About

The repository hosts some numerical etudes in the form of `jupyter` notebooks.
Te notebooks are on probabilistic models, machine learning, optimization, numerical etudes, algorithms, etc...

**I do not take warranty for the correctness or completeness of these.**

## Études on regression

- [01 Intro to Bayesian linear regression](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/bayesian_regression.ipynb) introduces the concept of *Bayesian inference* using a linear regression example and how we *move* from putting distributions on parameters to putting distributions on functions.
- [01a Bayesian Lasso](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/bayesian_lasso.ipynb) explains how the frequentist LASSO can be interpreted as conditional Laplace prior on the regression coefficients.
- [02 Gaussian process regression](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/gaussian_process_regression.ipynb) introduces non-parametric Bayesian regression.
- [03 Gaussian process classification](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/gaussian_process_classification.ipynb) *extends* Gaussian process regression to classification scenarios.
- [04 Student-t process regression](https://nbviewer.jupyter.org/github/dirmeier/probabilistic-modelling-notebooks/blob/master/t_process_regression.ipynb) uses a *t-process* instead of a GP for regression.

## Études on graphical models

- [Kalman  filter](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/extended_kalman_filter.ipynb) shows the application of undirected graphical models to expression data.
- [Gaussian Graphical Models](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/gaussian_graphical_models.ipynb) implements the *graphical LASSO* for sparse precision matrix estimation.
- [Gaussian mixture model](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/gaussian_mixture_model.ipynb) implements a Gaussian mixture model using Expectation Maximization, and using Variational inference and Monte Carlo estimation.
- [Pairwise binary Markov random fields](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/pb-mrf.ipynb) shows the application of undirected graphical models to expression data.
- [Particle filter](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/particle_filter.ipynb) introduces recursive estimation using the Kalman and particle filters.

## Études on numerical methods

- [Conjugate Gradients](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/conjugate_gradients.ipynb) is an implementation of a numerical solver for linear equations with positive-definite matrices in Python.
- [QR-decomposition](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/qr_decomposition.ipynb) can be used for solving a linear regression model instead of the analytical solution. This is supposedly more stable. The implementation is in Python.

## Études on machine learning

- [Deep Drama](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/deep_drama.ipynb) implements a long short-term memory network for creating Greek drama. It uses drama from Euripides, Sophocles, Aristophanes and
 Aischylos from the Gutenberg project to train a recurrent neural network and then uses the trained model to *write* drama. In that sense it acts similar to other sequence models, just like HMMs. The network is implemented in R's `Keras` interface.
- [Non-linear component analysis](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/non_linear_component_analysis.ipynb) does an approximation to a kernel PCA using Nystroem features (Williams *et al*. (2001))
and random Fourier features (Rahimi *et al* (2008)). The advantage of both is reduced runtime and approximately as good results as a full kernel-PCA. The implementation is in Python.
- [Single cell autoencoder](https://github.com/dirmeier/etudes/tree/master/sc-denoising-autoencoder.ipynb) uses deep-convolutional autoencoders to extract features from single-cell imaging data.
- [Single cell GAN](https://github.com/dirmeier/etudes/tree/master/sc-gan.ipynb) uses DC-Wasserstein-GANs to create artificial single-cell microscopy images.

## Études on optimization

- [SVM](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/svm.ipynb) is an implementation of the Lagrangian dual formulation of the usual hinge-loss for SVMs in Python.
- [Regularized regression](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/regularized_regression.ipynb) shows how disciplined convex programming can be used to solve a regularized regression model (`cvxpy`).


## Études on other stuff

- [Lady tasting tea](https://nbviewer.jupyter.org/github/dirmeier/etudes/blob/master/lady_tasting_tea..ipynb) is a Bayesian analysis of Fisher's *Lady testing tea* experiment



## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier@web.de</a>
