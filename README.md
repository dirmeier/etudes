# Études

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dirmeier/etudes/master)

:notes: A collection of études on probabilistic models.

## About

The repository hosts some notebooks on probabilistic models, such as Gaussian processes, graphical models, normalizing flows, and so on. The notebooks are mostly on topics I am interested in or papers I happen to come across.

**I do not take warranty for the correctness of these.**

- [Gaussian process regression](https://dirmeier.github.io/etudes/gaussian_process_regression.html) introduces non-parametric Bayesian regression.
- [Gaussian process classification](https://dirmeier.github.io/etudes/gaussian_process_classification.html) *extends* Gaussian process regression to classification scenarios.
- [Dirichlet process mixture models](https://dirmeier.github.io/etudes/dirichlet_process_mixture_models.html) extends the Bayesian mixture to the infinite case, i.e. when we don't know the number of clusters beforehand. We use the *Chinese restaurant process* and the *stick-breaking construction* for inference..
- [SBC](https://dirmeier.github.io/etudes/simulation_based_calibration.html) shows a method to validate Bayesian posterior inferences.
- [Structure MCMC](https://dirmeier.github.io/structure-learning-with-pymc/index.html) shows how `PyMC3` can be used to learn the structure of a Bayesian network.
- [Mixed models](https://dirmeier.github.io/mixed-models/index.html) shows concise reference implementations for optimization of the objective of (generalized) linear mixed models.
- [Sequential regression models](https://dirmeier.github.io/rstansequential/index.html) introduces a special class of ordinal regression models which assume a sequential response mechanism.
- [Causal structure learning using VAEs](https://dirmeier.github.io/etudes/causal_structure_learning.html) implements a novel graph variational autoencoder and compares it to *greedy equivalence search*, one of the state-of-the-art methods for causal discovery.
- [Normalizing flows](https://dirmeier.github.io/etudes/normalizing_flows.html) shows how TensorFlow Probability can be used to implement a custom normalizing flow.
- [Bayesian optimization](https://dirmeier.github.io/etudes/bayesian_optimization.html) introduces the basics of optimization of costly to evaluate functions with probabilistic surrogate models.
- [Hierarchical, coregionalized GPs](https://dirmeier.github.io/etudes/gp_coregionalization.html) implements two GP models and compares their predictive performance as well as MCMC diagnostics on an US election data set.
- [Variational LSTMs](https://dirmeier.github.io/etudes/variational_lstms.html) implements a variational multivariate LSTM for timeseries prediction of an US election data set.
- [Hilbert-space approximate copula processes](https://dirmeier.github.io/etudes/low_rank_copula_processes.html) explains how a copula process in conjunction with Hilbert-space approximations can be used to model stochastic volatility.
- [VI for stick-breaking constructions](https://dirmeier.github.io/etudes/stick_breaking_constructions.html) implements mean-field variational approximations for nonparametric mixture and factor models using stick-breaking constructions.
- [Tensor-product spline smoothers](https://dirmeier.github.io/etudes/causal_inference_using_tensor_product_smoothing_splines.html) implements a probabilistic model for causal inference with structured latent confounders.
- [Normalizing flows for variational inference](https://dirmeier.github.io/etudes/normalizing_flows_for_vi.html) implements an *inverse autoregressive flow* for variational inference of parameters in a simple bivariate Gaussian example in Jax, Distrax, Optax and Haiku.
- [Diffusion models I](https://dirmeier.github.io/etudes/diffusion_models.html) introduces a novel class of generative models that are inspired by non-equilibrium thermodynamics.
- [Probabilistic reconciliation](https://dirmeier.github.io/etudes/probabilistic_reconciliation.html) implements and tests two recent methods on reconciliation of hierarchical time series forecasts.
- [Diffusion models II](https://dirmeier.github.io/etudes/score_based_sdes.html) introduces a new class of generative models using stochastic differential equations and denoising score matching.

## Build

Compile the qmd files via

```
make file=FILE.qmd
```

Then move the created html and folder of images to docs.

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm.me">sfyrbnd @ pm me</a>
