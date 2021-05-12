# Études

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dirmeier/etudes/master)

:notes: A collection of études on probabilistic models.

## About

The repository hosts some notebooks on probabilistic models, such as Gaussian processes, graphical models, normalizing flows, and so on. The notebooks are mostly on topics I am interested in or papers I happen to come across.

**I do not take warranty for the correctness of these.**

- [Gaussian process regression](https://dirmeier.github.io/etudes/gaussian_process_regression.html) introduces non-parametric Bayesian regression.
- [Gaussian process classification](https://dirmeier.github.io/etudes/gaussian_process_classification.html) *extends* Gaussian process regression to classification scenarios.
- [Bayesian optimization](https://dirmeier.github.io/etudes/bayesian_optimization.html) introduces the basics of optimization of costly to evaluate functions with probabilistic surrogate models.
- [Dirichlet process mixture models](https://dirmeier.github.io/etudes/dirichlet_process_mixture_models.html) extends the Bayesian mixture to the infinite case, i.e. when we don't know the number of clusters beforehand. We use the *Chinese restaurant process* and the *stick-breaking construction* for inference.
- [Indian Buffets](https://dirmeier.github.io/etudes/indian_buffets.html) shows the generative process of binary matrices with possibly infinitely many numbers of columns.
- [Bayesian GPLVMs](https://dirmeier.github.io/etudes/gplvm.html) implements a nonparametric Bayesian approach for dimension reduction using GPs.
- [Causal structure learning using VAEs](https://dirmeier.github.io/etudes/causal_structure_learning.html) implements a novel graph variational autoencoder and compares it to *greedy equivalence search*, one of the state-of-the-art methods for causal discovery.
- [Gaussian Graphical Models](https://dirmeier.github.io/etudes/gaussian_graphical_models.html) implements the *graphical LASSO* for sparse precision matrix estimation.
- [Normalizing flows](https://dirmeier.github.io/etudes/normalizing_flows.html) shows how TensorFlow Probability can be used to implement a custom normalizing flow.
- [SBC](https://dirmeier.github.io/etudes/simulation_based_calibration.html) shows a method to validate Bayesian posterior inferences.
- [Sequential regression models](https://dirmeier.github.io/rstansequential/index.html) introduces a special class of ordinal regression models which assume a sequential response mechanism.
- [Structure MCMC](https://dirmeier.github.io/structure-learning-with-pymc/index.html) shows how `PyMC3` can be used to learn the structure of a Bayesian network.
- [Mixed models](https://dirmeier.github.io/mixed-models/index.html) shows concise reference implementations for optimization of the objective of (generalized) linear mixed models.
- [Hierarchical, coregionalized GPs](https://dirmeier.github.io/etudes/gp_coregionalization.html) implements two GP models and compares their predictive performance as well as MCMC diagnostics on an US election data set.

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier @ web.de</a>
