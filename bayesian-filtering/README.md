<h1 align="center"> bayesian-filtering </h1>

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

Bayesian filtering test implementations in Python.

## Introduction

Two implementations of Bayesian tracking algorithms as part of an exercise for *Recursive estimation*. The implemented models include an *extended Kalman filter* as well as a *particle filter*. Bayesian filters recursively apply a *prediction* and a *filtering* step for inference of latent states from noisy observations, such as the input from a sensor of a robot.

## Installation

Install the dependencies using:

```bash
  pip install -r requirements.txt
```

That should let you run the notebook.

## Bayesian filter

Example for state estimation using a particle filter.

<div align="center">
<p>Recursive estimation of states</p>
<img src="https://github.com/dirmeier/bayesian-filtering/blob/master/_fig/particle_filter.png" alt="Drawing" width="50%" />
</div>


## Author

* Simon Dirmeier <a href="simon.dirmeier@gmx.de">simon.dirmeier@gmx.de</a>


