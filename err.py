from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from pymc3.distributions.dist_math import bound
from theano import tensor as tt
import pandas as pd


class Error(pm.Discrete):
    def __init__(self, theta, alpha, beta, *args, **kwargs):
        super(Error, self).__init__(*args, **kwargs)
        self.theta = theta
        self.alpha = alpha
        self.beta = beta

    def random(self, point=None, size=None):
        raise NotImplementedError()

    def logp(self, value):
        theta = self.theta
        alpha = self.alpha
        beta = self.beta
        return bound(
          tt.sum(tt.log(value)),
          value >= 0, value <= 1,
          alpha >= 0, alpha <= 1,
          beta >= 0, beta <= 1,
          theta >= 0, theta <= 1)


def stick_breaking(beta):
    p = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * p


K = 30
data = sp.stats.bernoulli.rvs(.5, size=1000).reshape((100, 10))

with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    comps = [
        Error.dist(i, i, i)
        for i in range(K)
    ]
    like = pm.Mixture('like', w=w, comp_dists=comps, observed=data)

    pm.sample(100)
