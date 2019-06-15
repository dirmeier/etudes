import numpy as np
import pymc3 as pm
import scipy as sp
import scipy.stats as st
import theano.tensor as tt
import seaborn

seaborn.set_style("white")


with pm.Model() as model:
    mu = pm.Uniform('alpha', 1., 100)
    like = pm.Poisson('x', mu=mu, observed=st.poisson.rvs(mu=5, size=1000))
    pm.sample()

#
# def stick_breaking(beta):
#     p = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
#     return beta * p
#
#
# K = 5
# np.random.seed(23)
# data = sp.stats.bernoulli.rvs(.5, size=250).reshape((50, 5))
#
# with pm.Model() as model:
#     alpha = pm.Gamma('alpha', 1., 1.)
#     beta = pm.Beta('beta', 1., alpha, shape=K)
#     w = pm.Deterministic('w', stick_breaking(beta))
#
#     error_alpha = pm.Uniform("ea", lower=0, upper=1, shape=K)
#     error_beta = pm.Uniform("eb", lower=0, upper=1, shape=K)
#     error_theta = pm.Uniform("et", lower=0, upper=1, shape=K)
#
#     comp = pm.DensityDist(
#       "like", logp=logp(error_theta, error_alpha, error_beta), shape=K)
#     comp_dists = comp.distribution
#     comp_dists.mean = comp_dists.mode = np.array([1, 1, 1, 1, 1])
#
#     like = pm.Mixture('x', w=w, comp_dists=comp_dists, observed=data)
#
# with model:
#     trace = pm.f(300)
