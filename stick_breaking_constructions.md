---
title: "Stick-breaking constructions for nonparametric Bayesian models"
author: "Simon Dirmeier"
date: "August 2021"
bibliography: ./references/references.bib
link-citations: true
output:
  html_document:
    theme: lumen
    css: ./css/custom.css
    toc: yes
    toc_depth: 1
    toc_float:
      collapsed: no
      smooth_scroll: yes
    number_sections: no
---


```{r knitr_init, include=FALSE, echo=FALSE, cache=FALSE, message=FALSE}
knitr::opts_chunk$set(comment = NA, warning = FALSE, error = FALSE,
                      fig.align = "center",
                      fig.width=10, fig.height=5)

library(reticulate)
use_condaenv("etudes-dev")
```

In this notebook, we will explore stick-breaking constructions for non-parameteric mixture and factor models and fit them via variational inference.
Both of these model are traditionally fit via, for instance, slice sampling or Gibbs sampling, but recent developments in probabilistic programming languages are allowing us to fit them easily via automated variational inference. While nonparametric mixture models using the Dirichlet process (DP) as prior are found frequently in the literature, factor models using the Indian Buffet process (IBP) have received less attention.

I have long been enthusiastic about nonparametric Bayesian models but, except for GPs, have found them hard to work with in practice (at least for principled statistical data analysis).
Especially Hamiltonian Monte Carlo samplers, as implemented in Stan or NumPyro, where the simulated trajectories often divergence even for "easy" data sets, seem to be not very suited for this class of models, so I am curious of the results of this study.

We implement the models and variational surrogates using [Numpyro](http://num.pyro.ai/en/latest/index.html). Feedback and comments are welcome!

```{python, include=FALSE, echo=FALSE, cache=FALSE, message=FALSE}
import logging
logging.basicConfig(level=logging.ERROR, stream=sys.stdout)
```

We load some libraries for inference and working with data first.

```{python}
import pandas as pd

import jax
import jax.numpy as np
import jax.scipy as sp
import jax.random as random

import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.distributions.transforms import OrderedTransform
from numpyro.infer import SVI, Trace_ELBO
import numpyro.optim as optim

import tensorflow_probability.substrates.jax.distributions as tfp_jax

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import palettes

palettes.set_theme()
numpyro.set_host_device_count(4)
```

Check if JAX recognizes the four set cores.

```{python}
jax.local_device_count()
```

```{python, include=FALSE}
def plot_losses(res):
    fig, _ = plt.subplots(1, 1)
    ax = sns.lineplot(
      data=pd.DataFrame({"y": res.losses, "x": range(len(res.losses))}),
      y="y", x="x",
      color='black'
    );
    ax.set(xlabel="", ylabel="NLL");
    plt.show()
    
    
def plot_means(res):
    var_posterior_mean = dist.TransformedDistribution(
        dist.Normal(loc=res.params["q_mu_mu"], scale=res.params["q_mu_sd"]), 
        OrderedTransform()
    )
    var_posterior_mean_samples = var_posterior_mean.sample(
        random.PRNGKey(0), sample_shape=(1000,)
    )
    K = var_posterior_mean_samples.shape[1]
    df = pd.DataFrame(var_posterior_mean_samples, columns=[f"mu{i}" for i in range(K)])
    df = df.melt(var_name="Mu", value_name="Value")

    plt.figure(figsize=(10, 4))
    g = sns.FacetGrid(
        df,     
        col="Mu",    
        col_wrap=5,
        sharex=False, 
        sharey=False
    )
    _ = g.map_dataframe(
        sns.histplot, x="Value", color="darkgrey"
    )
    plt.show() 
```

# Infinite mixture models

Nonparametric Bayesian mixture models implement a observation model that consists of infinitely many component distributions. Using the stick-breaking construction of a Dirichlet process (@ghosal2017fundamentals, @blei2006variational), the generative model we assume here has the following form

$$\begin{align}
\beta & \sim \text{Gamma}(1.0, 1.0) \\
\nu_k & \sim\text{Beta}(1.0, \beta) \\
\pi_k & = \nu_k \prod_{j=1}^{k-1} (1 - \nu_j) \\
\mu_k & \sim \text{Normal}(0.0, 1.0)\\
\sigma_k & \sim \text{Normal}^+(1.0)  \\
y_i & \sim \sum_k^{\infty} \pi_k  \text{Normal}(\mu_k, \sigma_k)
\end{align}$$

where $k$ indexes a component distribution and $i$ indexes a data point. 

## Data

We begin by simulating a data set consisting of three components and 1000 samples.
The simulated data set should be fairly easy to fit, real world data is usually 
significantly more noisy. 

```{python}
n_samples = 1000
K = 3

means = np.linspace(-2.0, 2.0, K)
standard_deviations = np.array([0.25, 0.2, 0.3])

Z = random.randint(
    key=random.PRNGKey(23),
    minval=0,
    maxval=K, 
    shape=(n_samples,)
)

eps = dist.Normal(0.0, 1.0).sample(
    random.PRNGKey(23),
    sample_shape=(n_samples,)
)
y = means[Z] + eps * standard_deviations[Z]
```

The three components are centered around $-2$, $0$ and $2$ with a low standard deviation.

```{python, results='hide'}
df = pd.DataFrame(np.vstack([y, Z]).T, columns=["y", "z"])
_ = plt.figure(figsize=(15, 5))
_ = sns.histplot(
    x="y",
    hue="z",
    data=df,
    palette=palettes.discrete_sequential_colors(),
    legend=False,
    bins=50,
)
plt.show()
```

## Model

We will truncate the stick at a sufficiently large $K$ (the error of this truncation is, as I believe to recall from a reference I cannot find anymore, negligible).

```{python}
K_stick = 10
```

Next we define a routine to compute the mixing weights $\pi$ from $\nu$.

```{python}
def sample_stick(nu):
    ones = np.ones((*nu.shape[:-1], 1))
    rem = np.concatenate(
      [ones, np.cumprod(1 - nu, axis=-1)[:-1]],
      axis=-1
    )
    mix_probs = nu * rem
    return mix_probs
```

We will infer the posterior distributions over the latent variables using the marginal mixture representation above. We define the prior model first. In comparison to the generative model defined above, we will order the mean variables.

```{python}
def prior():
    beta = numpyro.sample("beta", dist.Gamma(1.0, 1.0))
    nu = numpyro.sample(
        "nu",
        dist.Beta(
          concentration1=np.ones(K_stick), 
          concentration0=beta
        )
    )
    pi = numpyro.deterministic("pi", sample_stick(nu))
    mu = numpyro.sample(
        "mu",
        dist.TransformedDistribution(
            dist.Normal(loc=np.zeros(K_stick)), 
            OrderedTransform()
        ),
    )
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=np.ones(K_stick)))

    return pi, mu, sigma
```

We then define the log-likelihood function:

```{python}
def log_likelihood(y, pi, mu, sigma):
    lpdf_weights = np.log(pi)
    lpdf_components = dist.Normal(loc=mu, scale=sigma).log_prob(y[:, np.newaxis])

    lpdf = lpdf_weights + lpdf_components
    lpdf = sp.special.logsumexp(lpdf, axis=-1)
    return np.sum(lpdf)
```

To test the implementation, we can make a draw from the prior and plug it into the likelihood.

```{python}
with numpyro.handlers.seed(rng_seed=23):
    pi, mu, sigma = prior()
    
log_likelihood(y, pi, mu, sigma)
```

The NumPyro model itself is then only a two-liner. We include the likelihood term using a `factor` in the model specification.

```{python}
def model():
    pi, mu, sigma = prior()
    numpyro.factor("log_likelihood", log_likelihood(y, pi, mu, sigma))
```

We approximate the posterior distributions using mean field variational inference which requires us to define surrogate distributions for each of the latent variables. Specifically, we will use the following variational surrogates, adopting from @blei2006variational

$$\begin{align}
q_{\lambda}(\beta) & = \text{Gamma}(\lambda_{\beta_0}, \lambda_{\beta_1}) \\
q_{\lambda}(\nu_k) & = \text{Beta}(\lambda_{\nu_{k0}}, \lambda_{\nu_{k1}}) \\
q_{\lambda}(\mu_k) & = \text{Normal}(\lambda_{\mu_{k0}}, \lambda_{\mu_{k1}})  \\
q_{\lambda}(\sigma_k) & = \text{Normal}^+(\lambda_{\sigma_{k}})\\
\end{align}$$

where we constraint the scale parameters to be positive and the vector $\mu$ to be ordered. 

```{python}
def guide():
    q_beta_concentration = numpyro.param(
        "beta_concentration", init_value=1.0, constraint=constraints.positive
    )
    q_beta_rate = numpyro.param(
        "beta_rate", init_value=1.0, constraint=constraints.positive
    )
    q_beta = numpyro.sample(
        "beta", dist.Gamma(q_beta_concentration, q_beta_rate)
    )

    q_nu_concentration1 = numpyro.param(
        "nu_concentration1",
        init_value=np.ones(K_stick),
        constraint=constraints.positive,
    )
    q_nu_concentration0 = numpyro.param(
        "nu_concentration0",
        init_value=np.ones(K_stick) * 2.0,
        constraint=constraints.positive,
    )
    q_nu = numpyro.sample(
        "nu",
        dist.Beta(
          concentration1=q_nu_concentration1, 
          concentration0=q_nu_concentration0
        )
    )

    q_mu_mu = numpyro.param(
        "q_mu_mu", 
        init_value=np.linspace(-2.0, 0.0, K_stick)
    )
    q_mu_sd = numpyro.param(
        "q_mu_sd", 
        init_value=np.ones(K_stick), 
        constraint=constraints.positive
    )
    q_mu = numpyro.sample(
        "mu",
        dist.TransformedDistribution(
            dist.Normal(loc=q_mu_mu, scale=q_mu_sd), 
            OrderedTransform()
        ),
    )

    q_sigma_scale = numpyro.param(
        "q_sigma_scale",
        init_value=np.ones(K_stick),
        constrain=constraints.positive,
    )
    q_sigma = numpyro.sample(
      "sigma", 
      dist.HalfNormal(scale=q_sigma_scale)
    )
```

## Inference

We optimize the variational parameters $\lambda$ using NumPyro's stochastic variational inference [@hoffman2013stochastic]:

```{python, warning=FALSE, message=FALSE, error=FALSE}
num_steps = 20000

adam = optim.Adam(0.01)
svi = SVI(model, guide, adam, loss=Trace_ELBO(20))
res = svi.run(random.PRNGKey(1), num_steps=num_steps, progress_bar=False)
```

Let's have a look at the posterior mixing weights. Ideally most of the density is on the first three weights:

```{python}
nu = dist.Beta(
    concentration1=res.params["nu_concentration1"],
    concentration0=res.params["nu_concentration0"]
)

sample_stick(nu.mean)
```

Let's also visualize the posterior means:

```{python}
plot_means(res)
```

# Infinite latent feature models

In statistical analysis, we a frequently interested in decomposing a high-dimensional data set into a small number of components. Latent feature models decompose a data set $Y$ into a binary matrix $Z$ and a matrix of loadings $\Psi$ 

$$\begin{equation}
Y = Z \Psi + \epsilon
\end{equation}$$

where $\epsilon$ is a Gaussian noise matrix. Nonparametric factor models implement a observation model that consists of infinitely many features. Using the stick-breaking construction of a Indian buffet process (@ghosal2017fundamentals, @doshi2009variational, @teh2007stick, @paisley2009nonparametric), we will explore fitting the following generative model using variational inference

$$\begin{align}
\beta & \sim \text{Gamma}(1.0, 1.0) \\
\nu_k & \sim\text{Beta}(1.0, \beta) \\
\pi_k & = \prod_{j=1}^{k} \nu_j \\
z_{ik} & \sim \text{Bernoulli}(\pi_k)\\
\Psi & \sim \text{MatrixNormal}(0, I, I) \\
\sigma_k & \sim \text{Normal}^+(1.0)  \\
y_i & \sim \text{MvNormal}(z_i^T \Psi, \sigma_k I)
\end{align}$$

where $k$ indexes a component distribution and $i$ indexes a data point. 

## Data

As above, we simulate some artificial data for inference. We'll simulate data with $Q=10$ dimensions and a latent dimensionality of $K=5$.

```{python}
n_samples = 100
K = 5
Q = 10
```

Next we simulate the probabilities that a latent feature is active and the binary feature matrix itself.

```{python}
nu = dist.Beta(5.0, 1.0).sample(random.PRNGKey(0), sample_shape=(K,))
pi = np.cumprod(nu)

Z = dist.Bernoulli(probs=pi).sample(
  random.PRNGKey(1),
  sample_shape=(n_samples,)
)
```

The binary feature matrix is shown below.

```{python, message=FALSE, results='hide'}
_ = plt.figure(figsize=(15, 5))
ax = sns.heatmap(
    Z.T,
    linewidths=0.1,
    cbar=False,
    cmap=["white", "black"],
    linecolor="darkgrey",
)
_ = ax.set_ylabel("Active features")
_ = ax.set_xlabel("Samples")
_ = ax.minorticks_off()
plt.show()
```

We finally simulate the actual data and visualize it.

```{python}
psi = dist.Normal(0.0, 3.0).sample(
  random.PRNGKey(0), 
  sample_shape=(K, Q)
)
eps = dist.Normal(0.0, 0.1).sample(
  random.PRNGKey(0), 
  sample_shape=(n_samples, Q)
)
y = Z @ psi + eps
```

```{python}
df = pd.DataFrame(y, columns=[f"y{i}" for i in range(Q)])
df = df.melt(var_name="y", value_name="Value")

_ = plt.figure(figsize=(10, 4))
g = sns.FacetGrid(
  df, 
  col="y", 
  col_wrap=5, 
  sharex=False, 
  sharey=False
)
_ = g.map_dataframe(
  sns.histplot, 
  x="Value",
  color="darkgrey"
)
plt.show()
```

## Model

Since our model involves binary latent parameters, we will use a continuous relaxation using the concrete distribution (@maddison2016concrete, @jang2016categorical) which we, since this distribution is still
missing in NumPyro, can use via a TensorFlow Probability wrapper. The prior model is shown below:

```{python}
temperature = 0.000001
rec_temperature = np.reciprocal(temperature)

def prior():
    nu = numpyro.sample("nu", dist.Beta(np.ones(K), 1.0))
    pi = numpyro.deterministic("pi", np.cumprod(nu))

    Z = numpyro.sample(
        "Z",
        tfp_jax.RelaxedBernoulli(temperature, probs=pi),
        sample_shape=(n_samples,),
    )

    psi = numpyro.sample("psi", dist.Normal(np.zeros((K, Q)), 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(np.ones(Q)))

    return nu, pi, Z, psi, sigma
```

We implement the log likelihood again as a separate function to be able to easily check the code:

```{python}
def log_likelihood(y, pi, Z, psi, sigma):
    mean = Z @ psi
    lpdf = dist.Independent(
        dist.Normal(loc=mean, scale=sigma), reinterpreted_batch_ndims=1
    ).log_prob(y)
    
    return np.sum(lpdf)
```

The actual NumPyro model consists of the prior model and a factor that increments the log-density via a `factor`:

```{python}
def model():
    _, pi, Z, psi, sigma = prior()
    numpyro.factor("log_likelihood", log_likelihood(y, pi, Z, psi, sigma))
```

We approximate the posterior distributions again using mean field variational inference again which requires us to place surrogates over each latent variable. We use the following guides, adopting from @doshi2009variational

$$\begin{align}
q_{\lambda}(\beta) & = \text{Gamma}(\lambda_{\beta_0}, \lambda_{\beta_1}) \\
q_{\lambda}(\nu_k) & = \text{Beta}(\lambda_{\nu_{k0}}, \lambda_{\nu_{k1}})\\
q_{\lambda}(z_{ik}) & = \text{RelaxedBernoulli}(\lambda_{z_{ik}})\\
q_{\lambda}(\Psi) & = \text{MatrixNormal}\left(\lambda_{\Psi_{0}}, I, \text{diag}(\lambda_{\Psi_{1}})\right) \\
q_{\lambda}(\sigma_k) & = \text{Normal}^+(\lambda_{\sigma_{k}})\\
\end{align}$$

where we constraint the scale parameters to be positive.

```{python}
def guide():
    q_nu_concentration1 = numpyro.param(
        "nu_concentration1",
        init_value=np.ones(K),
        constraint=constraints.positive,
    )
    q_nu_concentration0 = numpyro.param(
        "nu_concentration0",
        init_value=np.ones(K) * 2.0,
        constraint=constraints.positive,
    )
    q_nu = numpyro.sample(
        "nu",
        dist.Beta(
            concentration1=q_nu_concentration1,
            concentration0=q_nu_concentration0,
        ),
    )

    z_logits = numpyro.param(
        "z_logits",
        init_value=np.tile(np.linspace(3.0, -3.0, K), (n_samples, 1))
    )
    Z = numpyro.sample(
        "Z",
        dist.TransformedDistribution(
            dist.Logistic(z_logits * rec_temperature, rec_temperature),
            dist.transforms.SigmoidTransform(),
        ),
    )

    q_psi_mu = numpyro.param("psi_mu", init_value=np.zeros((K, Q)))
    q_psi_sd = numpyro.param(
        "psi_sd",
        init_value=np.ones((K, Q)),
        constraint=constraints.positive,
    )
    psi = numpyro.sample(
      "psi", 
      dist.Normal(q_psi_mu, q_psi_sd)
    )

    q_sigma_scale = numpyro.param(
        "sigma_scale",
        init_value=np.ones(Q),
        constrain=constraints.positive
    )
    q_sigma = numpyro.sample(
      "sigma",
      dist.HalfNormal(scale=q_sigma_scale)
    )
```

## Inference

We use NumPyro again for fitting the variational parameters.

```{python, warning=FALSE, message=FALSE, error=FALSE}
adam = optim.Adam(0.01)
svi = SVI(model, guide, adam, loss=Trace_ELBO(20))
res = svi.run(random.PRNGKey(1), num_steps=num_steps, progress_bar=False)
```

Let's have a look at the losses

```{python}
res.losses
```

Next we check out the means of the variational posterior of $\Psi$

```{python}
res.params["psi_mu"]
```

In comparison, these are the real features:

```{python}
psi
```


# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>

The notebook is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Session info

```{python, include=FALSE}
import session_info
session_info.show()
```

# References