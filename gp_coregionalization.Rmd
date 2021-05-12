---
title: "Hierarchical, coregionalized GPs"
author: "Simon Dirmeier <simon.dirmeier @ web.de>"
date: "May 2021"
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
    highlight: pygments
---

```{r knitr_init, include=FALSE, echo=FALSE, cache=FALSE, message=FALSE}
knitr::opts_chunk$set(comment = NA, warning = FALSE, error = FALSE,
                      fig.align = "center",
                      fig.width=10, fig.height=5)

library(reticulate)
use_condaenv("pax-dev")
```

In this case study we compare a multi-level Gaussian process model to a hierarchical coregionalized Gaussian process model in terms of their predictive performance and their MCMC sampling diagnostics.
The case study is inspired by Rob Trangucci's talk at StanCon 2017 [@trangucci2017], where he demonstrated a multi-level GP model to predict US presidential votes.

Usually I implement my notebooks in Stan, but since I wanted to test [Numpyro](http://num.pyro.ai/en/latest/index.html) for a long time, we will be using it here for a change. Feedback and comments are welcome!

```{python}
import warnings 
import numpy as onp
import pandas as pd

import jax
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as nd
from numpyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

import palettes

palettes.set_theme()
warnings.filterwarnings('ignore')
```

# Presidential elections 

The data set consists of counts of votes for US parties between 1976 and 2016 adopted from the [data of Rob's talk](https://github.com/stan-dev/stancon_talks/tree/master/2017/Contributed-Talks/08_trangucci/data_pres_forecast). The counts are available for every state in the US. We can either model these counts directly, or, following @trangucci2017, we model the proportion of votes one party received in a state.

```{python}
D = pd.read_csv("./data/elections.csv")
D.loc[:, "proportion"] = D.dem / (D.dem + D.rep)
D
```

The geographical location of the states suggest a grouping per region. Below we plot the proportion of votes for the democratic party for every state and region for the period between 1976 and 2016.

```{python, message=FALSE}
g = sns.FacetGrid(
    D, 
    col="region",
    hue="state",
    palette=palettes.discrete_sequential_colors(),
    col_wrap=4,
    sharex=False, 
    sharey=False
)
_ = g.map_dataframe(
    sns.lineplot, x="year", y="proportion", style="state", markers="o"
)
_ = g.set_axis_labels("Total bill", "Tip")
sns.despine(left=True)
plt.show()
```

The data suggests a model that considers a general national trend per election year and region, and a baseline offset for democratic votes. Within regions there is also a clear correlation between states that suggests either coregionalisation or a hierarchical prior.

# Preprocessing

Before we model the data, we implement some utility functions. We use a exponentiated quadratic covariance function throughout the case study, which in Jax can be implemented like this.

```{python}
def rbf(X1, X2, sigma=1.0, rho=1.0, jitter=1.0e-6):
    X1_e = np.expand_dims(X1, 1) / rho
    X2_e = np.expand_dims(X2, 0) / rho
    d = np.sum((X1_e - X2_e) ** 2, axis=2)    
    K = sigma * np.exp(-0.5 * d) + np.eye(d.shape[0]) * jitter
    return K
```

To measure the sampling time our models take, we also implement a decorator that wraps a function and times it.

```{python}
def timer(func):
    from timeit import default_timer    
    def f(*args, **kwargs):
        start = default_timer()
        res = func(*args, **kwargs)
        stop = default_timer()
        print(f"Elapsed time: {stop - start}")
        return res
    return f
```

For inference, we use Numpyro's NUTS and add our previously defined decorator to it:

```{python}
@timer
def sample(model, niter=1000):
    rng_key, rng_key_predict = random.split(random.PRNGKey(23))

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=niter,
        num_samples=niter,
        num_chains=4,
        progress_bar=False,
    )
    mcmc.run(
        rng_key,
        y,
        Xu,
        n_times,
        time_idxs,
        n_states,
        state_idxs,
        n_regions,
        region_idxs,
        train_idxs,
        n_states_per_region,
    )
    return mcmc
```

Finally, we make a prediction by sampling from the joint posterior.

```{python}
def predict(mcmc, n=5):
    samples = mcmc.get_samples()
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    vmap_args = (
        random.split(rng_key_predict, samples["nu"].shape[0]),
        samples["nu"],
        samples["eta"],
    )
    preds_map = jax.vmap(
        lambda key, nu, eta: np.mean(
            nd.Beta(eta, nu - eta).sample(key, sample_shape=(n,)), axis=0
        )
    )
    preds = preds_map(*vmap_args)
    means = np.mean(preds, axis=0)
    quantiles = np.percentile(preds, [5.0, 95.0], axis=0)
    return means, quantiles
```

We implement both models by regressing the proportions on the electoral year. For that we convert the years to numerical values first and then sort the data frame by region, state and year.


```{python}
years = pd.to_datetime(D.year)
years = (years - years.min()) / pd.Timedelta(1)

D.loc[:, ("year_numerical")] = years
D.loc[:, ("region_idxs")] = D["region"].apply(
    lambda x: list(D.region.unique()).index(x)
)
D.loc[:, ("state_idxs")] = D["state"].apply(
    lambda x: list(D.state.unique()).index(x)
)
D.loc[:, ("time_idxs")] = D["year_numerical"].apply(
    lambda x: list(D.year_numerical.unique()).index(x)
)
D = D.sort_values(["region", "state", "year_numerical"])
D
```

```{python}
X = np.array(D["year_numerical"].values).reshape(-1, 1)
Xu = np.unique(X).reshape(-1, 1)
y = np.array(D["proportion"].values)
```

In addition, we need to compute the indexes of the time points, states and regions, such that we can correctly assign everything.

```{python}
time_idxs = np.array(D["time_idxs"].values)
n_times = len(np.unique(time_idxs))
state_idxs = np.array(D["state_idxs"].values)
n_states = len(np.unique(state_idxs))
region_idxs = np.array(D["region_idxs"].values)
n_regions = len(np.unique(region_idxs))

n_states_per_region = np.array(
    D.groupby(["region", "state"]).size().groupby("region").size()
)
```

Since we also want to compare predictive performance, we treat one of 11 data points as test point and the other 10 as train indexes. That means for a data set of 550 observations (50 * 11 states), we use 50 test points.


```{python}
train_idxs = np.tile(np.arange(11) != 7, n_states)
```

# Likelihood

To model the data, we will need to choose a suitable likelihood. A fitting one having the same support as our data $Y$ is the Beta distribution:

$$
P(y \mid \alpha, \beta) = \frac{1}{B(\alpha, \beta)} y^{\alpha - 1} (1 - y)^{\beta - 1}
$$
For regression modelling, this one is a bit awkward to use, so we use its alternative parameterization following the [Stan manual](https://mc-stan.org/docs/2_26/functions-reference/beta-proportion-distribution.html):

$$
P(y \mid \mu, \kappa) = \frac{1}{B\left(\mu\kappa, (1 - \mu)\kappa\right)} y^{\mu\kappa -1} (1 - y)^{(1 - \mu)\kappa -1}
$$
Unfortunately, Numpyro does not have this parameterization, but this is not a problem as we can easily reparameterize.

# Prediction

For Gaussian likelihoods and given a posterior sample of the parameters of the covariance function, prediction boils down to Gaussian conditioning. For non-Gaussian likelihoods, we would first
infer the posterior of the latent GP $P(f \mid y, X)$ and given this compute the posterior predictive 

$$
P(y^* \mid y, X, X^*) = \int \int  P(y^* \mid f^*) P(f^* \mid f, X, X^*) P)(f \mid y, X) df df^*
$$
We can directly model this via the joint distribution of the observed (training data) and unobserved (testing data) responses, i.e.,

$$
P(y, y^* \mid X, X^*) 
$$
Specifically, we the generative model is defined on the entire set of predictors $X, X*$, but the likelihood only considers observed values $y$.

# A multi-level GP

Our first model will be an adoption of Rob's model, with the exception that we will use GPs entirely. The generative model reads:

$$\begin{aligned}
y_{rst} &\sim \text{Beta}\left(\eta_{rst}, \nu - \eta_{rst} \right) \\
\eta_{rst} & = \nu * \text{logit}^{-1}\left(\mu + f_{rt} + g_{st} + h_{st}  \right) \\
f_r & \sim GP(0, K(\sigma^f, \rho^f)) \\
g_{s} & \sim GP(0, K(\sigma^g, \rho^g )) \\
h_{s} & \sim GP(0, K(\sigma^h, \rho^h_{s})) \\
\nu  & \sim \text{Gamma}(5, 0.01) \\
\mu  & \sim \text{Normal}(0, 0.5)
\end{aligned}
$$
The other parameters (covariance variances and lengthscales) are drawn from distributions with appropriate support. Notably, we are using a sum of three GPs per state: a regional GP with common variance and lengthscale between all regions, a state-level GP with common variance and length-scale between all states, and another state-level GP with common variance between states, but individual lengthscales for every state to account for everything that is not explained by the other two. The function below implements the generative model above:

```{python}
def multilevel_model(
    y,
    Xu,
    n_times,
    time_idxs,
    n_states,
    state_idxs,
    n_regions,
    region_idxs,
    train_idxs,
    n_states_per_region,
):
    n = 3
    sigma_tot = numpyro.sample("sigma_tot", nd.Gamma(3.0, 3.0))
    sigma_prop = numpyro.sample("sigma_prop", nd.Dirichlet(np.repeat(2.0, n)))
    sigmas = n * sigma_prop * sigma_tot

    rho_region_gp = numpyro.sample("rho_region_gp", nd.LogNormal(0.0, 1.0))
    K_region_gp = rbf(Xu, Xu, sigmas[0], rho_region_gp)
    L_region_gp = np.linalg.cholesky(K_region_gp)
    with numpyro.plate("regions", size=n_regions):
        f_reg_tilde = numpyro.sample(
            "f_reg_tilde", nd.Normal(loc=np.zeros((Xu.shape[0], 1)))
        )
        f_reg = numpyro.deterministic("f_reg", L_region_gp @ f_reg_tilde)
    f_reg = np.repeat(f_reg, n_states_per_region, axis=1)
    f_reg = f_reg.T.reshape(-1)

    rho_state_gp = numpyro.sample("rho_state_gp", nd.LogNormal(0.0, 1.0))
    K_state_gp = rbf(Xu, Xu, sigmas[1], rho_state_gp)
    L_state_gp = np.linalg.cholesky(K_state_gp)
    with numpyro.plate("states", size=n_states):
        f_stat_tilde = numpyro.sample(
            "f_stat_tilde", nd.Normal(loc=np.zeros((Xu.shape[0], 1)))
        )
        f_stat = numpyro.deterministic("f_stat", L_state_gp @ f_stat_tilde)
    f_stat = f_stat.reshape(-1)

    with numpyro.plate("states", size=n_states):
        rho = numpyro.sample("rho", nd.LogNormal(0.0, 1.0))
        K = rbf(Xu, Xu, sigmas[2], rho)
        L = np.linalg.cholesky(K)
        f_tilde = numpyro.sample(
            "f_tilde", nd.Normal(loc=np.zeros((Xu.shape[0], 1)))
        )
        f = numpyro.deterministic("f", L @ f_tilde)
    f = f.reshape(-1)

    nu = numpyro.sample("nu", nd.Gamma(5.0, 0.01))
    mu = numpyro.sample("mu", nd.Normal(0.0, 0.5))
    eta = numpyro.deterministic(
        "eta", nu * jax.scipy.special.expit(mu + f_reg + f_stat + f)
    )
    numpyro.sample(
        "y", nd.Beta(eta[train_idxs], nu - eta[train_idxs]), obs=y[train_idxs]
    )
```

Having defined the model, posterior inference is fairly easy. We also compute 
MCMC diagnostics to check if the posteriors mix, no divergences occur, etc.

```{python, warning=FALSE, message=FALSE, error=FALSE}
mcmc_multilevel = sample(multilevel_model)
rhat_multilevel = az.rhat(mcmc_multilevel)
ess_multilevel = az.ess(mcmc_multilevel)
```

```{python}
rhat_multilevel.data_vars
```

```{python}
ess_multilevel.data_vars
```

Both of the diagnostics are ok. The R-hats should be ideally around one, while the effective sample sizes should be as high as possible.

Next we make sample from the posterior predictive, compute its mean and quantiles.

```{python}
means_multilevel, quantiles_multilevel = predict(mcmc_multilevel)

Dm = D.copy()
Dm.loc[:, "y_hat"] = onp.array(means_multilevel)
Dm.loc[:, "y_hat_lower"] = onp.array(quantiles_multilevel[0])
Dm.loc[:, "y_hat_upper"] = onp.array(quantiles_multilevel[1])
```

Let's overlay the mean of the posterior predictive to the actual data.

```{python, warning=FALSE, message=FALSE, error=FALSE}
g = sns.FacetGrid(
    Dm, 
    col="region",
    hue="state",
    col_wrap=4,
    palette=palettes.discrete_diverging_colors(),
    sharex=False, 
    sharey=False
)
_ = g.map_dataframe(
    sns.lineplot,
    x="year",
    y="y_hat", 
    style="state",
    marker="o",
    alpha=0.5
)
_ = g.map_dataframe(
    sns.lineplot,
    x="year",
    y="proportion", 
    style="state",
)
_ = g.set_axis_labels("Total bill", "Tip")
sns.despine(left=True)
plt.show()
```

The predictions also look good. Let's in the end compute the average absolute error
of the test indexes before we turn to fitting the second model:

```{python}
np.mean(np.abs(Dm.proportion[~train_idxs].values - Dm.y_hat[~train_idxs].values))
```

# A coregionalized GP

In the model above, we used two GPs for a state, one where the lengthscale of the covariance function is shared among states and one where it is allowed to vary for every state. We can alternatively try to explicitely correlate the state-level GPs using a coregion covariance function. The generative model has the following form:

$$\begin{aligned}
y_{rst} &\sim \text{Beta}\left(\eta_{rst}, \nu - \eta_{rst} \right) \\
\eta_{rst} & = \nu * \text{logit}^{-1}\left(\mu + f_{rt} + h_{r[s]t}  \right) \\
f_r & \sim GP \left(0, K(\sigma^f, \rho^f)\right) \\
h_{r} & \sim GP\left(0, K(\sigma^h_{r}, \rho^h_{r}), C(\omega_r)\right) \\
\nu  & \sim \text{Gamma}(5, 0.01) \\
\mu  & \sim \text{Normal}(0, 0.5)
\end{aligned}
$$

Here, the sampling statement $h_{r} \sim GP\left(0, K(\sigma^h_{r}, \rho^h_{r}), C(\omega_r)\right)$ represents a "Matrix" GP, i.e., a stochastic process of which any finite sample is a matrix normal random variable with a *fixed* number of columns. In this case the number of columns is the cardinality of $\{i : i \in r\}$. Or: supposing region $r$ consists of six states, then a sampling from $h_r$ yields always a matrix with six columns. 

But how do we sample from it? Fortunately we can reparameterize the sampling statement. We first sample a $T \times q$ matrix of standard normals, where $T$ is the number of timepoints and $q$ is the cardinality of $\{i : i \in r\}$, then left multiply it with the square of the covariance matrix $ K(1, \rho^{h}_{r})$ (its Cholesky decomposition) while setting the covariance function's variance parameter to $1.0$ (since the variance is not identified) and then right multiply the product with the root of the covariance $C(\omega)$. We can implement the model like this:

```{python}
def coregional_model(
    y,
    Xu,
    n_times,
    time_idxs,
    n_states,
    state_idxs,
    n_regions,
    region_idxs,
    train_idxs,
    n_states_per_region,
):
    rho_region_gp = numpyro.sample("rho_region_gp", nd.LogNormal(0.0, 1.0))
    sigma_region_gp = numpyro.sample("sigma_region_gp", nd.LogNormal(0.0, 1.0))
    K_region_gp = rbf(Xu, Xu, sigma_region_gp, rho_region_gp)
    L_region_gp = np.linalg.cholesky(K_region_gp)
    with numpyro.plate("regions", size=n_regions):
        f_reg_tilde = numpyro.sample(
            "f_reg_tilde", nd.Normal(loc=np.zeros((Xu.shape[0], 1)))
        )
        f_reg = numpyro.deterministic("f_reg", L_region_gp @ f_reg_tilde)
    f_reg = np.repeat(f_reg, n_states_per_region, axis=1)
    f_reg = f_reg.T.reshape(-1)

    fs = []
    for i, q in enumerate(n_states_per_region):
        rho = numpyro.sample(f"rho_{i}", nd.LogNormal(0.0, 1.0))
        K = rbf(Xu, Xu, 1.0, rho)
        L = np.linalg.cholesky(K)
        sigma = numpyro.sample(f"sigma_{i}", nd.LogNormal(np.zeros(q), 1.0))
        omega = numpyro.sample(f"omega_{str(i)}", nd.LKJCholesky(q, 2.0))
        f_tilde = numpyro.sample(
            f"f_tilde_{i}", nd.Normal(loc=np.zeros((Xu.shape[0], q)))
        )
        f = numpyro.deterministic(
            f"f_{i}", L @ f_tilde @ (np.diag(np.sqrt(sigma)) @ omega).T
        )
        fs.append(f.reshape(-1))
    f = np.concatenate(fs)

    nu = numpyro.sample("nu", nd.Gamma(5.0, 0.01))
    mu = numpyro.sample("mu", nd.Normal(0.0, 0.5))
    eta = numpyro.deterministic(
        "eta", nu * jax.scipy.special.expit(mu + f_reg + f)
    )
    numpyro.sample(
        "y", nd.Beta(eta[train_idxs], nu - eta[train_idxs]), obs=y[train_idxs]
    )
```

As above, after sampling we compute MCMC diagnostics.

```{python, warning=FALSE, message=FALSE, error=FALSE}
mcmc_coregional = sample(coregional_model)
rhat_coregional = az.rhat(mcmc_coregional)
ess_coregional = az.ess(mcmc_coregional)
```


```{python}
rhat_coregional.data_vars
```

```{python}
ess_coregional.data_vars
```

The diagnostics look fine for this model as well: R-hats are roughly one, no divergences, and sufficiently high effective sample sizes. Next we compute the means and quantiles of the posterior predictive:

```{python}
means_coregional, quantiles_coregional = predict(mcmc_coregional)

Dc = D.copy()
Dc.loc[:, "y_hat"] = onp.array(means_coregional)
Dc.loc[:, "y_hat_lower"] = onp.array(quantiles_coregional[0])
Dc.loc[:, "y_hat_upper"] = onp.array(quantiles_coregional[1])
```

We also visualize the predictions again, overlayed over the actual data.

```{python, warning=FALSE, message=FALSE, error=FALSE}
g = sns.FacetGrid(
    Dc, 
    col="region",
    hue="state",
    col_wrap=4,
    palette=palettes.discrete_diverging_colors(),
    sharex=False, 
    sharey=False
)
_ = g.map_dataframe(
    sns.lineplot,
    x="year",
    y="y_hat", 
    style="state",
    marker="o",
    alpha=0.5
)
_ = g.map_dataframe(
    sns.lineplot,
    x="year",
    y="proportion", 
    style="state",
)
_ = g.set_axis_labels("Total bill", "Tip")
sns.despine(left=True)
plt.show()
```

As before, the predictions look great, and are hardly different to the other model. Hence, lets'
have a look at the average absolute error of the test data:

```{python}
np.mean(np.abs(Dc.proportion[~train_idxs].values - Dc.y_hat[~train_idxs].values))
```

The prediction is a bit better, but the runtime is significantly worse (probably due to the increased dimensionality of the variance paramaters). Hence Rob's model seems to be the clear winner if we factor in both aspects.

# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>

The notebook is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# References
