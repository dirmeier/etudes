---
title: "Gaussian process regression"
author: "Simon Dirmeier <simon.dirmeier @ web.de>"
date: "October 2018"
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
                      fig.align = "center", fig.width=11, fig.height=4,
                      dpi = 360)
```

Gaussian processes (GPs) offer an elegant framework for non-parametric Bayesian regression
by endowing a prior distribution over a function space. In this notebook we derive the basics of how they can be used for regression and machine learning. Most of the material is based on @rasmussen2006, but I also recommend @betancourt2020gp as a resource. Throughout this notebook, we will use [*Stan*](https://mc-stan.org/) to fit GPs. Feedback and comments are welcome!

```{r}
suppressMessages({
  library(tidyverse)
  library(ggthemes)
  library(colorspace)

  library(rstan)
  library(bayesplot)
})

set.seed(42)
color_scheme_set("darkgray")
```

# Priors over functions

Bayesian linear regression models assume a dependency

\begin{align*}
f_{\boldsymbol \beta}& :  \ \mathcal{X} \rightarrow \mathcal{Y},\\
f_{\boldsymbol \beta}(\mathbf{x}) & =  \ \mathbf{x}^T \boldsymbol \beta + \epsilon,
\end{align*}

parametrized by a coefficient vector $\boldsymbol \beta$. In order to quantifiy uncertainty we have about the parameters, we put a prior distribution on $\boldsymbol \beta$. When we use Gaussian processes, we instead put a prior on the function $f$ itself:

\begin{align*}
f(\mathbf{x}) & \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
\end{align*}

So a Gaussian process is a distribution over functions. It is parameterized by a *mean function* $m$ that returns a vector of length $n$ and a *covariance function* $k$ that returns a matrix of dimension $n \times n$, where $n$ is the number of samples. For instance, the mean function could be a constant (which we will assume throughout the rest of this notebook), and the kernel could be an exponentiated quadratic which is defined as:

\begin{align*}
k(\mathbf{x}, \mathbf{x}') &= \alpha^2 \exp\left(- \frac{1}{2\rho^2} ||\mathbf{x} - \mathbf{x}' ||^2 \right)
\end{align*}

where $\alpha$ and $\rho$ are hyperparameters.

# Sampling from a GP prior

To sample from a GP we merely need to create points $x_i \in \mathcal{X}$ from some domain $\mathcal{X}$ and specify the hyperparameters of the covariance function.

```{r}
n.star <- 1000L
x.star <- seq(-1, 1, length.out=n.star)

alpha <- 1
rho <- .1
```

We specify the covariance function using Stan.

```{r}
prior.model <- "_models/gp_prior.stan"
cat(readLines(prior.model), sep = "\n")
```

Having all components set up we sample five realizations from the prior using Stan.

```{r, results='hide', warning=FALSE, message=FALSE, error=FALSE}
prior <- stan(
  prior.model, 
  data=list(n=n.star, x=x.star, alpha=alpha, rho=rho),
  iter=5,
  warmup=0,
  chains=1,
  algorithm="Fixed_param"
)
```

```{r}
prior <- extract(prior)$f
```

Let's plot the prior samples. Every line in the plot below represents one realization

```{r}
prior %>%
  data.frame(x=x.star, f=t(.)) %>% 
  tidyr::pivot_longer(starts_with("f"), names_to="sample", values_to = "f") %>%
  ggplot() +
  geom_line(aes(x, f, color=sample)) +
  scale_color_discrete_sequential(l1 = 1, l2 = 60) +
  theme_tufte() +
  theme(
    axis.text = element_text(colour = "black", size = 15),
    strip.text = element_text(colour = "black", size = 15)
  ) +
  xlab(NULL) +
  ylab(NULL) +
  guides(color=FALSE)
```

# Sampling from a GP posterior

To make use of GPs for regression, we model the conditional distribution of a random variable $Y$ (for which we observe $n$ data points) as

\begin{align*}
Y \mid f \sim \mathcal{N}(f, \Sigma)
\end{align*}

where $\Sigma = \sigma^2 \mathbf{I}$. Since we assume both $Y$ and every finite realization of $f$ to be Gaussian, $Y$ and $f$ are also jointly Gaussian

\begin{align*}
\left[
\begin{array}{c}
\mathbf{y} \\
{f}
\end{array}
\right] \sim
\mathcal{N} \left(\mathbf{0},
\begin{array}{cc}
k(\mathbf{x}, \mathbf{x}')+ \Sigma & k(\mathbf{x}, \mathbf{x}') \\
k(\mathbf{x}, \mathbf{x}') & k(\mathbf{x}, \mathbf{x}')
\end{array}
\right)
\end{align*}

Conditioning on $\mathbf{y}$ gives:

\begin{align*}
f \mid \mathbf{y}, \mathbf{x} & \sim \mathcal{GP}\left(\tilde{m}(\tilde{\mathbf{x}}), \tilde{k}({\mathbf{x}}, {\mathbf{x}}')\right)
\end{align*}

where the posterior mean function $\tilde{m}(\mathbf{x})$ is specified as

\begin{align*}
\tilde{m}(\mathbf{x}) & = k({\mathbf{x}}, \mathbf{x}')\left( k(\mathbf{x}, \mathbf{x}') + \Sigma \right)^{-1} \mathbf{y}
\end{align*}

and the posterior covariance function $\tilde{k}(\mathbf{x}, \mathbf{x}')$

\begin{align*}
\tilde{k}(\mathbf{x}, \mathbf{x}')  & = k({\mathbf{x}}, {\mathbf{x}}') - k({\mathbf{x}}, \mathbf{x}') \left( k(\mathbf{x}, \mathbf{x}') + \Sigma \right)^{-1} k(\mathbf{x}, \mathbf{x}')
\end{align*}

So the posterior is again a Gaussian process with modified mean and variance functions. This is straightforward to compute in `R`, but let's rather compute it in Stan again.

First, we create a set of observations $\mathbf{y}$. We can create 
such as set, for instance, by taking a sample from the prior and adding noise to it:

```{r}
sigma <- 0.1

n <- 30L
idxs <- sort(sample(seq(prior[1, ]), n, replace=FALSE))

x <- x.star[idxs]
f <- prior[1, idxs]
y <- f + rnorm(n, sigma)
```

```{r}
D <- data.frame(y=y, x=x)

ggplot(D) +
  geom_point(aes(x, y), size=1) +
  theme_tufte() +
  theme(
    axis.text = element_text(colour = "black", size = 15),
    strip.text = element_text(colour = "black", size = 15)
  ) +
  xlab(NULL) +
  ylab(NULL)
```

The model file to sample from the posterior can be found below.

```{r}
posterior.model <- "_models/gp_posterior.stan"
cat(readLines(posterior.model), sep = "\n")
```
We sample 1000 times this time to get a good estimate of the posterior quantiles (for this we of course also could compute the variance analytically, but we can also take the quantiles of the sample). More specifically, we compute the posterior mean and covariance and then sample from a multivariate normal.

```{r, results='hide', warning=FALSE, message=FALSE, error=FALSE}
posterior.model <- "_models/gp_posterior.stan"
posterior <- stan(
  posterior.model, 
  data=list(n=n, x=x, y=y, 
            n_star=n.star, x_star = x.star, 
            alpha=alpha, rho=rho, sigma=sigma),
  iter=1000,
  warmup=0,
  chain=1,
  algorithm="Fixed_param"
)
```

Having the posterior samples, we compute their mean and 90% quantiles, and plot them.

```{r}
posterior <- extract(posterior, "f_star")$f_star
posterior.mean      <- apply(posterior, 2, mean)
posterior.quantiles <- apply(posterior, 2, quantile, prob=c(0.05, 0.95))
```

```{r}
posterior.frame <- data.frame(
  x=x.star,
  m=posterior.mean, 
  q=t(posterior.quantiles)) %>%
  set_names(c("x", "mean", "lower", "upper"))

ggplot() +
  geom_point(data=D, aes(x, y), size=1) +
  geom_ribbon(data = posterior.frame, 
              aes(x=x, ymin=lower, ymax=upper), 
              fill="#A1A6C8") +
  geom_line(data = posterior.frame,  aes(x, mean), color="darkblue") +
  theme_tufte() +
  theme(
    axis.text = element_text(colour = "black", size = 15),
    strip.text = element_text(colour = "black", size = 15)
  ) +
  xlab(NULL) +
  ylab(NULL)
```

## Posterior predictive

We can use the same formalism as above to derive the posterior predictive distribution, i.e. the distribution of function values $f^*$ for new observations $\mathbf{x}^*$. This is useful, when we want to do prediction.

The predictive posterior is given like this:

\begin{align*}
p(f^* \mid \mathbf{y}, \mathbf{x}, \mathbf{x}^*) = \int p(f^* \mid f) \ p(f \mid \mathbf{y}, \mathbf{x}),
\end{align*}

(where we included $\mathbf{x}$ for clarity). However, since our original data set $\mathbf{y}$ and $f^*$ have a joint normal distribution, we can just use Gaussian conditioning again. 

# Fitting hyperparameters

Usually, the kernel hyperparameters as well as the noise variances are not given, 
so we need to estimate them from data. We can do that for instance by endowing the hyperparameters with priors, or by optimizing them using maximum marginal likelihood. In this notebook we'll do the former. For a detailed discussion of both, see for instance @betancourt2020gp.

Since the hyperparameters are not known and need to be fit, we need to update our posterior code a bit.

```{r}
posterior.model <- "_models/gp_posterior_parameters.stan"
cat(readLines(posterior.model), sep = "\n")
```
We then infer the posteriors of the hyperparameters and the noise variance as before with the latent GP.

```{r, results='hide', warning=FALSE, message=FALSE, error=FALSE}
posterior <- stan(
  posterior.model, 
  data=list(n=n, x=x, y=y),
  chains=4,
  iter=2000
)
```

Let's have a look at the summary:

```{r}
posterior
```

The inferences look good: high-effective sample sizes as well as $\hat{R}$s of one. Let's in the end plot the traces as well as the histograms of the posteriors.

```{r, message=FALSE, error=FALSE, warning=FALSE}
bayesplot::mcmc_trace(posterior, pars=c("sigma", "rho", "alpha")) +
  theme_tufte() +
  theme(
    axis.text = element_text(colour = "black", size = 15),
    strip.text = element_text(colour = "black", size = 15)
  ) +
  scale_color_discrete_sequential(l1 = 1, l2 = 60) +
  xlab(NULL) +
  ylab(NULL) +
  guides(color=FALSE)
```

```{r, message=FALSE, error=FALSE, warning=FALSE}
bayesplot::mcmc_hist(posterior, pars=c("sigma", "rho", "alpha")) +
  theme_tufte() +
  theme(
    axis.text = element_text(colour = "black", size = 15),
    strip.text = element_text(colour = "black", size = 15)
  ) +
  xlab(NULL) +
  ylab(NULL)
```

# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>

The notebook is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Session info

```{r}
sessionInfo()
```

# References
