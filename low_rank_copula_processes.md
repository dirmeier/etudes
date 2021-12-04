---
title: "Hilbert-space approximate copula processes"
author: "Simon Dirmeier <simon.dirmeier @ web.de>"
date: "July 2021"
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
                      fig.align = "center", fig.width=11, fig.height=4)
```

While reading some papers on copulas, I stumbled over the paper by @wilson2010copula, where they introduced *copula processes*. The paper introduces how copulas can be used to encode correlation between marginally non-Gaussian random variables using a Gaussian process. Conceptually, this seemed to me to be the same as a pushforward from a GP measure to another one. After reading the paper, I was a bit puzzled what the advantage of a copula process would be in comparison to modelling the random variates conditional on a latent Gaussian process, which ultimately lead to the creation of this notebook.

Hence in this notebook, we try to reproduce the *Gaussian Copula Process Volatility* model @wilson2010copula introduced in their paper and compare it to a model that uses a latent GP for parameterization directly. Since working with latent correlated variables can easily frustrate our analysis, we use tbe Hilbert-space approximation to the GP which was recently introduced in @solin2020hilbert and @riutort2020practical .

We begin by shortly reviewing copulas and then implement the two models in the probabilistic programming language [*Stan*](https://mc-stan.org/). Feedback and comments are welcome!

```{r }
suppressMessages({
  library(tidyverse)
  library(ggthemes)
  library(colorspace)
  
  library(cmdstanr)
  library(posterior)
  library(bayesplot)
})

set.seed(42)
color_scheme_set("darkgray")
```
# Copula processes

Following the introduction by @wilson2010copula, consider a multivariate random variable
$Y = (Y_1, \dots, Y_Q)$ with cumulative distribution function (cdf) $F(y_1, \dots y_Q)$ and marginal cdfs $F_1, \dots, F_Q$. According to Sklar's theorem there exists an Q-copula $C: [0, 1]^Q \rightarrow [0, 1]$ such that

$$\begin{equation}
F(y_1, \dots y_Q) = C(F_1(y_1), \dots, F_Q(y_Q))
\end{equation}$$

Hence, with continuous marginals $C$ is given via $Q$ probability integral transforms $u_i = F_i(y_i)$, i.e.,

$$\begin{equation}
C(u_1, \dots, u_Q) = P(U_1 \le u_1, \dots U_Q \le u_Q)
\end{equation}$$

where $U_i \sim \text{Uniform}(0, 1) \, \forall i = 1 \dots Q$. For instance, we can construct a Gaussian copula from a set of Gaussian random variables $Y$ via

$$\begin{align}
C(u_1, \dots, u_Q) &= \phi(\Phi^{-1} \circ u_1, \dots, \Phi^{-1} \circ u_Q) \\
&= \phi(\Phi^{-1} \circ F_1(y_1), \dots, \Phi^{-1} \circ F_Q(y_Q))
\end{align}$$

where $\phi$ is the cdf of a multivariate Gaussian (with appropriate mean and covariance) and $\Phi^{-1}$ is the quantile function of a (univariate) standard normal. 

The generalization from a Gaussian copula to a Gaussian copula process is then straightforward by assuming a Gaussian process measure instead of a Gaussian measure. Specifically, if there exists a mapping $\Psi$, such that

$$\begin{align}
\Psi(Y) \sim GP(\cdot, K)
\end{align}$$

is a Gaussian process with some covariance function $K$, then we call

$$\begin{align}
Y \sim GCP(\Psi, \cdot, K)
\end{align}$$

a Gaussian copula process (for instance, $\Psi = \Phi^{-1} \circ F$ for some cdf $F$). For more details on copulas and copula processes please be referred to @wilson2010copula.

# Stochastic volatility

Consider the data generating process $y_t \sim \mathcal{N}(0, \sigma^2_t)$ for $t = 1, \dots, 3$ for which we want to accurately estimate the standard deviations $\sigma_t$ and their correlation structure. We first simulate the data and visualize it.

```{r}
N <- 500
times  <- seq(1, 3, length.out=N)
sigmas <- sin(times) * cos(times**2) + 1
y      <- rnorm(n=length(sigmas), 0, sigmas)
```

```{r}
data.frame(y=y, t=times, sigma=sigmas) %>% 
  tidyr::pivot_longer(cols = c(y, sigma)) %>%
  dplyr::mutate(name = factor(name, levels = c("y", "sigma"))) %>%
  ggplot() +
  geom_line(aes(t, value), color="black") +
  facet_grid(. ~ name) +
  theme_tufte() +
  theme(
    axis.text = element_text(colour = "black", size = 15),
    strip.text = element_text(colour = "black", size = 15)
  ) +
  xlab(NULL) +
  ylab(NULL)
```

# Fitting a copula process

Following @wilson2010copula, we model the standard deviations as Gaussian copula process. We use the following generative model for the data set:

$$\begin{align}
f & \sim GP(0, K)\\
\sigma & \sim g(f, \omega) \\
y_t & \sim \mathcal{N}(0, \sigma_t)
\end{align}$$

where $g$ is a warping function with parameters $\omega = (\beta, \gamma)$:

$$\begin{align}
g(f, \omega) = \sum_i^M \beta_i^ 2  \log \left(\exp(f + \gamma_i) + 1.0 \right)
\end{align}$$

and $K = k(t, t')$ is a covariance function defined on the time points. Note that we changed the definition of the warping function a bit from the definition in the paper to avoid potential weak identifiability issues when we don't have much data. The standard deviations above are indeed GCP distributed, since (following the paper):

$$\begin{align}
P(\sigma_1 \le a_1, \dots, \sigma_Q \le a_Q) &= P\left(g^{-1}(\sigma_1) \le  g^{-1}(a_1), \dots, g^{-1}(\sigma_Q) \le  g^{-1}(a_Q)\right) \\
&= \phi \left(  g^{-1}(a_1), \dots, g^{-1}(a_Q) \right) \\
&= \phi \left( \Phi^{-1} \circ F(a_1), \dots, \Phi^{-1} \circ F(a_Q) \right)  \\
&= C(u_1, \dots, u_Q)
\end{align}$$

which we recogize as the copula defined above.

## Low-rank approximation

Fitting this model in Stan can be frustratingly slow and inefficient due to the difficulty in sampling a correlated latent variable and the high memory-footprint. For that reason we approximate the latent variable $f$ using the low-rank representation recently introduced by @solin2020hilbert and @riutort2020practical. For the model above, we will use an exponentiated quadratic covariance function

$$\begin{align}
k(x, x') = \alpha^2 \exp \left(-0.5 \frac{(x - x')^2}{\rho^2} \right)
\end{align}$$

@solin2020hilbert observed that this kernel (or rather any stationary kernel I believe) can be approximated as

$$\begin{align}
k(x, x') \approx \sum_j^J S\left(\sqrt{\lambda_j} \right) \xi_j(x) \xi_j(x')
\end{align}$$

where $S(\cdot)$ is the spectral density of the covariance function, $\lambda_j$ is the $j$th eigenvalue and $\xi_j$ is the eigenfunction of the Laplace operator. For the exponentiated quadratic covariance the spectral density is

$$
S(x) = \alpha^2 \sqrt{2  \pi} \rho  \exp \left(-0.5 (\rho  x)^2 \right)
$$
the $j$th eigenvalue

$$
\lambda_j = \left(\frac{j\pi}{2L}\right)^2
$$
and the $j$th eigenfunction

$$
\xi_j(x) = \frac{1}{\sqrt{L}}  \sin\left(\sqrt{\lambda_j}(x + L)\right)
$$
@riutort2020practical use this result to approximate the latent variable $f$ as

$$\begin{align}
f &\approx \sum_j^J \sqrt{ S\left(\sqrt{\lambda_j}  \right) } \xi_j(x) \beta_j \\
  & = \Xi(x) \left( \sqrt{ S\left(\sqrt{\lambda}  \right) }  \odot \beta  \right)
\end{align}$$

where $\beta$ is a vector of standard normals, $\Xi$ is the matrix of eigenfunctions $\xi_j$ and $\sqrt{  S(\sqrt{\lambda}) }$ is a vector of all $J$ spectral densities.

## Fitting the model

We implement the model using Stan and fit the posterior using its dynamic Hamiltonian Monte Carlo sampler. The model code of the copula process using the Hilbert space approximation by @solin2020hilbert is shown below.

```{r}
cp.stan.file <- "./_models/cp_volatility_model.stan"
cat(readLines(cp.stan.file), sep="\n")
```

Fitting this model using [cmdstanr](https://mc-stan.org/cmdstanr/index.html) is done like this:

```{r, results='hide', warning=FALSE, message=FALSE, error=FALSE}
m <- cmdstanr::cmdstan_model(cp.stan.file)

fit <- m$sample(
  data=list(N = length(y),
            times = times,
            y = y,
            Q = 50,
            L = 5 / 2 * max(times),
            K = 1),
  seed=123,
  chains=4,
  parallel_chains=4
)
```

Fitting the model was thanks to the approximation fairly fast. The diagnostics also show that the sampler didn't have any problems and the chains coverged.

```{r}
fit$cmdstan_diagnose()
```

To assess the quality of the fit, let's compare the true standard deviations with the posterior estimates. First, we extract the posterior from the fit object:

```{r}
sigmas.hat <- fit$draws("sigma")
sigmas.hat.mean <- apply(as_draws_matrix(sigmas.hat), 2, mean)
sigmas.hat.ints <- t(apply(as_draws_matrix(sigmas.hat), 2, quantile, c(0.05, 0.95)))
```

Then we define a function to plot everything:

```{r}
plot.sigmas <- function(df) { 
    ggplot(df) +
    geom_ribbon(
      aes(x=time, ymin = sigma.hat.lower, ymax = sigma.hat.upper), fill="#A1A6C8"
    ) +
    geom_line(aes(time, sigma, color="black")) +
    geom_line(aes(time, sigma.hat, color="darkblue"))  +
    scale_color_manual(values=c("black", "darkblue"),
                       labels=c("Sigma", expression(paste(widehat("Sigma"))))) +
    theme_tufte() +
    theme(
      axis.text = element_text(colour = "black", size = 15),
      strip.text = element_text(colour = "black", size = 15),
      legend.text = element_text(colour = "black", size = 15),
      legend.title = element_blank()
    ) +
    xlab(NULL) +
    ylab(NULL)
}
```

The figure shows the true standard deviations in black, the posterior means in blue and the 90% posterior invervals as contours.

```{r}
tibble(time=times,
       sigma=sigmas,
       sigma.hat=sigmas.hat.mean,
       sigma.hat.lower=sigmas.hat.ints[, 1],
       sigma.hat.upper=sigmas.hat.ints[, 2]) %>%
  plot.sigmas()
```

In the end let's look at the MSE of the standard deviations and their estimated posterior means.

```{r}
mean((sigmas - sigmas.hat.mean)**2)
```

# Fitting a latent Gaussian process

One alternative to modelling this data set is to model the standard deviations explicitely, i.e., by endowing them with a prior, and using a latent GP to model the mean of this prior.


$$\begin{align}
f & \sim GP(0, K)\\
\sigma & \sim \text{Normal}^+(f, \tau) \\
y_t & \sim \mathcal{N}(0, \sigma_t)
\end{align}$$

To help Stan a bit with sampling from the posterior, we use a non-centered parameterization for $\sigma$. The model code is shown below:

```{r}
gp.stan.file <- "./_models/gp_volatility_model.stan"
cat(readLines(gp.stan.file), sep="\n")
```

We fit the model as above:

```{r, results='hide', warning=FALSE, message=FALSE, error=FALSE}
m <- cmdstanr::cmdstan_model(gp.stan.file)

fit <- m$sample(
  data=list(N = length(y),
            times = times,
            y = y,
            Q = 50,
            L = 5 / 2 * max(times)),
  seed=123,
  chains =4,
  parallel_chains=4
)
```

The diagnostics of the fit show that the sampler didn't have many problems. However, we observe a divergence.


```{r}
fit$cmdstan_diagnose()
```

Let's also have a look at the posteriors again as above.

```{r}
sigmas.hat <- fit$draws("sigma")
sigmas.hat.mean <- apply(as_draws_matrix(sigmas.hat), 2, mean)
sigmas.hat.ints <- t(apply(as_draws_matrix(sigmas.hat), 2, quantile, c(0.05, 0.95)))

tibble(time=times,
       sigma=sigmas,
       sigma.hat=sigmas.hat.mean,
       sigma.hat.lower=sigmas.hat.ints[, 1],
       sigma.hat.upper=sigmas.hat.ints[, 2]) %>%
  plot.sigmas()
```

Finally, let's have a look at the MSE of the standard deviations and their estimated posterior means gain.

```{r}
mean((sigmas - sigmas.hat.mean)**2)
```

# Discussion

Interestingly the fit using the GP is significantly worse than the fit using the copula process, i.e., the variance around 1 is drastically overestimated. This result is insofar surprising as for the CP model the warping function was not fine-tuned and the priors weren't elicitated properly, but chosen semi-arbitraily. Fitting the GP model required a bit of more fine tuning, specifically parameterizing the standard deviations via a latent mean seemed bit awkward.

# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>

The notebook is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Session info

```{r}
sessionInfo()
```

# References
