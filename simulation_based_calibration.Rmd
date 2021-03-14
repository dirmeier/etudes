---
title: "Simulation based calibration"
author: "Simon Dirmeier <simon.dirmeier @ web.de>"
date: "June 2019"
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
knitr::opts_chunk$set(
  comment = NA, warning = FALSE, error = FALSE,
  fig.align = "center", fig.width = 11, fig.height = 4,
  dpi = 360
)
```

Simulation-based calibration (SBC; @talts2020validating) is a method for visually validating Bayesian inferences. SBC is useful for detection of either misspecified models, inaccurate computation or bugs in the 
implementation of a probabilistic program. So SBC is not a validation for the inference itself, but the technical aspect of the program.

# SBC

Consider a generative model:

\begin{align}
\tilde{\theta} & \sim P(\theta) \\
\tilde{y} & \sim P(y \mid \tilde{\theta}) \\
\{ \theta_1, \dots, \theta_L \} & \sim P(\theta \mid \tilde{y})
\end{align}

The rank of a prior sample $\tilde{\theta}$ in comparison to an *exact* posterior sample $\{ \theta_1, \dots, \theta_L \}$: 

\begin{align}
r(\{ \theta_1, \dots, \theta_L \}, \tilde{\theta}) = \sum_l^L \mathbb{I}(\theta_l < \tilde{\theta}) + 1
\end{align}

is a discrete-uniform random variable in $[1, L + 1 ]$ (see the original paper for a proof). Thus we can use this as a testing procedure if our inferences work.

We follow Algorithm 2 from the original paper and implement SBC in Stan.

```{r}
suppressMessages({
  library(tidyverse)
  library(ggthemes)
  library(colorspace)

  library(rstan)
})

set.seed(23)
options(mc.cores = parallel::detectCores())
```

# Implementation

The idea behind SBC is fairly simple. For $N$ iterations we run a while loop to create a posterior sample that has at least a target effective sample size $n_{eff}$ (which is set by us). We do so by resampling and thinning until the posterior sample for every iteration reaches the target $n_{eff}$ (within the while loop). Then we compute the rank for every parameter using the sum as defined above. That is it. If the inference worked, the ranks are uniformly distributed.

```{r}
sbc <- function(model, data) {
  ranks <- matrix(0, N, 2, dimnames = list(NULL, c("mu", "sigma")))
  for (n in seq(N))
  {
    thin <- init_thin
    while (thin < max_thin) {
      fit <- suppressWarnings(
        sampling(model,
          data = data,
          chains = 1, iter = 2 * thin * L,
          thin = thin, control = list(adapt_delta = 0.99), refresh = 0
        )
      )
      n_eff <- summary(fit)$summary["lp__", "n_eff"]
      if (n_eff >= target_neff) break
      thin <- 2 * thin
    }
    ranks[n, ] <- apply(rstan::extract(fit)$idsim, 2, sum) + 1
  }
  ranks
}
```

# Use cases

We start with a simple example where we set two parameters, generate data from them and then compare the posterior to these parameters. The comparison is done in the `generated quantities` block.

```{r}
model.file <- "_models/sbc_1.stan"
cat(readLines(model.file), sep = "\n")
```

```{r, message=FALSE, warning=FALSE, echo=FALSE, error=FALSE, results='hide'}
model <- stan_model(model.file)
```

We run the loop $5000$ times and sample $100$ times. We also set some other parameters that Stan or SBC needs, such as `init_thin` which specifies the period for saving samples or `target_neff` which is the effective sample size we want to have.

```{r}
N <- 5000
L <- 100
init_thin <- 1
max_thin <- 64
target_neff <- .8 * L
```

We also define a histogram plotting method for the ranks.

```{r}
plot.fit <- function(fit) {
  as.data.frame(fit) %>%
    tidyr::gather("param", "value") %>%
    ggplot() +
    geom_histogram(aes(value), bins = 30) +
    facet_grid(. ~ param) +
    theme_tufte() +
    theme(
        axis.text = element_text(colour = "black", size = 15),
        strip.text = element_text(colour = "black", size = 15)
    ) +
    xlab(NULL) +
    ylab(NULL)
}
```

Then we run SBC and plot the ranks. We create some artifical data for the model first.

```{r, results='hide'}
n <- 10
x <- rnorm(n)

fit <- sbc(model, list(x = x, n = n))
plot.fit(fit)
```

Given the low number of trials, the histogram looks fairly uniform (as expected since the model was specified correctly). Next we test some pathological cases to see if the ranks change with mis-specified models.

```{r}
model.file <- "_models/sbc_2.stan"
cat(readLines(model.file), sep = "\n")
```

```{r, message=FALSE, warning=FALSE, echo=FALSE, error=FALSE, results='hide'}
model <- stan_model(model.file)
fit <- sbc(model, list(x = x, n = n))
plot.fit(fit)
```

And another pathological example:

```{r}
model.file <- "_models/sbc_3.stan"
cat(readLines(model.file), sep = "\n")
```

```{r, message=FALSE, warning=FALSE, echo=FALSE, error=FALSE, results='hide'}
model <- stan_model(model.file)
fit <- sbc(model, list(x = x, n = n))
plot.fit(fit)
```


# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>

The notebook is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Session info

```{r}
sessionInfo()
```

# References
