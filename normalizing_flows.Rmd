---
title: "Normalizing flows for density estimation"
author: "Simon Dirmeier <simon.dirmeier @ web.de>"
date: "October 2020"
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
use_condaenv("ppl-dev")
```

In this notebook we will implement a normalizing flow, the masked autoregressive flow (@papamakarios2017masked), for density estimation using Tensorflow Probability. A useful review on recent trends on normalizing flows is for instance by @papamakarios2019normalizing. Feedback and comments are welcome!

# Normalizing flows

Normalizing flows (NFs) transform simple *base distributions* into rich, complex families of distributions that can be used for density estimation, variational inference, reparameterization of MCMC, or data generation. NFs express the distribution of a random vector $\mathbf{y} \in \mathbb{R}^p$ by applying a transformation $f$ to some random vector $\mathbf{x}$ sampled from $P_X(\mathbf{x})$:

$$\begin{align}
\mathbf{x} & \sim P_Y(\mathbf{x}) \\
\mathbf{y} &= f(\mathbf{x})
\end{align}$$

The defining property of normalizing flows is that the transformation $f$ is invertible as well as differentiable. In order for these two properties to hold, $\mathbf{x}$ must have the same dimensionality as $\mathbf{y}$. With these conditions, the density of $\mathbf{y}$ is well-defined and reads as:

$$\begin{align}
P_Y(\mathbf{y}) & = P_X\left(f^{-1}(\mathbf{y})\right)\left|\text{det} \frac{\partial f^{-1}}{\partial \mathbf{y}}\right| \\
& = P_X\left(\mathbf{x}\right)  \left| \text{det} \frac{\partial f}{\partial \mathbf{x}}  \right|^{-1}
\end{align}$$

where $\frac{\partial f^{-1}}{\partial \mathbf{y}}$ is the $p \times p$-dimensional Jacobian of $f^{-1}$ w.r.t. $\mathbf{y}$. In practice the transformation $f$ consists of a series of invertible, differentiable functions $f_1, \dots, f_K$:

$$
\mathbf{y} = \mathbf{x}_K = f_K \circ \dots f_2 \circ f_1(\mathbf{x}_0)
$$

The density of this transformation is given by:

$$\begin{align}
P_Y(\mathbf{y}) & = P_X\left(\mathbf{x}_0 \right) \prod_{k=1}^K \left| \text{det} \frac{\partial f_k}{\partial \mathbf{x}_{k - 1}}  \right|^{-1}
\end{align}$$

Computing the determinant of a Jacobian is cubic in $p$. In order to be able to use NFs in production, we would like to be able to efficiently compute the determinants of the Jacobians. Most approaches to NFs achieve that by constructing transformations which have triangular Jacobians for which the determinants can be computed in linear time.

## MAFs

[Masked autoregressive flows](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation) (MAFs) model the transformation of a sample $\mathbf{x}$ of the base distribution $P_X$ autoregressively as

$$\begin{align}
y_i &= f\left(x_i\right)\\
y_i &= x_i  \exp \left( \alpha_i \right) + \mu_i \\\\
x_i &= f^{-1}\left(y_i\right)\\
x_i &= \left(y_i - \mu_i \right)  \exp \left( -\alpha_i \right)
\end{align}$$

where $\mu_i = f_{\mu_i}\left( \mathbf{y}_{1:i-1}  \right)$ and $\alpha_i = f_{\alpha_i}\left( \mathbf{y}_{1:i-1}  \right)$ are two scalar functions. Due to the autoregressive structure the Jacobian of the inverse function $f^{-1}$ is lower triangular:

$$\begin{align}
\frac{\partial f^{-1}}{\partial \mathbf{y}} = \begin{pmatrix}
\exp(-\alpha_1)&& \mathbf{0}\\ 
&\ddots&\\
\frac{\partial f^{-1}_{2:p}}{\partial \mathbf{y}_{1:p}} && \exp(-\alpha_{2:p})
\end{pmatrix}
\end{align}$$

such that the determinant is merely the product on the diagonal.

In order to make $\mathbf{y}$ have an autoregressive structure, the authors make use of the approach used by [MADE](https://arxiv.org/abs/1502.03509), i.e., an autoencoder that enforces the autoregressive property by multiplying binary masks with the weight matrices of the autoencoder.

# Implementation

In the following we implement the *masked autoregressive flow* and *masked autoregressive distribution estimation* from scratch using TensorFlow Probability. We first load some required libraries.

```{python}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfb = tfp.bijectors

sns.set_style("ticks", {'font.family':'serif', 'font.serif':'Times New Roman'})
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
```

To implement a MAF, we first need to be able to compute the underlying autoencoder MADE. MADE uses binary masks to enforce autoregressive structure. To do that we first need to compute degree vectors which describe the maximum number of inputs an element $y_i$ can have.

```{python}
def make_degrees(p, hidden_dims):
    m = [tf.constant(range(1, p + 1))]
    for dim in hidden_dims:
        n_min = min(np.min(m[-1]), p - 1)
        degrees = (np.arange(dim) % max(1, p - 1) + min(1, p - 1))
        degrees = tf.constant(degrees, dtype="int32")
        m.append(degrees)
    return m
```

```{python}
make_degrees(2, [4, 4])
```

From these we can compute the binary masks. We don't do that exactly as in the original publication for reason explained later.

```{python}
def make_masks(degrees):
    masks = [None] * len(degrees)
    for i, (ind, outd) in enumerate(zip(degrees[:-1], degrees[1:])):
        masks[i] = tf.cast(ind[:, tf.newaxis] <= outd, dtype="float32")        
    masks[-1] = tf.cast(degrees[-1][:, np.newaxis] < degrees[0], dtype="float32")
    return masks
```

In order to mask a weight matrix, we can use a Dense layer and apply a constraint (hence this way of encoding the mask and not as in the paper).

```{python}
def make_constraint(mask):    
    def _constraint(x):
        return mask * tf.identity(x)
    return _constraint
```

Likewise we initialize a layer using the same mask such that all elements that are being masked are initialized as zero.

```{python}
def make_init(mask):
    def _init(shape, dtype=None):
        return mask * tf.keras.initializers.GlorotUniform(23)(shape)
    return _init
```

We build the autoencoder as a sequence of Keras layers. Since we are going to need two parameters for every component, i.e., to compute $y_i$ we need to compute $f_{\mu_i}\left(\mathbf{y}_{1:i-1}\right)$ and $f_{\sigma_i}\left(\mathbf{y}_{1:i-1}\right)$, we need the last layer to have $2p$ parameters.

```{python}
def make_network(p, hidden_dims, params):
    masks = make_masks(make_degrees(p, hidden_dims))    
    masks[-1] = tf.tile(masks[-1][..., tf.newaxis], [1, 1, params])
    masks[-1] = tf.reshape(masks[-1], [masks[-1].shape[0], p * params])
    
    network =  tf.keras.Sequential([
        tf.keras.layers.InputLayer((p,))
    ])
    for dim, mask in zip(hidden_dims + [p * params], masks):
        layer = tf.keras.layers.Dense(
            dim,
            kernel_constraint=make_constraint(mask),
            kernel_initializer=make_init(mask),
            activation=tf.nn.leaky_relu)
        network.add(layer)    
    network.add(tf.keras.layers.Reshape([p, params]))
    
    return network
```

Let's test this:

```{python}
network = make_network(2, [5, 5], 2)
X = tfd.Normal(0.0, 1.0).sample([5, 2])
network(X)
```

```{python}
network.trainable_variables
```

In order to implement the normalizing flow, we can use TensorFlow Probability's Bijector API. To do that we create a class that inherits form `tfb.Bijector` and override functions for the forward transformation, its inverse and the determinant of the Jacobian. The inverse is easy to compute in a single pass. To sample from $P_Y$ requires performing $p$ sequential passes.

```{python}
class MAF(tfb.Bijector):
    def __init__(self, shift_and_log_scale_fn, name="maf"):
        super(MAF, self).__init__(forward_min_event_ndims=1, name=name)
        self._shift_and_log_scale_fn = shift_and_log_scale_fn
        
    def _shift_and_log_scale(self, y):
        params = self._shift_and_log_scale_fn(y)          
        shift, log_scale = tf.unstack(params, num=2, axis=-1)
        return shift, log_scale
        
    def _forward(self, x):
        y = tf.zeros_like(x, dtype=tf.float32)
        for i in range(x.shape[-1]):            
            shift, log_scale = self._shift_and_log_scale(y)            
            y = x * tf.math.exp(log_scale) + shift
        return y

    def _inverse(self, y):
        shift, log_scale = self._shift_and_log_scale(y)
        return (y - shift) * tf.math.exp(-log_scale)

    def _inverse_log_det_jacobian(self, y):
        _, log_scale = self._shift_and_log_scale(y)
        return -tf.reduce_sum(log_scale, axis=self.forward_min_event_ndims)
    
```

That is all. The inverse and the determinant of its Jacobian are computed as described above. Let's test it:

```{python}
maf = MAF(make_network(2, [5, 5], 2))
maf.forward(X)
```

# Density estimation

We will first test our MAF to estimate the density of the moon data set. We can sample from the moon density using `sklearn`.

```{python}
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

n = 10000
X, _ = datasets.make_moons(n_samples=n, noise=.05)
X = StandardScaler().fit_transform(X)
```

```{python}
ax = sns.scatterplot(
  data=pd.DataFrame(X[:1000], columns=["x", "y"]), 
  x="x", y="y",
  color="black", marker='.', linewidth=0
);
plt.show()
```

We implement the NF by stacking several layers of MAF, i.e. multiple functions $f_i$, followed by a permutation of the components.

```{python}
hidden_dim = [512, 512]  
layers = 2
bijectors = []
for i in range(0, layers):
    made = make_network(2, hidden_dim, 2)
    bijectors.append(MAF(made))
    bijectors.append(tfb.Permute(permutation=[1, 0])) 
    
bijectors = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))
```

To sample and be able to compute the log probabilty of a transformation using an NF we wrap the chain of bijectors as `TransformedDistribution` with a standard normal as base distribution.

```{python}
distribution = tfd.TransformedDistribution(
    distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(2)),
    bijector=bijectors
)
```

To fit the density, we can again use TensorFlow's API with the negative log likelihood as loss function

```{python}
x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.optimizers.Adam(), loss=lambda _, log_prob: -log_prob)
_ = model.fit(x=X,
              y=np.zeros((X.shape[0], 0), dtype=np.float32),
              batch_size= X.shape[0],
              epochs=2000,
              steps_per_epoch=1,
              verbose=0,
              shuffle=True)
```

Having the weights optimized, we can sample from the distribution and check if the trained model is similar to the moon distribution.

```{python}
samples = distribution.sample(1000)
samples = pd.DataFrame(samples.numpy(), columns=["x", "y"])

ax = sns.scatterplot(
  data=samples,
  x="x", y="y", 
  color='black', marker='.', linewidth=0
);
plt.show()
```

# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>

The notebook is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# References
