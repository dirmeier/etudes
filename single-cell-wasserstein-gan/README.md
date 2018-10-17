# Single-cell GAN

A GAN for creation of single-cell flourescence microscopy images.

## Introduction

This is a basic implementation of a deep-convolutional Wasserstein-GAN to create single-cell images as they are produced by flourescence microscopes.


## Wasserstein GAN

We used Keras to set up a simple Wasserstein generative adversarial network to estimate a distribution on single cell images as they are produced by flourescence microscopes.

The GAN's discriminator consists of

* four convolutional layers,
* leaky rectifier activation functions,
* a dropout of 25%,
* and batch normalization layers after each convolutional layer.

The GAN's generator uses

* a 100 dimensional (multivariate normal noise model) as input,
* three convolutional layers,
* rectifier activation functions,
* batch normalisation.

As in the original paper ([Wasserstein GAN](https://arxiv.org/abs/1701.07875)) we use `RMSprop` as optimizer.

We use roughly 10000 images of dimension $75 \times 100$ and three different color channels that look like this:

<div align="center">
<img src="https://rawgit.com/dirmeier/machine-learning-notebooks/develop/single-cell-wasserstein-gan/_fig/original.png" alt="Drawing" width="50%" />
</div>

The learnt GAN produces the following images:

<div align="center">
<img src="https://rawgit.com/dirmeier/machine-learning-notebooks/develop/single-cell-wasserstein-gan/_fig/generated_images_channels.png" alt="Drawing" width="50%" />
</div>

<div align="center">
<img src="https://rawgit.com/dirmeier/machine-learning-notebooks/develop/single-cell-wasserstein-gan/_fig/generated_images_full.png" alt="Drawing" width="50%" />
</div>


* Simon Dirmeier <a href="simon.dirmeier@web.de">simon.dirmeier@web.de</a>


