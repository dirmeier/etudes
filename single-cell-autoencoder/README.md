# Single-cell image Autoencoder

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

A denoising, deep convolutional Autoencoder for orthogonal feature extraction from single-cell images.

## Introduction

This is a basic implementation of a denoising, deep-convolutional Autoencoder that can be used to extract orthogonal features from images of populations of single-cells.
This project is still in the making so nothing is final.

## Autoencoding network

We used Keras to set up the simple denoising, deep convolutional autoencoding network.

The Encoder uses

* some convolutional layers with max-pooling,
* two dense layers,
* relu activations

The Decoder uses

* pretty much the same in reverse,
* upsampling instead of max-pooling,
* a sigmoid activation function in the last layer.

We used AdaDelta as optimizer with standard parameters, the cross-entropy as loss and batches of size 64.
For the basic setup, we dont use batch-normalization, dropout, etc.

We use roughly 10000 images of dimension $149 \times 200$ and a single channel.
The autoencoder, with this architecture and parameterization, did not manage to reproduce the images at all. 
To be continued!

<div align="center">
<img src="https://rawgit.com/dirmeier/machine-learning-notebooks/develop/single-cell-autoencoder/_fig/restored_images.png" alt="Drawing" width="100%" />
</div>

* Simon Dirmeier <a href="simon.dirmeier@web.de">simon.dirmeier@web.de</a>


