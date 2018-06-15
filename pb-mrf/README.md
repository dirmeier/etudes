<h1 align="center"> pb-mrf </h1>

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

A pairwise binary Markov random field test in Python.

## Introduction

This is a reimplementation of [Robinson's paper](https://doi.org/10.1093/bioinformatics/btx244) of a pairwise binary Markov random field using graph cuts for energy function optimization.

## Installation

The notebook depends on a couple of standard libraries, including a [python wrapper](https://github.com/yujiali/pygco) for gco to do graph cuts on the MRF. Install the wrapper as described and the dependencies using

```bash
  pip install -r requirements.txt
```

That should let you run the notebook.

## Markov random field

**Disclaimer: there might be bugs so don't underestimate results here**.

Robinson *et al* present a toy example using a MRF for inference of binary labels from a mixture of Gaussians.

We frist create the graph as in the paper and sample data from two normal distributions with means $ mu_0 = -2 $, $ mu_1 = 2 $ and $ var_0 = var_1 = 2 $.

<div align="center">
<p>Adjacency matrix</p>
<img src="https://github.com/dirmeier/pb-mrf/blob/master/_fig/adj.png" alt="Drawing" width="50%" />
<p>Densities of the two distributions</p>
<img src="https://github.com/dirmeier/pb-mrf/blob/master/_fig/density.png" alt="Drawing" width="50%" />
<p>Graph with nodes colored using values of sampled data</p>
<img src="https://cdn.rawgit.com/dirmeier/pb-mrf/a6c8c868/_fig/graph_obs.svg" alt="Drawing" width="35%" />
</div>

Then we minimize the MRF's energy function using the negative log-likelihood as unary potentials and the generated graph with edge weights of 1 as pairwise potentials. This gives us labels for very observation. Ideally every subgraph has the same color. However, it seems like there is some bug or I missed something.

<div align="center">
<p>Graph with nodes colored using assigned labels</p>
<img src="https://cdn.rawgit.com/dirmeier/pb-mrf/a6c8c868/_fig/graph_labels.svg" alt="Drawing" width="35%" />
</div>

## Author

* Simon Dirmeier <a href="simon.dirmeier@gmx.de">simon.dirmeier@gmx.de</a>


