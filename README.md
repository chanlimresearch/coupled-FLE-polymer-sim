# Code for "Anomalous diffusion in coupled viscoelastic media: A fractional Langevin equation approach"

This repository contains the simulation code used in:

> Chan Lim and J.-H. Jeon,  
> "Anomalous diffusion in coupled viscoelastic media: A fractional Langevin equation approach", (2025).  
> [arXiv:2507.08291](https://doi.org/10.48550/arXiv.2507.08291)

These Python scripts simulate the dynamics of two polymer-based coupled systems:

1. A flexible polymer coupled to a Brownian/macromolecular tracer.
2. A flexible polymer crosslinked to a semiflexible polymer.

Both models follow the coarse-grained Langevin framework described in the manuscript and reproduce the results presented in Fig. 7.

## Requirements

The simulations in the paper were run with:

- Python 3.10.15 (conda-forge)
- NumPy 1.26.4
- CuPy 13.3.0 (CUDA 11.8 build)
- tqdm ≥ 4.60

For reproducibility, we recommend the following `requirements.txt`:

```txt
numpy==1.26.4
tqdm>=4.60
cupy-cuda11x==13.3.0
