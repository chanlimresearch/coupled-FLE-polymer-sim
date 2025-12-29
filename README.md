# Code for "Anomalous diffusion in coupled viscoelastic media: A fractional Langevin equation approach"

This repository contains the simulation code used in:

> Chan Lim and J.-H. Jeon,  
> "Anomalous diffusion in coupled viscoelastic media: A fractional Langevin equation approach", (2025).  
> [Physical Review Research](https://doi.org/10.1103/thv9-s9mq)

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
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/chanlimresearch/coupled-fle-sim.git
cd coupled-fle-sim
```

### 2. Flexible polymer + Brownian macromolecule

This script simulates a flexible polymer (system A) crosslinked to a single Brownian/macromolecular tracer (system B).

```bash
python simulate_flexible_polymer_crosslinked_particle.py \
    --gb 10 \
    --k_spring 0.0 \
    --T 1000000 \
    --gpu 0 \
    --prefix msd_flexA_B \
    --DIR ./data/
```

**Main options:**
- `--gb` : friction coefficient γ_B of system B  
- `--k_spring` : coupling spring constant between A and B  
- `--T` : number of integration steps  
- `--gpu` : GPU device ID  
- `--DIR` : output directory  
- `--prefix` : output filename prefix  

This creates a file such as:

```
./data/msd_flexA_B_gb10.0_kS0.0.npz
```

containing:
- `tLags`  
- `msd_A`  
- `msd_B`  

---

### 3. Flexible + semiflexible polymers

This script simulates a flexible polymer (system A) crosslinked to a semiflexible polymer (system B).

```bash
python simulate_flexible_and_semiflexible_crosslinked_polymers.py \
    --k_A 500 \
    --k_spring 100 \
    --T 1000000 \
    --gpu 0 \
    --prefix msd_flexA_semiB \
    --DIR ./data/
```

**Main options:**
- `--k_A` : spring constant k_A for system A  
- `--k_spring` : coupling spring constant between A and B  
- `--T` : number of integration steps  
- `--gpu` : GPU device ID  
- `--DIR` : output directory  
- `--prefix` : output filename prefix  

This creates a file such as:

```
./data/msd_flexA_semiB_kA500.0_kS100.0.npz
```

containing:
- `tLags`  
- `msd_A`  
- `msd_B`  

---

### 4. Notes

- The example parameters approximate those used in Fig. 7 of the paper.  
- You can adjust `--T`, `--k_spring`, `--k_A`, and `--gb` to explore different regimes.  
- The time step is fixed to `dt = 1e-3` inside the scripts.


