# Physics-Inspired Phenomenological Constraints for Adversarial Battery Degradation Trajectory Synthesis

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![GAN](https://img.shields.io/badge/Model-WGAN--GP-green)
![Research](https://img.shields.io/badge/Research-Battery%20Digital%20Twin-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

## Overview

This repository contains the implementation of a **Physics-Inspired Phenomenological WGAN-GP** framework for generating physically plausible lithium-ion battery degradation trajectories.

Unlike conventional Generative Adversarial Networks that optimize only statistical similarity, this work introduces lightweight **phenomenological inequality constraints** into the adversarial optimization objective to encourage physically meaningful degradation behaviour.

The project was developed as part of undergraduate research in **AI for Reliability Engineering**, with a focus on synthetic battery degradation data generation for Digital Twins and predictive maintenance.

---

## Motivation

High-quality battery degradation datasets are extremely scarce.

Obtaining complete run-to-failure trajectories requires:

- months or years of battery cycling
- expensive laboratory experiments
- specialized hardware

This data scarcity limits the development of

- Battery Digital Twins
- Remaining Useful Life (RUL) prediction
- State-of-Health estimation
- AI-driven predictive maintenance

Generative models can alleviate this problem, but unconstrained GANs frequently generate

- unrealistic capacity regeneration
- excessive oscillations
- physically impossible degradation behaviour

This repository explores whether lightweight phenomenological constraints can improve synthetic battery trajectory realism without requiring computationally expensive electrochemical simulations.

---

# Proposed Framework

The proposed generator combines:

- Wasserstein GAN with Gradient Penalty (WGAN-GP)
- 1D CNN Generator
- 1D CNN Critic

augmented with three phenomenological constraints:

- Monotonicity Loss
- Total Variation Loss
- Capacity Bound Loss

These constraints are incorporated into the Generator objective:

\[
L_G = L_{adv} + \lambda_{phys}L_{physics}
\]

where

\[
L_{physics}
=
L_{mono}
+
L_{TV}
+
L_{bound}
\]

The objective is to generate trajectories that remain statistically realistic while respecting observable battery degradation behaviour.

---

# Features

- Physics-inspired WGAN-GP
- Phenomenological inequality constraints
- CNN-based generator
- Architecture comparison (CNN vs LSTM)
- Multi-seed evaluation pipeline
- NASA + CALCE benchmark support
- Derivative-distribution analysis
- PCA manifold visualization
- Downstream forecasting evaluation
- Fully reproducible experiments

---

# Repository Structure

```text
Battery-degradation-trajectory/

├── baselines/
│   ├── baseline_wgan/
│   ├── monotonicity/
│   ├── total_variation/
│   └── proposed/
│
├── initial_results/
│
├── notebooks/
│
├── phase_2/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── evaluate_master_tables.py
├── evaluate_final_tables.py
│
└── README.md
```

---

# Datasets

Experiments were performed using two public benchmark datasets.

### NASA Randomized Battery Usage Dataset

- Run-to-failure lithium-ion degradation
- Four batteries
- Capacity measurements over charge/discharge cycles

---

### CALCE Battery Dataset

- CS2 battery series
- Longer degradation trajectories
- Used for cross-dataset validation

---

# Experimental Pipeline

The complete workflow is

```text
Battery Dataset
        │
        ▼
Sliding Window Generation
        │
        ▼
Normalization
        │
        ▼
WGAN-GP Training
        │
        ▼
Phenomenological Constraints
        │
        ▼
Synthetic Trajectories
        │
        ▼
Evaluation
```

Evaluation consists of

- DTW
- MMD
- Derivative Wasserstein Distance
- Violation Area
- PCA
- CDF
- Downstream Forecasting

---

# Results

The proposed phenomenological constraints consistently

- reduce non-physical degradation behaviour
- improve derivative realism
- preserve latent manifold geometry
- stabilize adversarial training

An interesting observation from this work is what we refer to in the accompanying paper as the **Generative Augmentation Paradox**:

> Improved marginal generative realism does not necessarily imply improved downstream predictive performance.

This suggests that physically realistic synthetic trajectories do not automatically preserve the conditional dynamics required for forecasting tasks.

---

# Installation

Clone the repository

```bash
git clone https://github.com/AbhijnanBC/Battery-degradation-trajectory.git

cd Battery-degradation-trajectory
```

Create a virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running Experiments

Train the baseline model

```bash
python train_baseline.py
```

Train the proposed model

```bash
python train_physics.py
```

Evaluate

```bash
python evaluate_master_tables.py
```

Generate publication tables

```bash
python evaluate_final_tables.py
```

---

# Technologies

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib

---

# Research Paper

This repository accompanies the research paper

> **Physics-Inspired Phenomenological Constraints for Adversarial Battery Degradation Trajectory Synthesis**

The manuscript includes

- methodology
- experiments
- benchmark evaluation
- qualitative analysis
- discussion
- limitations
- future work

---

# Future Work

Potential directions include

- Diffusion-based battery generators
- Transformer architectures
- Full trajectory generation
- Adaptive constraint weighting
- Multi-chemistry datasets
- Physics-informed diffusion models
- Sequence-aware downstream predictors

---

# Citation

If you use this work in your research, please cite:

```bibtex
@article{Abhijnan2026,
  title={Physics-Inspired Phenomenological Constraints for Adversarial Battery Degradation Trajectory Synthesis},
  author={Abhijnan B C},
  year={2026}
}
```

---

# Author

**Abhijnan B C**

Department of Computer Science and Engineering

PES University

Bengaluru, India

GitHub

https://github.com/AbhijnanBC

---

# License

This project is released under the MIT License.

---

## Acknowledgements

- NASA Ames Prognostics Center
- CALCE Battery Research Group
- PyTorch Community
