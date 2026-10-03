# Physics-Inspired Phenomenological Constraints for Adversarial Battery Degradation Trajectory Synthesis

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![GAN](https://img.shields.io/badge/Model-WGAN--GP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This repository contains the code for a **physics-inspired, phenomenological WGAN-GP** that generates lithium-ion battery capacity-degradation trajectories. Lightweight inequality penalties that encode observable degradation behaviour are added to the generator objective of a 1D-CNN WGAN-GP. They do not model electrochemistry.

The project was developed as undergraduate research in AI for reliability engineering, with a focus on synthetic degradation data for digital twins and predictive maintenance.

---

## Method

The generator and critic are 1D CNNs trained with the Wasserstein loss and a gradient penalty (computed on random interpolates between real and generated windows).

The proposed generator objective is

```
L_G = L_adv + lambda_phys * L_physics          (lambda_phys = 50)
L_physics = L_mono + L_rate + L_bound
```

with each term a squared-hinge penalty averaged over generated windows and time steps (see `src/physics_loss.py`, `calculate_all_physics_loss`):

| Term | Penalizes |
|---|---|
| `L_mono` | positive capacity increments, `max(0, C_t - C_{t-1})^2` |
| `L_rate` | capacity drops larger than 0.1 per cycle (normalized units), `max(0, C_{t-1} - C_t - 0.1)^2` |
| `L_bound` | values outside the normalized range [-1, 1] |

Four configurations share the same architecture, optimizer and hyperparameters:

| `--loss_type` | Generator penalty |
|---|---|
| `none` | none (baseline WGAN-GP) |
| `tv` | total variation only, `mean(abs(C_t - C_{t-1}))` |
| `monotonicity` | `L_mono` only |
| `all` | `L_physics` (the proposed model, called "All Priors" in the tables) |

Total variation is therefore a **separate baseline** and is not part of the proposed loss.

---

## Repository Structure

```text
Battery-degradation-trajectory/
├── notebooks/
│   ├── 01_data_extraction.ipynb      # NASA B0005/B0006/B0007/B0018 -> 50-cycle windows
│   ├── 02_calce_extraction.ipynb     # CALCE CS2_35..CS2_38 -> 50-cycle windows, 80/20 split
│   └── requirements.txt
├── src/
│   ├── models.py                     # CNN generator and critic
│   ├── models_lstm.py
│   └── physics_loss.py               # TV, monotonicity and combined losses
├── baselines/
│   ├── models_lstm.py
│   └── train_lstm.py                 # LSTM-generator comparison
├── initial_results/                  # NASA training/evaluation scripts
│   ├── prepare_split.py
│   ├── train_wgan.py
│   ├── run_all.py                    # 4 configurations x 5 seeds
│   └── ...
├── phase_2/                          # CALCE and downstream/analysis scripts
│   ├── train_calce_wgan.py
│   ├── run_calce_all.py
│   ├── evaluate_downstream_rul.py
│   ├── evaluate_augmentation_ratio.py
│   ├── evaluate_lstm_comparison.py
│   ├── evaluate_manifold.py / evaluate_manifold_pca.py
│   └── evaluate_mmd.py
├── evaluate_master_tables.py         # five-seed tables for both datasets
├── evaluate_final_tables.py
└── README.md
```

---

## Datasets

Raw data are **not** included; download them and place them under `data/raw/` (NASA `.mat` files) and `data/raw/calce/<cell>/` (CALCE `.xlsx` files).

- **NASA Ames battery aging data** (Saha and Goebel, 2007): cells B0005, B0006, B0007 and B0018, discharge capacity per cycle.
- **CALCE CS2 cells**: CS2_35 to CS2_38, per-cycle discharge capacity (3-point rolling median, cycles outside 0.3-1.3 Ah discarded).

---

## Preprocessing and data split (as implemented)

1. Within each battery, discharge capacities are cut into 50-cycle windows with stride 1.
2. Capacities are Min-Max normalized to [-1, 1]: NASA jointly over all four batteries, CALCE per battery.
3. All windows are shuffled with a fixed seed (42) and split 80/20 into training and test windows.

Adjacent windows overlap by 49 of 50 cycles, so the test set measures **window-level** generalization, not generalization to unseen batteries. Normalization is computed before the split.

| Dataset | Batteries | Cycles | Windows | Train | Test |
|---|---|---|---|---|---|
| NASA | 4 | (not printed by the notebook) | 440 | 352 | 88 |
| CALCE | 4 | 3,876 | 3,680 | 2,944 | 736 |

---

## Experiments

- Main comparison: 4 configurations x 5 seeds (10, 20, 30, 40, 50) on each dataset; 3,000 epochs, batch size 32, Adam (lr 1e-4, betas 0.0/0.9), 5 critic steps, gradient-penalty weight 10.
- `evaluate_master_tables.py` computes DTW (100 random real/synthetic pairs), derivative Wasserstein distance, Violation Area (mean sum of positive increments x 1000) and a linear-kernel MMD, using **1,000 synthetic windows per generator**.
- Architecture comparison (`phase_2/evaluate_lstm_comparison.py`), derivative/CDF and PCA figures, and the downstream experiments use **2,000 synthetic windows** and the **seed-10** generators. The CNN/LSTM comparison is on NASA.
- Downstream forecasting (`phase_2/evaluate_downstream_rul.py`, `phase_2/evaluate_augmentation_ratio.py`) is on **CALCE**: a 49-64-32-1 MLP maps 49 cycles to the next cycle, trained for 200 epochs on 250 real windows plus synthetic windows, evaluated on real test windows. Each condition is a single run.

---

## Findings

Relative to the unconstrained baseline, the proposed constraints reduce violations of the prescribed degradation constraints and improve agreement with real derivative distributions on both datasets. Total-variation regularization attains good global similarity scores (DTW, and MMD on NASA) while suppressing local degradation dynamics. Downstream, adding the proposed synthetic windows did not lower forecasting RMSE in these single-run experiments; the accompanying paper discusses this as a limitation and calls the observation the *Generative Augmentation Paradox* in the context of these experiments.

---

## Installation

```bash
git clone https://github.com/AbhijnanBC/Battery-degradation-trajectory.git
cd Battery-degradation-trajectory
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r notebooks/requirements.txt
```

---

## Research Paper

This repository accompanies the manuscript **Physics-Inspired Phenomenological WGAN for Battery Degradation Trajectory Synthesis**.

---

## Citation

```bibtex
@article{Abhijnan2026,
  title={Physics-Inspired Phenomenological WGAN for Battery Degradation Trajectory Synthesis},
  author={Abhijnan B C},
  year={2026}
}
```

---

## Author

**Abhijnan B C**, Department of Computer Science and Engineering, PES University, Bengaluru, India.
GitHub: https://github.com/AbhijnanBC

## License

Released under the MIT License.

## Acknowledgements

- NASA Ames Prognostics Center
- CALCE Battery Research Group
- PyTorch community
