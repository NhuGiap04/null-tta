# Null-TTA (Null-Text Embedding Optimisation) — CVPR 2026 (Official Code)

This repository contains the **official implementation** of our CVPR 2026 paper:

**Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation (Null-TTA)**  
Taehoon Kim, Henry Gouk, Timothy Hospedales  
arXiv: https://arxiv.org/abs/2511.20889  

---

## Overview

Null-TTA is a **training-free, test-time alignment method** for text-to-image diffusion models.  
Instead of fine-tuning model parameters, Null-TTA optimizes the **null-text (unconditional) embedding** during sampling to improve a target objective, while preserving prior structure via regularization.

---

## Features

- Training-free alignment (no model fine-tuning)
- Null-text embedding optimisation at inference time
- Compatible with:
  - Stable Diffusion v1.5
  - Stable Diffusion XL (SDXL)
- Optional support for:
  - PickScore
  - Aesthetic score
  - HPSv2
  - ImageReward

---

## Installation

### 1) Create environment

```bash
conda create -n nulltta python=3.10
conda activate nulltta
pip install -e .
```

### 2) Optional reward models

Install only the reward models required for your experiments.

**ImageReward**
```bash
pip install --no-deps image-reward
```

**HPSv2**
Install from:
https://github.com/tgxs002/HPSv2

---

## Usage

### Running Experiments

```bash
python examples/null_tta_sd.py
python examples/null_tta_sd_nograd.py
python examples/null_tta_sdxl.py
```

- `null_tta_sd.py` — Gradient-based Null-TTA on Stable Diffusion v1.5  
- `null_tta_sd_nograd.py` — Non-gradient variant on SD v1.5  
- `null_tta_sdxl.py` — Gradient-based Null-TTA on SDXL  

---

## Important Arguments

- `--target_reward {pickscore,aesthetic,hpsv2,imagereward}`
- `--seed`
- `--num_inference_steps`
- `--num_particles`
- `--min_inner_steps`
- `--max_inner_steps`
- `--lr_uncond`
- `--tampering_coef`
- `--lambda_alpha`
- `--lambda_reg`
- `--phi_variance`

---

## Outputs

Each run typically saves:

- `*_base.png` — Baseline sample
- `*_opt.png` — Null-TTA optimized sample
- `results.csv` — Per-prompt scores
- (Optional) hyperparameter summary CSV

Example directory structure:

```
logs_sdxl/
  SDXL_NullTTA_particle_3_tampering_steps_5_to_25_42_target-pickscore/
    A_100_B_0.002_G_0.01/
      0_base.png
      0_opt.png
      results.csv
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kim2026nulltta,
  title     = {Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation},
  author    = {Taehoon Kim and Henry Gouk and Timothy Hospedales},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgments

This implementation builds upon components and ideas from:

- https://github.com/krafton-ai/DAS
- https://github.com/huggingface/diffusers

We thank the authors for open-sourcing their work.

---