# 🧠 PGRMamba

**PGRMamba** is a geometry-guided state-space framework for 3D neuronal soma detection in large-scale whole-brain microscopy images. It combines a Dual-Stage Filtering Block (DSFB), Geometry-Guided Sparse Mamba (GSM), and Gradient-Calibrated Affinity Loss (GCA-Loss).

This repository provides the **training, inference, and evaluation code**, together with **pretrained model weights**. The 3D-NSD dataset is distributed separately.

## 📂 Code Overview

| File / Directory | Description |
| --- | --- |
| `feature_extract_net.py` | PGRMamba backbone and DSFB |
| `mamba/` | PGE, RegMamba, GSM, and bundled Mamba implementation |
| `get_loss.py` | GCA-Loss |
| `train_detction_um.py` | Model training and validation |
| `predict.py` | Soma detection on unlabeled TIFF volumes |
| `evaluate.py` | Evaluation on labeled TIFF/CSV pairs |
| `causal-conv1d/` | Bundled CUDA dependency |

## 🗂️ 3D-NSD Dataset

**[Download the 3D-NSD dataset (Google Drive)](https://drive.google.com/file/d/1nXf-EZZFgj0EDn7J1_erZiWTF3JN8ztq/view?usp=drive_link)**

3D-NSD contains **184 annotated 3D microscopy sub-volumes** from **five mouse brains**, with **42,654 soma annotations**. Each sub-volume contains **256 × 256 × 256 voxels**, with voxel spacing of **1 × 1 × 2 µm**. Image volumes are provided as TIFF files, and soma centers as matching CSV files containing `x,y,z` voxel coordinates.

The released block-wise split comprises **125 training** and **59 validation** sub-volumes:

```text
3D_NSD/
├── train/        # TIFF/CSV pairs (125 volumes)
├── val/          # TIFF/CSV pairs (59 volumes)
├── README.md     # Dataset description
└── LICENSE.txt   # Dataset license (CC BY-NC 4.0)
```

Place the downloaded data in `3D_NSD/` under the repository directory, or specify its location using `--data_root`. **This split is block-wise, not brain-independent**; the five-fold leave-one-brain-out evaluation is described separately in the paper.

## ⚙️ Installation

The reference environment uses **Linux, Python 3.12, PyTorch 2.6.0, CUDA 12.4**, and an NVIDIA GPU. Building the bundled CUDA extensions requires the CUDA development toolkit (`nvcc`).

```bash
conda create -n pgrmamba python=3.12 -y
conda activate pgrmamba
cd PGRMamba

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

# Adjust this path to the CUDA toolkit installed on your system.
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/install_extensions.sh
```

Installation and GPU execution may require adjustments for different CUDA/PyTorch environments.

## 🚀 Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_detction_um.py \
  --data_root ./3D_NSD \
  --seed 2026 --epochs 1000 \
  --distance_um 6 --spacing_xyz 1 1 2
```

Training uses Adam with batch size 1 and selects the checkpoint with the best validation micro-F1 score. The paper reports results from three independent runs with seeds **2024, 2025, and 2026**.

## 🔎 Inference

Pretrained model weights are publicly provided. Specify the selected weight file with `--model_path` to detect somata in unlabeled TIFF volumes:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --model_path /path/to/model_weights.pth \
  --data_path ./unlabeled_blocks \
  --out_dir outputs/predictions \
  --seed 2026 --distance_um 6 --spacing_xyz 1 1 2
```

Predicted soma centers are saved as CSV files with `x,y,z` voxel coordinates.

## 📊 Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model_path /path/to/model_weights.pth \
  --data_path ./3D_NSD/val \
  --out_dir outputs/validation \
  --seed 2026 --distance_um 6 --spacing_xyz 1 1 2
```

The 3D-NSD evaluation uses physical-distance matching with a **6 µm** radius and reports Precision, Recall, and F1-score. The paper describes the full benchmark protocols, including WMBSD and brain-independent validation.

## 📜 Licenses

- **Source code:** [MIT License](https://opensource.org/license/mit). See the repository's code license file.
- **3D-NSD dataset:** [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Attribution is required, and commercial use is not permitted. Adapted versions may be shared for non-commercial purposes subject to the license terms.
- **Bundled third-party components:** Their respective original licenses and notices apply.
