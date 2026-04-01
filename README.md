# MobileSAM-LoRA for Nuclei Instance Segmentation

Parameter-efficient fine-tuning of MobileSAM using LoRA (Low-Rank Adaptation)
for nuclei instance segmentation on the NuInsSeg dataset.
Evaluated by five-fold cross-validation using Dice, AJI, and PQ metrics,
following the experimental protocol of Table 3 in the NuInsSeg paper.

---

## Environment Setup

### 1. Create a Python 3.10 virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate   # Linux/macOS
```

### 2. Install PyTorch (match your CUDA version)

```bash
# CUDA 11.8 example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install MobileSAM

```bash
git clone https://github.com/ChaoningZhang/MobileSAM
cd MobileSAM
pip install -e .
cd ..
```

### 4. Download the MobileSAM checkpoint

Download `mobile_sam.pt` from the MobileSAM GitHub releases and place it in
the project root directory.

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Data Download and Preprocessing

### 1. Download the NuInsSeg dataset

```bash
# Option A: Zenodo (direct download)
# Visit https://zenodo.org/records/10518968 and download the archive.

# Option B: Kaggle
# Visit https://www.kaggle.com/datasets/ipateam/nuinsseg
```

Extract the contents to `data/nuinsseg/`. The expected layout after extraction:

```
data/nuinsseg/
├── Adrenal_gland/
│   ├── tissue images/
│   │   └── *.png
│   └── mask/
│       └── *.xml
├── Breast/
│   └── ...
```

### 2. Generate masks from XML annotations

```bash
python preprocessing/generate_masks.py --data_root data/nuinsseg
```

This creates `masks/`, `instances/`, and `distances/` subdirectories alongside
each tissue type's images. It may take several minutes depending on dataset size.

---

## Training

Run five-fold cross-validation training:

```bash
python train.py \
    --checkpoint mobile_sam.pt \
    --data_root data/nuinsseg
```

To train a single fold (zero-indexed):

```bash
python train.py \
    --checkpoint mobile_sam.pt \
    --data_root data/nuinsseg \
    --fold 0
```

Checkpoints are saved to `checkpoints/fold_N/`. Per-epoch metrics are logged
to `logs/fold_N_metrics.csv`. The best checkpoint per fold is selected based
on validation AJI.

---

## Evaluation

Evaluate a trained checkpoint on its held-out fold:

```bash
python evaluate.py \
    --checkpoint checkpoints/fold_1/best_checkpoint.pt \
    --fold_index 0 \
    --data_root data/nuinsseg \
    --output_csv fold_1_results.csv
```

Optional flags:
- `--grid_step 64`: spacing (pixels) between grid prompt points (smaller = more prompts)
- `--iou_threshold 0.5`: minimum predicted IoU to keep a candidate mask

---

## Visualisation

Generate side-by-side comparison figures (H&E image, GT mask, predicted mask):

```bash
python visualize.py \
    --checkpoint checkpoints/fold_1/best_checkpoint.pt \
    --fold_index 0 \
    --data_root data/nuinsseg \
    --num_examples 6 \
    --figures_dir figures
```

Figures are saved as PNG files under `figures/`. The script selects examples
with strong predictions and weaker predictions to illustrate a range of performance.

---

## Architecture Overview

MobileSAM consists of:
- **Image encoder**: TinyViT-5M with window attention blocks
- **Prompt encoder**: embeds point, box, and mask prompts
- **Mask decoder**: two-way transformer producing mask logits and IoU predictions

LoRA is injected into the **Q and V projection matrices** of every window
attention block in the image encoder. The fused QKV linear (dim -> 3*dim) is
split into separate Q (LoRA-wrapped), K (frozen), V (LoRA-wrapped) projections,
replaced by a QKVLoRAWrapper that concatenates their outputs to preserve the
original interface.

All image encoder parameters remain frozen except the LoRA matrices.
The mask decoder is kept fully trainable for domain adaptation.

### Tunable parameter counts (rank=4)

| Component         | Parameters        |
|-------------------|-------------------|
| LoRA matrices     | ~130 K            |
| Mask decoder      | ~3.9 M            |
| Total trainable   | ~4.0 M            |
| Total model       | ~5.7 M            |

---

## Hyperparameters

| Parameter            | Value  |
|----------------------|--------|
| LoRA rank            | 4      |
| LoRA alpha           | 1.0    |
| Learning rate        | 1e-4   |
| Optimiser            | AdamW  |
| LR schedule          | Cosine annealing |
| Epochs per fold      | 50     |
| Prompts per image    | 16     |
| Image size           | 1024   |
| Loss                 | Dice + Focal |

---

## References

- NuInsSeg dataset: https://arxiv.org/pdf/2308.01760
- MobileSAM: https://github.com/ChaoningZhang/MobileSAM
- LoRA: https://arxiv.org/abs/2106.09685
- finetuneSAM: https://github.com/mazurowski-lab/finetuneSAM
- LoRA-ViT: https://github.com/JamesQFreeman/LoRA-ViT
