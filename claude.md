# CLAUDE.md — CAP5516 Assignment 3: LoRA Fine-tuning of MobileSAM for Nuclei Instance Segmentation

## Project Overview

This project implements parameter-efficient fine-tuning of MobileSAM using LoRA (Low-Rank Adaptation) for nuclei instance segmentation on the NuInsSeg dataset. The goal is to reproduce and improve upon the baseline results reported in Table 3 of the NuInsSeg paper, evaluated via five-fold cross-validation using Dice, AJI, and PQ metrics.

---

## Repository Structure

```
assignment3/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   └── nuinsseg/                  # Raw dataset placed here after download
├── preprocessing/
│   └── generate_masks.py          # Adapted from NuInsSeg GitHub repo
├── datasets/
│   └── nuinsseg_dataset.py        # PyTorch Dataset class
├── models/
│   ├── lora.py                    # LoRA layer definition and injection logic
│   └── mobile_sam_lora.py         # MobileSAM wrapped with LoRA
├── utils/
│   ├── metrics.py                 # Dice, AJI, PQ computation
│   ├── prompt_utils.py            # Centroid extraction for SAM prompts
│   └── losses.py                  # Dice loss + Focal loss
├── train.py                       # Training loop with 5-fold CV
├── evaluate.py                    # Evaluation script, outputs results table
└── visualize.py                   # Qualitative comparison figure generation
```

---

## Phase 1: Environment Setup

Create a Python 3.10 virtual environment. Install dependencies in this order:

1. PyTorch (CUDA-enabled, match the available CUDA version)
2. MobileSAM: clone from https://github.com/ChaoningZhang/MobileSAM and install with `pip install -e .`
3. Additional packages: `opencv-python`, `scikit-learn`, `scikit-image`, `matplotlib`, `tqdm`, `albumentations`, `numpy`, `pandas`

Generate a `requirements.txt` after setup.

---

## Phase 2: Data Preparation

### Download

The dataset is available at: https://zenodo.org/records/10518968

Place the extracted contents under `data/nuinsseg/`. The dataset contains H&E histological images alongside XML annotation files for multiple tissue types.

### Mask Generation

Adapt the mask generation scripts from https://github.com/masih4/NuInsSeg to produce three outputs per image:

- Binary semantic mask (foreground nuclei vs background)
- Instance label map (each nucleus has a unique integer ID)
- Distance map (used for centroid extraction at inference)

Save outputs alongside the original images in a consistent structure, for example:

```
data/nuinsseg/
├── <tissue_type>/
│   ├── images/
│   ├── masks/         # binary masks
│   ├── instances/     # instance label maps
│   └── distances/     # distance maps
```

### Five-Fold Split

Build the cross-validation splits at the image level using `sklearn.model_selection.KFold` with `shuffle=True` and `random_state=42`. Generate and save the five fold index files as JSON so splits are reproducible across runs.

---

## Phase 3: Dataset Class

Implement `datasets/nuinsseg_dataset.py` as a PyTorch `Dataset`. It should:

- Accept a list of image paths and a mode (`train` or `val`)
- Load the image, binary mask, and instance label map for each sample
- Apply augmentations during training (horizontal flip, vertical flip, random rotation, color jitter) using `albumentations`
- Resize images to 1024x1024 to match MobileSAM's expected input resolution
- Return a dictionary with keys: `image`, `mask`, `instances`, `centroids`

Centroids should be extracted from the instance label map using `skimage.measure.regionprops` and included in the returned dictionary so the training loop can construct SAM point prompts directly.

---

## Phase 4: LoRA Implementation

Implement `models/lora.py` with the following components.

### LoRALayer

A wrapper around an existing `nn.Linear` layer that:

- Freezes the original weight and bias
- Adds two trainable matrices A (shape: rank × d_in) and B (shape: d_out × rank)
- Initialises A with Kaiming uniform, B with zeros
- In the forward pass, computes `original(x) + scaling * (x @ A.T @ B.T)` where `scaling = alpha / rank`

Expose `rank` and `alpha` as constructor arguments. A sensible default is `rank=4`, `alpha=1.0`.

### inject_lora function

A function that takes a MobileSAM model and injects LoRA into the query and value projection matrices of every attention block in the image encoder. MobileSAM's ViT uses a fused `qkv` linear layer. Handle this by splitting the fused projection into three separate linear layers (Q, K, V), wrapping only Q and V with `LoRALayer`, and leaving K frozen and unwrapped.

After injection, freeze all parameters not belonging to a LoRALayer. Print and return the count of tunable parameters.

---

## Phase 5: Model Wrapper

Implement `models/mobile_sam_lora.py`. This module should:

- Load MobileSAM with pretrained weights from the official checkpoint
- Call `inject_lora` on the image encoder
- Expose a `forward` method that accepts a batched image tensor and a list of point prompts, runs the image encoder, constructs the prompt embeddings, runs the mask decoder, and returns predicted masks and IoU scores
- Keep the mask decoder unfrozen so it can also adapt to the nuclei domain

---

## Phase 6: Loss Functions

Implement `utils/losses.py` with:

- **DiceLoss**: operates on sigmoid-activated predictions and binary targets
- **FocalLoss**: binary focal loss with configurable gamma (default 2.0) and alpha (default 0.25)
- **CombinedLoss**: returns the sum of Dice loss and Focal loss, with equal weighting

---

## Phase 7: Prompt Utility

Implement `utils/prompt_utils.py` with two functions.

**extract_centroids_from_instances**: takes an instance label map, uses `skimage.measure.regionprops`, and returns a list of (row, col) centroid coordinates, one per nucleus.

**build_point_prompts**: takes a list of centroids and formats them into the tensor structure expected by MobileSAM's prompt encoder, specifically a point coordinates tensor of shape (N, 1, 2) and a point labels tensor of shape (N, 1) with all values set to 1 (foreground).

---

## Phase 8: Metrics

Implement `utils/metrics.py` with the following. Adapt directly from the NuInsSeg evaluation code at https://github.com/masih4/NuInsSeg where possible rather than reimplementing from scratch.

- **dice_coefficient**: pixel-level Dice between predicted binary mask and ground truth binary mask
- **aggregated_jaccard_index (AJI)**: instance-level metric that matches predicted instances to ground truth instances via greedy assignment and computes the aggregated Jaccard
- **panoptic_quality (PQ)**: computes detection quality (DQ) and segmentation quality (SQ) separately, returns PQ = DQ × SQ
- **compute_all_metrics**: convenience wrapper that returns a dictionary with keys `dice`, `aji`, `pq`, `sq`, `dq`

---

## Phase 9: Training Loop

Implement `train.py` with the following structure.

### Configuration

At the top of the file, define all hyperparameters as a plain dictionary or dataclass:

- `lora_rank`: 4
- `lora_alpha`: 1.0
- `learning_rate`: 1e-4
- `batch_size`: 4
- `num_epochs`: 50
- `image_size`: 1024
- `num_folds`: 5
- `random_state`: 42
- `checkpoint_dir`: `checkpoints/`
- `device`: `cuda` if available, else `cpu`

### Training Procedure

For each fold:

1. Instantiate the dataset for the train split and the val split
2. Instantiate the model, inject LoRA, print tunable parameter count
3. Use AdamW optimiser with the configured learning rate, applied only to parameters with `requires_grad=True`
4. Use a cosine annealing learning rate scheduler over the number of epochs
5. In each training step: load a batch, extract centroids, build point prompts, run the model, compute combined loss, backpropagate, update weights
6. At the end of each epoch, run validation and log Dice, AJI, and PQ
7. Save the best checkpoint per fold based on validation AJI
8. At the end of all folds, compute and print mean and standard deviation across folds for each metric

Log all per-epoch metrics to a CSV file under `logs/`.

---

## Phase 10: Evaluation Script

Implement `evaluate.py` that:

- Loads a trained checkpoint for a given fold
- Runs inference on the corresponding held-out fold
- At inference time, generates point prompts using a uniform grid of points across the image (since ground truth centroids are not available at test time), then filters predicted masks by predicted IoU score and applies non-maximum suppression by mask overlap to produce the final instance set
- Computes and prints Dice, AJI, and PQ
- Saves a CSV summary of results per image

---

## Phase 11: Visualisation

Implement `visualize.py` that:

- Loads a set of example images from the validation set of a given fold
- Runs inference to get predicted instance masks
- Produces side-by-side figures with three panels: original H&E image, ground truth instance mask (colour-coded by instance ID), predicted instance mask (colour-coded by instance ID)
- Saves figures to `figures/` as PNG files
- Selects at least one strong prediction and one weaker prediction to show varied performance

---

## Implementation Notes

- Do not use em dashes anywhere in comments, docstrings, or generated text.
- Prefer descriptive variable names over single-letter abbreviations except for standard conventions like `B` for batch size, `H`, `W` for spatial dimensions.
- Every function must have a docstring stating its inputs, outputs, and purpose.
- All random seeds (NumPy, PyTorch, Python) must be set at the start of `train.py` for reproducibility.
- Intermediate checkpoints should be saved every 10 epochs in addition to the best checkpoint.
- The README must include: environment setup instructions, data download and preprocessing steps, the command to run training, the command to run evaluation, and the command to generate visualisations.

---

## Key External References

- MobileSAM repository: https://github.com/ChaoningZhang/MobileSAM
- NuInsSeg dataset and mask generation code: https://github.com/masih4/NuInsSeg
- LoRA fine-tuning reference for SAM: https://github.com/mazurowski-lab/finetuneSAM
- LoRA for ViT reference: https://github.com/JamesQFreeman/LoRA-ViT
- NuInsSeg paper (for Table 3 experimental protocol): https://arxiv.org/pdf/2308.01760