# Robustness-Aware Multi-Task Brain Tumor MRI

A robustness-aware multi-task learning pipeline for **brain tumor classification and segmentation** on the Figshare brain tumor MRI dataset. The project uses patient-level evaluation to avoid slice-level data leakage and studies model behavior under distribution shifts, adversarial attacks, and uncertainty.

## Overview

The project investigates whether brain tumor classifiers remain reliable when evaluated under stricter patient-level splits and input perturbations.

The pipeline:

- Trains a modular **multi-task U-Net** with a segmentation decoder and classification head.
- Supports **ResNet34, DenseNet121, and MobileNetV2** pretrained encoders.
- Supports optional **attention gates** and **mask-guided classification**.
- Compares **patient-level vs. slice-level splits** to quantify data leakage.
- Evaluates robustness to:
  - Gaussian noise
  - Bias-field perturbations
  - Gamma shifts
  - FGSM
  - PGD
- Evaluates calibration using **Expected Calibration Error (ECE)** and **Negative Log-Likelihood (NLL)**.
- Uses **MC-dropout uncertainty** to study whether uncertainty can identify attacked or failing inputs.
- Includes ablations for the base model, attention, mask guidance, and the full model.
- Includes a fast-FGSM adversarially trained variant.

## Dataset

The project uses the **Figshare Brain Tumor Dataset**:

- 3,064 T1-CE MRI slices
- 233 patients
- 3 tumor classes
- Manual tumor segmentation masks
- Original `.mat` files containing the `cjdata` structure, including patient IDs (`PID`) and `tumorMask`

The dataset is **not included** in this repository.

## Experimental Design

### Patient-Level Evaluation

Slices from the same patient are kept within the same split. This prevents slices from the same patient appearing in both training and test sets.

A slice-level split is also evaluated using the same architecture to quantify the resulting **data leakage gap**.

### Robustness Evaluation

Models are evaluated on clean inputs and under multiple input perturbations and white-box adversarial attacks:

- Gaussian noise
- Bias-field shifts
- Gamma shifts
- FGSM
- PGD

FGSM and PGD attacks use a joint segmentation + classification loss.

The experiments also examine whether the ranking of different encoders changes under increasing attack strength.

### Uncertainty and Calibration

The pipeline evaluates:

- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- MC-dropout predictive uncertainty
- Whether uncertainty can identify attacked or incorrectly classified samples

## Models

| Model | Encoder | Multi-task | Attention | Mask Guidance |
|---|---|---:|---:|---:|
| ResNet34 | ResNet34 | ✓ | Optional | Optional |
| DenseNet121 | DenseNet121 | ✓ | Optional | Optional |
| MobileNetV2 | MobileNetV2 | ✓ | Optional | Optional |

Additional experiments include:

- Base multi-task model
- Attention-based model
- Mask-guided classification model
- Full model
- Adversarially trained ResNet34
- Slice-level split comparison

## Run on Kaggle

The recommended way to run the full experiment is using **Kaggle Notebooks with GPU acceleration**.

### 1. Upload the Notebook

Create a new Kaggle notebook and upload:

```text
brain-tumor-multitask.ipynb
```

### 2. Add the Dataset

Use **Add Input** and add the Figshare Brain Tumor Dataset.

The pipeline expects the **original `.mat` files** containing:

```text
cjdata
├── PID
├── tumorMask
└── ...
```

Converted JPG/PNG versions without patient IDs and tumor masks are not sufficient for this pipeline.

### 3. Configure the Accelerator

Use a GPU accelerator.

A **T4 GPU** can be used if another accelerator causes compatibility issues.

Internet access should be enabled if pretrained ImageNet weights need to be downloaded.

### 4. Run a Smoke Test

Before running the complete experiment, set:

```python
QUICK = True
```

This runs a shortened version of the experiment to verify that:

- the dataset is detected correctly,
- preprocessing works,
- the model can be initialized,
- training runs successfully, and
- the evaluation pipeline works.

### 5. Run the Full Experiment

After the smoke test succeeds, set:

```python
QUICK = False
```

The full configuration trains the selected models for the configured number of epochs, subject to early stopping.

By default, the notebook runs:

```python
folds_to_run = [0]
```

For full cross-validation, this can be changed to:

```python
folds_to_run = [0, 1, 2, 3, 4]
```

Completed runs are automatically skipped when their result files already exist, allowing experiments to be distributed across multiple Kaggle sessions.

### 6. Generated Outputs

Successful runs generate the following structure under:

```text
/kaggle/working/outputs/
```

```text
outputs/
├── results/
│   └── *.json
├── ckpt/
│   └── *.pt
├── figs/
│   └── *.png
├── results_long.csv
└── uncertainty_and_meta.csv
```

The generated outputs contain model results, uncertainty metrics, evaluation data, plots, and checkpoints.

These files are **generated artifacts** and are not required to be committed to the repository.

## Repository Structure

```text
brain-tumor-multitask/
├── brain_tumor_multitask.py
├── brain-tumor-multitask.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Large datasets, model checkpoints, caches, and generated artifacts are excluded using `.gitignore`.

## Limitations

- Evaluation uses 2D slices from a single public dataset.
- The dataset represents a limited acquisition setting and is not an external clinical validation cohort.
- No external clinical dataset is used for validation.
- Adversarial attacks are white-box attacks against the deterministic model.
- The study focuses on robustness and reliability rather than clinical deployment.
- The system is **not a clinical diagnostic tool**.

## Citation

J. Cheng et al., **"Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition,"** *PLoS ONE*, 10(10), 2015.

Dataset: Figshare, DOI: `10.6084/m9.figshare.1512427`