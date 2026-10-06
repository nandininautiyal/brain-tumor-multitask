# Robustness-aware multi-task brain tumor MRI

Segmentation-guided tumor classification on the **Figshare brain tumor dataset** (Cheng et al., 3064 T1-CE slices, 233 patients, three tumor types, manual tumor masks), evaluated with **patient-level splits**, input perturbations, adversarial attacks and MC-dropout uncertainty.

## Motivation
Published brain-tumor classifiers on this dataset commonly shuffle *slices* before splitting, so slices from the same patient land in both train and test. They also rarely test robustness or calibration. This project:

1. Trains a modular multi-task U-Net (pretrained ResNet34 / DenseNet121 / MobileNetV2 encoder, optional attention gates, optional **mask-guided classification head**).
2. Quantifies the **leakage gap** between patient-level and slice-level splits using the same architecture.
3. Stress-tests every model with Gaussian noise, bias field, gamma shift, FGSM and PGD (white-box, joint seg+cls loss), and checks whether the **encoder ranking changes under attack**.
4. Measures calibration (ECE, NLL) and whether MC-dropout uncertainty flags attacked or failing inputs.
5. Ablations (base / +attention / +mask guidance / full) and a fast-FGSM adversarially trained variant.

## Run on Kaggle
1. New notebook, upload `brain_tumor_multitask.ipynb`.
2. **Add Input**: the Figshare brain tumor dataset. It must contain the original `.mat` files (struct `cjdata` with `PID` and `tumorMask`). Mirrors converted to JPG/PNG without patient IDs and masks will not work.
3. Settings: accelerator = GPU (if you get "no kernel image is available", pick T4), Internet = On (ImageNet weights).
4. First run with `QUICK = True` (smoke test), then set it to `False`.
5. `folds_to_run=[0]` by default. Extend to `[0,1,2,3,4]` for full cross-validation; finished runs are skipped, so you can split across sessions.

Outputs (in `/kaggle/working/outputs`): `results_long.csv`, `uncertainty_and_meta.csv`, `figs/*.png`, `results/*.json`, `ckpt/*.pt`.

## Limitations
2D slices of a single public dataset (single-center-style acquisition, T1-CE only); no external clinical validation; attacks are white-box on the deterministic model; not a clinical tool.

## Data and citation
Dataset: J. Cheng et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition", PLoS ONE 10(10), 2015 (figshare DOI 10.6084/m9.figshare.1512427). Data is not included in this repository.