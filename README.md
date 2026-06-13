# ECG Origin Classification for VT/PVC Localization

## Description

This repository implements a biomedical machine learning pipeline for classifying ventricular tachycardia / premature ventricular contraction (VT/PVC) origin from ECG data. The script `final_code.py` loads multiple ECG datasets and clinical data, preprocesses the signals, extracts QRS morphology features, and trains classification models.

Key features of the project:
- Loads and normalizes 12-lead ECG data from MAT files.
- Extracts binary RV vs LV origin labels and finer sublocation labels.
- Uses Teknon clinical dataset (`data/full_data_corrected_2024.pkl`) together with ECG segmentation models in `models/`.
- Performs QRS segmentation using an ensemble of pretrained PyTorch models.
- Builds a balanced classification pool with holdout validation.
- Trains classical classifiers and a 1D CNN for binary RVOT/LVOT prediction.
- Saves evaluation confusion matrices and feature analysis figures in `figures/`.

## Data sources

The script depends on the following files:
- `data/QRS_CARTO2.mat`
- `data/QRS_Database2.mat`
- `data/QRS_Sims2.mat`
- `data/full_data_corrected_2024.pkl`
- `models/model.1`, `models/model.2`, `models/model.3`, `models/model.4`, `models/model.5`

## Results table

The project reports the following evaluation metrics for each experiment:

| Experiment | Evaluation set | Primary metrics |
|---|---|---|
| Binary classification (RVOT vs LVOT) | Internal balanced test | Accuracy, Balanced Accuracy |
| Binary classification (RVOT vs LVOT) | Teknon-only holdout | Accuracy, Balanced Accuracy |
| Intermediate multiclass classification | Internal balanced test | Accuracy, Balanced Accuracy |
| Fine-grained multiclass classification | Internal balanced test | Accuracy, Balanced Accuracy |
| CNN binary classification | Cross-validation | CV Accuracy, CV Balanced Accuracy |
| CNN binary classification | Holdout test | Accuracy, Balanced Accuracy, Macro-average Sensitivity |

> Exact numeric values are printed by `final_code.py` during execution and are also saved through the script's figure output routines.

## Reproduce results

Open a command prompt in the repository root and run:

```powershell
python -m pip install -r requirements.txt
python final_code.py | tee output.txt
```

## Expected outputs

After running `final_code.py`, the script prints summary metrics for each task and saves figures in the `figures/` folder, including:
- `missing_values.png`
- `soo_distribution.png`
- `qrs_extraction.png`
- Confusion matrix plots for internal and holdout evaluations
- SHAP explanation plots if `shap` is installed

## Notes

- The classification pipeline is designed for the available datasets and relies on correct data file locations.
- If any required data files are missing, the script will print warnings or raise errors.
- The script may take significant time to run because it performs multiple training experiments and evaluates cross-validation metrics.
