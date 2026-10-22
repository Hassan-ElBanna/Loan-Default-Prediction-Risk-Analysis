# Loan Default Prediction & Risk Analysis

An end-to-end machine learning pipeline for predicting credit default risk, built on the
[**Home Credit — Credit Risk Model Stability**](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
Kaggle competition dataset. The project spans exploratory data analysis, multi-table feature
engineering with Polars, GPU-accelerated XGBoost training, and MLflow-tracked experiment
management — all version-controlled with Git + DVC and backed by DagsHub.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Model & Training](#model--training)
- [Results](#results)
- [Experiment Tracking](#experiment-tracking)
- [Setup](#setup)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)

## Overview

The Home Credit dataset is split across 20+ relational parquet tables organized by depth
(0, 1, 2). Each loan application (`case_id`) has a single base + static row, but is linked to
dozens of historical records in credit bureaus, previous applications, tax registries, deposit
accounts, and person/debitcard info. The challenge is two-fold:

1. **Aggregation** — collapse variable-length historical records into a single feature vector
   per `case_id`.
2. **Class imbalance + stability** — defaults are rare (~3%), and the competition's evaluation
   metric penalizes models whose performance drifts over time.

This project tackles both by:

- Building type-aware Polars aggregators for depth-1 and depth-2 tables (mean / min / max /
  mode / n_unique / any / count), separating the primary applicant record (`num_group1 == 0`)
  from associated contacts.
- Training a GPU XGBoost classifier with `scale_pos_weight`, L1/L2 regularization, and early
  stopping on validation AUC, followed by threshold tuning to trade off recall on the minority
  class.

## Pipeline

```
┌──────────────┐   ┌────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌────────────────┐
│  Kaggle API  │──▶│  DVC + DagsHub │──▶│  Polars Aggregation  │──▶│  XGBoost (CUDA)  │──▶│  MLflow        │
│  (3.14 GB)   │   │  (versioned)   │   │  Depth 0 / 1 / 2     │   │  + early stop    │   │  (DagsHub)     │
└──────────────┘   └────────────────┘   └──────────────────────┘   └──────────────────┘   └────────────────┘
                          │                                                   │
                          ▼                                                   ▼
                   train_data.parquet                                  ROC-AUC, F1, CM
                   test_data.parquet
```

## Project Structure

```
Loan-Default-Prediction-Risk-Analysis/
├── 1_Data_Exploratoin.ipynb        # EDA: shapes, dtypes, duplicates, null %, column lists
├── 2_Feature_Engineering_1.ipynb   # Merge split shards (static, credit_bureau_a, applprev)
├── 2_Feature_Engineering_2.ipynb   # Aggregate depth-1/depth-2 tables, build final train/test
├── 3_Training_Experiments.ipynb    # XGBoost training, threshold tuning, MLflow logging
├── Scripts/
│   └── setup_colab.py               # DVC install, Git/DagsHub config, dvc pull
├── data/
│   ├── Merged data/                # Concatenated shards (applprev, credit_bureau, static)
│   ├── Aggregated-data/d1/         # Depth-1 aggregations (train)
│   ├── Aggregated-data/d2/         # Depth-2 aggregations (train)
│   ├── Aggregated-test-data/d1/    # Depth-1 aggregations (test)
│   ├── Aggregated-test-data/d2/    # Depth-2 aggregations (test)
│   ├── train_data.parquet          # Final merged train (1.5M × 1,286)
│   ├── test_data.parquet           # Final merged test (10 × 1,285)
│   ├── train_data_processed.parquet # Post-processed (encoded cats, parsed dates)
│   └── test_data_processed.parquet
└── .dvc/                           # DVC cache & remote config
```

## Data

The dataset comes from the Kaggle competition
[**Home Credit — Credit Risk Model Stability**](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).

- **Size**: ~3.14 GB (parquet files)
- **Tables**: ~20 relational parquet tables grouped into 3 depths
  - **Depth 0** (1 row per `case_id`): `base`, `static_0`, `static_cb_0`
  - **Depth 1** (many rows per `case_id`, indexed by `num_group1`): `applprev_1`,
    `credit_bureau_a_1`, `credit_bureau_b_1`, `person_1`, `tax_registry_{a,b,c}_1`,
    `deposit_1`, `debitcard_1`, `other_1`
  - **Depth 2** (many rows per `(case_id, num_group1)`, indexed by `num_group2`):
    `applprev_2`, `credit_bureau_a_2`, `credit_bureau_b_2`, `person_2`
- **Train rows**: 1,526,659 (after merge: 1,286 columns)
- **Test rows**: 10 (gated test set)
- **Target**: `target` (0 = repaid, 1 = default) — heavily imbalanced (~97% / 3%)

Column-naming convention (from the competition):

| Suffix | Meaning |
|---|---|
| `A` | Numeric amount |
| `P` | Numeric DPD (Day Past Due)/ policy value |
| `D` | Date |
| `M` | masked string |
| `L` / `T` | unspecified  transform|

## Feature Engineering

Implemented in Polars with lazy evaluation (`scan_parquet` → `group_by` → `collect`) for
memory efficiency on multi-gigabyte tables.

### Depth-1 aggregation (`aggregate_data_d1`)

Splits each table into:

1. **Primary applicant** (`num_group1 == 0`) — taken as-is, one row per `case_id`.
2. **Associated contacts** (`num_group1 > 0`) — aggregated per `case_id` with:
   - Numeric: `mean`, `max`, `min`
   - Date: `min` (earliest), `max` (most recent)
   - Boolean: `any`, `sum` (count of trues)
   - Categorical: `n_unique`, `mode`
   - Plus `contacts_count` per `case_id`

### Depth-2 aggregation (`aggregate_data_d2`)

Two-stage aggregation:

1. **Stage 1**: group by `(case_id, num_group1)` aggregating `num_group2` rows.
2. **Stage 2a**: take primary instance (`num_group1 == 0`) per `case_id`.
3. **Stage 2b**: aggregate other instances (`num_group1 > 0`) per `case_id`.

### Final merge

All 16 aggregated tables are prefixed (`static_`, `cb_a1_`, `tax_a_`, …) and left-joined onto
`base_train` / `base_test` on `case_id`.

Post-processing:

- Parse object / date columns → datetime → extract `year`, `month`, `day`, `day_of_week`,
  `is_weekend`, `days_since_start`.
- Ordinal-encode categorical columns (with `unknown_value=-1` for unseen categories).
- Cast booleans → `int8` for XGBoost.

## Model & Training

**Algorithm**: XGBoost (`XGBClassifier`) on GPU (`device='cuda'`, `tree_method='hist'`).

**Split**: 60% train / 20% validation / 20% test (via two sequential `train_test_split` calls
with `random_state=2`).

**Hyperparameters** (full `params` dict in notebook 3):

| Parameter | Value |
|---|---|
| `objective` | `binary:logistic` |
| `eval_metric` | `auc` |
| `max_depth` | 6 |
| `learning_rate` | 0.04 |
| `n_estimators` | 3500 |
| `early_stopping_rounds` | 150 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `reg_lambda` (L2) | 10 |
| `reg_alpha` (L1) | 1 |
| `gamma` | 0.1 |
| `min_child_weight` | 20 |
| `scale_pos_weight` | `neg_count / pos_count` (auto-computed) |

Class imbalance is handled via `scale_pos_weight` (ratio of negative to positive labels) rather
than SMOTE — tree-based learners generally benefit more from reweighting than from synthetic
oversampling, and reweighting keeps training probability-calibration control with the
decision-threshold tuning step.

## Results

**Best iteration**: 606 (early stopped at 756 rounds on validation AUC).

| Metric | Value |
|---|---|
| Best validation ROC-AUC | **0.8565** |
| Test ROC-AUC (threshold 0.68) | **0.8568** |
| Accuracy (default threshold 0.5) | 0.8030 |
| Accuracy (tuned threshold 0.68) | 0.9055 |
| Precision (class 1, tuned) | 0.16 |
| Recall (class 1, tuned) | 0.50 |
| Weighted F1 (tuned) | 0.93 |

The threshold of 0.68 (tuned on the validation set) trades a small drop in precision on the
minority class for a substantial gain in recall — important in credit risk, where catching
defaults typically matters more than flagging a few extra non-defaults for manual review.

## Experiment Tracking

Experiments are tracked with **MLflow**, hosted on **DagsHub** (the same place DVC storage
lives). Each run logs:

- Full `params` dict (XGBoost hyperparameters)
- Metrics: `accuracy`, `f1_score`, `roc_auc`
- Tag: `github_commit` (current Git HEAD)
- Model artifact: the trained `XGBClassifier` (sklearn-flavored)

DagsHub + MLflow are initialized at the top of notebook 3:

```python
dagshub.init(
    repo_owner='hmelbanna100',
    repo_name='Loan-Default-Prediction-Risk-Analysis',
    mlflow=True
)
```

View runs at:
`https://dagshub.com/hmelbanna100/Loan-Default-Prediction-Risk-Analysis.mlflow`

The `DAGSHUB_TOKEN` environment variable must be set before running `setup_colab.py` — the
script reads it to configure DVC remote authentication (basic auth, user `hmelbanna100`).

## Setup

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU + CUDA** (for `device='cuda'` in XGBoost; falls back to CPU if you change the
  param)
- **Kaggle account** with API token (the data is gated by the competition)
- **DagsHub account** with a personal access token (free)
- **GitHub** account with a repo created

### Secrets (Google Colab `userdata`)

Store these in Colab's secret manager (key icon on the left sidebar) — or as environment
variables if running locally:

| Secret name | Used for |
|---|---|
| `kaggle_token` | Kaggle API auth (download competition data) |
| `dagshub_token` | DagsHub DVC remote + MLflow tracking |
| `github_token` | Push commits back to the GitHub repo |

`DAGSHUB_TOKEN` is read from the environment by `setup_colab.py`:

```bash
export DAGSHUB_TOKEN=your_dagshub_token
```

### Install

The notebooks are designed to run on **Google Colab** (A100 GPU recommended). The first cell
of each notebook will:

1. Clone the GitHub repo into `/content/`.
2. Run `Scripts/setup_colab.py`, which installs DVC, configures Git/DagsHub credentials, and
   runs `dvc pull` to fetch all parquet data from DagsHub DVC storage.

To run locally instead:

```bash
git clone https://github.com/Hassan-ElBanna/Loan-Default-Prediction-Risk-Analysis.git
cd Loan-Default-Prediction-Risk-Analysis
pip install dvc dagshub mlflow polars xgboost scikit-learn imbalanced-learn \
            pandas numpy matplotlib seaborn kagglehub
export DAGSHUB_TOKEN=your_dagshub_token
dvc pull
```

## Usage

Run the notebooks in order:

| # | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | `1_Data_Exploratoin.ipynb` | EDA — shapes, dtypes, duplicates, null %, column lists for all 20 tables | — |
| 2 | `2_Feature_Engineering_1.ipynb` | Merge split shards into one table per type (static, credit_bureau_a_1, applprev_1, credit_bureau_a_2) | `data/Merged data/*.parquet` |
| 3 | `2_Feature_Engineering_2.ipynb` | Aggregate depth-1 & depth-2 tables, join everything onto `base_train` / `base_test` | `data/train_data.parquet`, `data/test_data.parquet`, `data/Aggregated-data/**` |
| 4 | `3_Training_Experiments.ipynb` | Train XGBoost, tune threshold, log to MLflow | Trained model + MLflow run URL |

## Future Work

- **Ensemble** XGBoost with LightGBM and CatBoost (both support categorical features natively,
  may capture different patterns from the gradient-boosted trees).
- **Cross-validation by `WEEK_NUM`** instead of random split — this matches the actual
  competition metric (stability over time), which penalizes temporal drift.
- **Hyperparameter optimization** with Optuna or Hyperopt instead of a manual grid.
- **Feature selection** — the merged train set has 1,286 features, many sparse / high-null.
  Boruta or recursive feature elimination could reduce noise and training time.
- **SHAP analysis** for model interpretability — important for credit-risk decisions, where
  regulators often require explanations for adverse action.
- **Calibrated probabilities** — `scale_pos_weight` distorts raw probabilities; apply
  `CalibratedClassifierCV` (isotonic / sigmoid) for threshold-stable probabilities.
- **Submission generation** — write the test-set prediction script that produces a
  `submission.csv` matching the Kaggle format.

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

The dataset itself is governed by the
[Kaggle competition rules](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/rules)
and is **not** redistributed in this repository — you must download it yourself via
`kagglehub` after accepting the competition rules on Kaggle.

## Author

**Hassan El-Banna**

- GitHub: [Hassan-ElBanna](https://github.com/Hassan-ElBanna)
- DagsHub: [hmelbanna100](https://dagshub.com/hmelbanna100)
- Email: `hmelbanna100@gmail.com`

<!-- TODO: replace the line below with your real LinkedIn profile URL -->
<!-- LinkedIn: [Hassan El-Banna](https://www.linkedin.com/in/your-profile) -->

---

*This project was built as part of coursework at MUST (Misr University for Science and
Technology).*
