# Fraud Detection in Insurance — End-to-End Project

This repository scaffolds an **end-to-end ML workflow** for fraud detection on the popular `insurance_claims.csv` dataset.

> **Important:** The link you shared appears malformed (`https://https://...`) and points to a draft URL that may require login. 
> A commonly used public version of this dataset is here: https://data.mendeley.com/datasets/992mh7dk9y/2
>
> Download `insurance_claims.csv` and place it in `data/insurance_claims.csv` before running.

## What this project shows
- Data loading & initial exploration (Pandas)
- Cleaning & preprocessing (missing values, outliers in `umbrella_limit`, encoding)
- EDA (Matplotlib/Seaborn)
- Feature selection (RFECV)
- Class imbalance handling (SMOTE)
- Models: SVM, RandomForest, and an ensemble VotingClassifier
- Evaluation: precision/recall/F1/ROC-AUC, confusion matrix
- Interpretation: SHAP
- Persistence: save best model with `joblib`

## Quickstart

1. **Create environment & install deps**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Add data**  
   - Place `insurance_claims.csv` at `data/insurance_claims.csv`

3. **Run the notebook**  
   ```bash
   jupyter lab
   ```
   Open `notebooks/01_fraud_insurance_end_to_end.ipynb` and execute cells sequentially.

## Project Structure
```
fraud_insurance_project/
├─ data/
│  └─ insurance_claims.csv   # <-- place dataset here
├─ models/
│  └─ best_model.joblib
├─ notebooks/
│  └─ 01_fraud_insurance_end_to_end.ipynb
├─ reports/
│  └─ figures/               # (matplotlib output if you choose to save)
├─ src/
│  └─ __init__.py
├─ requirements.txt
└─ README.md
```

## Notes & Decisions
- **Outliers:** We include a robust treatment for `umbrella_limit` via winsorization/capping. Adjust thresholds as you see fit.
- **Encoding:** We use `ColumnTransformer` with OneHotEncoder for categoricals and a scaler for numerics.
- **SMOTE:** Applied within an imbalanced-learn `Pipeline` to avoid leakage.
- **RFECV:** Performed on a strong baseline estimator to identify a compact feature subset.
- **Interpretability:** We use SHAP TreeExplainer for tree-based models; for linear/SVM, KernelExplainer can be used (more compute).
- **Reproducibility:** `random_state` fixed where applicable.

Good luck — and make sure your README and executive summary tell a *business story*, not just a modeling story.
