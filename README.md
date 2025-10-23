# Fraud Detection in Insurance

This repository builds an end-to-end ML workflow for fraud detection on the popular `insurance_claims.csv` dataset.


## What this project shows
- Data loading & initial exploration (Pandas)
- Cleaning & preprocessing (missing values, outliers in `umbrella_limit`, encoding)
- EDA (Matplotlib/Seaborn)
- Feature selection (RFECV)
- Class imbalance handling (SMOTE)
- Models: SVM, RandomForest, and an ensemble VotingClassifier
- Evaluation: precision/recall/F1/ROC-AUC, confusion matrix
- Interpretation: SHAP
- save best model with `joblib`

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
│  └─ insurance_claims.csv   
├─ models/
│  └─ best_model.joblib
├─ notebooks/
│  └─ 01_fraud_insurance_end_to_end.ipynb
├─ reports/
│  └─ figures/               
├─ src/
│  └─ __init__.py
├─ requirements.txt
└─ README.md
```

## Notes & Decisions
- **Outliers:** include a robust treatment for `umbrella_limit` via winsorization/capping. Adjust thresholds as see fit.
- **Encoding:** `ColumnTransformer` with OneHotEncoder for categoricals and a scaler for numerics.
- **SMOTE:** Applied within an imbalanced-learn `Pipeline` to avoid leakage.
- **RFECV:** Performed on a strong baseline estimator to identify a compact feature subset.
- **Interpretability:** SHAP TreeExplainer; for linear/SVM, KernelExplainer used
- **Reproducibility:** `random_state` fixed where applicable.
