# Machine Learning Model for Detecting Cyber Attacks through URL Analysis

## Overview
This repository implements a machine learning pipeline to detect malicious/phishing URLs using URL-based features and classical ML classifiers. The project follows the objectives and implementation details contained in the learning materials (`learning materials/Project objectives and implementation details .docx`).

## Project objectives
The goals are:
- Build a robust classifier to flag phishing/malicious URLs.
- Extract meaningful features from raw URLs (lengths, token counts, special-character statistics, domain features, keyword presence, etc.).
- Train, validate, and evaluate models with clear metrics and reproducible notebooks.

## Dataset
- Source - https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset


## Methodology (high-level)
1. Data loading and cleaning.
2. URL feature extraction (lengths, counts, presence of suspicious tokens, TLD/domain-based signals, n-gram or tokenization features).
3. Feature encoding and scaling.
4. Model training (e.g., Random Forest, XGBoost, Logistic Regression, etc.) with cross-validation and hyperparameter tuning.
5. Evaluation using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices.
6. Save the trained model and inference code for deployment/testing.

## Requirements
Install the typical Python packages used in the notebooks:

pip install -r requirements.txt  # if you create one
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost


## How to run
1. Open the repository in VS Code or Jupyter Notebook/Lab.
2. Install dependencies.
3. Open `url_checked_final.ipynb` and run cells top-to-bottom to reproduce preprocessing, training, and evaluation.
4. Optionally inspect earlier notebooks for experiments and feature-engineering ideas.

## Results & artifacts
Trained models, evaluation output, and plots are produced by the notebooks. If you wish to export a final model, look for joblib/pickle save steps inside `url_checked_final.ipynb`.


# machine-learning-model-for-detecting-cyber-attacks-through-url-analysis-
