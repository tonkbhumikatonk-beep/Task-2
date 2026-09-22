# Credit Card Fraud Detection Using Machine Learning

A beginner-friendly supervised-learning project aligned with AI Project 2:
- Load and understand a dataset
- Split data into training and testing sets
- Apply a classification algorithm
- Train, test, validate, and evaluate the model

## Project
Predict whether a credit-card transaction is:
- 0 = Genuine
- 1 = Fraudulent

## Dataset
This project uses a synthetic transaction dataset generated locally by `generate_dataset.py`.
This avoids requiring an external download and makes the project reproducible.

## Features
- amount
- transaction_hour
- distance_from_home_km
- distance_from_last_transaction_km
- online_transaction
- international_transaction
- merchant_risk_score
- card_present
- transaction_velocity
- account_age_days

Target:
- fraud

## Algorithms
The main model is Random Forest, a strong classification algorithm for tabular data.
The project also includes Logistic Regression as a simple baseline.

## Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python generate_dataset.py
```

### 3. Train and evaluate
```bash
python train.py
```

### 4. Predict a new transaction
```bash
python predict.py
```

The training script creates:
- models/random_forest.joblib
- models/scaler.joblib
- reports/metrics.json
- reports/confusion_matrix.png
- reports/feature_importance.png

## Important
This is an educational demonstration, not a real banking/fraud-prevention system.
Synthetic data is used, so reported accuracy should not be treated as real-world performance.
