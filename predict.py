import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.joblib"

model = joblib.load(MODEL_PATH)

# Example new transaction.
# Change these values to test different cases.
transaction = pd.DataFrame([{
    "amount": 1850.00,
    "transaction_hour": 2,
    "distance_from_home_km": 120.0,
    "distance_from_last_transaction_km": 90.0,
    "online_transaction": 1,
    "international_transaction": 1,
    "merchant_risk_score": 0.82,
    "card_present": 0,
    "transaction_velocity": 7,
    "account_age_days": 400
}])

prediction = int(model.predict(transaction)[0])
probability = float(model.predict_proba(transaction)[0, 1])

print("Fraud probability:", round(probability * 100, 2), "%")
if prediction == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: GENUINE TRANSACTION")
