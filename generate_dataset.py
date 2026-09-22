import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

rng = np.random.default_rng(RANDOM_STATE)

amount = np.round(rng.lognormal(mean=3.5, sigma=1.0, size=N), 2)
amount = np.clip(amount, 1, 5000)

hour = rng.integers(0, 24, N)
distance_home = np.round(rng.gamma(shape=1.8, scale=12, size=N), 2)
distance_last = np.round(rng.gamma(shape=1.5, scale=8, size=N), 2)
online = rng.binomial(1, 0.58, N)
international = rng.binomial(1, 0.12, N)
merchant_risk = np.round(rng.beta(2, 5, N), 3)
card_present = rng.binomial(1, 0.62, N)
velocity = rng.poisson(2.0, N)
account_age = np.maximum(30, rng.normal(900, 500, N)).astype(int)

# Synthetic fraud probability: intentionally imbalanced, but with learnable patterns.
night = ((hour <= 5) | (hour >= 23)).astype(int)
logit = (
    -5.0
    + 0.00055 * amount
    + 0.95 * night
    + 0.018 * distance_home
    + 0.012 * distance_last
    + 1.10 * online
    + 1.25 * international
    + 3.0 * merchant_risk
    - 1.15 * card_present
    + 0.32 * velocity
    - 0.00025 * account_age
)
prob = sigmoid(logit)
fraud = rng.binomial(1, prob)

df = pd.DataFrame({
    "amount": amount,
    "transaction_hour": hour,
    "distance_from_home_km": distance_home,
    "distance_from_last_transaction_km": distance_last,
    "online_transaction": online,
    "international_transaction": international,
    "merchant_risk_score": merchant_risk,
    "card_present": card_present,
    "transaction_velocity": velocity,
    "account_age_days": account_age,
    "fraud": fraud
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/credit_card_transactions.csv", index=False)

print("Dataset created:", df.shape)
print("Fraud transactions:", int(df["fraud"].sum()))
print("Fraud rate:", round(df["fraud"].mean() * 100, 2), "%")
