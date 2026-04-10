import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("Loading data for Weighted 10-Year HMM...")
    df = pd.read_csv('hmm_training_data.csv')

    # 1. APPLY WEIGHTS
    # This intensifies the signal for high-importance news events
    topic_weights = {
        'Topic_0': 1.5, 'Topic_1': 2.5, 'Topic_2': 0.5, 'Topic_3': 1.2,
        'Topic_4': 0.5, 'Topic_5': 0.8, 'Topic_6': 2.5, 'Topic_7': 1.0,
        'Topic_8': 0.5, 'Topic_9': 1.8
    }

    topic_cols = [col for col in df.columns if 'Topic_' in col]
    for topic, weight in topic_weights.items():
        if topic in df.columns:
            df[topic] = df[topic] * weight

    # 2. PREPARE OBSERVATION MATRIX
    # We include 'Return' and all 10 weighted topics
    feature_cols = ['Return'] + topic_cols
    X = df[feature_cols].values

    # 3. STANDARDIZE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. TRAIN GAUSSIAN HMM
    # We use 3 regimes: Bearish (High Vol), Bullish (Steady), and Neutral
    num_regimes = 3
    print(f"Training HMM with 10 Topics and {num_regimes} states...")

    # Use 'diag' covariance to handle the 11-dimensional feature space safely
    hmm_model = GaussianHMM(n_components=num_regimes, covariance_type="diag",
                            n_iter=1000, random_state=42)
    hmm_model.fit(X_scaled)

    # 5. DECODE & SAVE
    df['Market_Regime'] = hmm_model.predict(X_scaled)
    df.to_csv('weighted_regime_results.csv', index=False)

    print("\n--- TRANSITION MATRIX ---")
    print(hmm_model.transmat_.round(3))
    print("\nRegime decoding complete.")