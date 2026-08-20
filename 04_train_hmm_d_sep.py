import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("🚀 Running Refined Sequential HMM (D-Separation Phase)...")

    # Load the 11-year aligned dataset
    df = pd.read_csv('hmm_training_data.csv')

    # 1. APPLY SELECTIVE EXPERT WEIGHTS
    # We focus only on the topics with the highest 'Down_Weight'
    # Topic 0: Macro Labor & Policy (Weight 1.5)
    # Topic 8: General Sentiment (Weight 0.5)
    selected_topics = {
        'Topic_0': 1.5,  # Macro Labor
        'Topic_8': 0.5  # General Sentiment
    }

    for topic, weight in selected_topics.items():
        if topic in df.columns:
            df[topic] = df[topic] * weight

    # 2. FEATURE SELECTION (D-SEPARATION)
    # Instead of all 10 topics, we use only the most informative nodes
    # This reduces 'Blurring' in the Gaussian emission distributions
    feature_cols = ['Return', 'Topic_0', 'Topic_8']
    X = df[feature_cols].values

    # 3. STANDARDIZATION
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. TRAIN REFINED GAUSSIAN HMM
    # 3 Regimes: 0=Structural/Bearish, 1=Correction, 2=Steady Bull
    num_regimes = 3
    print(f"Training Refined HMM with 3 States and 2 Top-Tier Topics...")

    # Use 1000 iterations for convergence in the sparse feature space
    hmm_model = GaussianHMM(n_components=num_regimes, covariance_type="diag",
                            n_iter=1000, random_state=42)
    hmm_model.fit(X_scaled)

    # 5. DECODE & EXPORT
    df['Market_Regime'] = hmm_model.predict(X_scaled)

    # Save as a refined results file for the visualization script
    output_csv = 'weighted_regime_results.csv'
    df.to_csv(output_csv, index=False)

    # 6. ANALYZE TRANSITION STABILITY
    print("\n--- REFINED TRANSITION MATRIX ---")
    # High diagonal values indicate stable, narrative-driven regimes
    print(hmm_model.transmat_.round(3))

    print(f"\n✅ Refined decoding complete. Results saved to {output_csv}")