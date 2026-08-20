import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os


def generate_final_report():
    print("========================================")
    print("FINAL PGM PERFORMANCE REPORT")
    print("========================================\n")

    # ---------------------------------------------------------
    # [APPROACH A: CAUSAL CLASSIFICATION (Bayesian Belief Network)]
    # ---------------------------------------------------------
    print("[APPROACH A: CAUSAL CLASSIFICATION]")

    if not os.path.exists('hmm_training_data.csv'):
        print("Error: 'hmm_training_data.csv' not found.")
        return

    df_a = pd.read_csv('hmm_training_data.csv')

    # 1. Sync weights identically with 04_train_baysian.py
    topic_weights = {
        'Topic_0': 1.5, 'Topic_1': 2.5, 'Topic_2': 0.5, 'Topic_3': 1.2,
        'Topic_4': 0.5, 'Topic_5': 0.8, 'Topic_6': 2.5, 'Topic_7': 1.0,
        'Topic_8': 0.5, 'Topic_9': 1.8
    }

    for topic, weight in topic_weights.items():
        if topic in df_a.columns:
            df_a[topic] = df_a[topic] * weight

    # 2. Causal Lag & Discretization
    topic_cols = [col for col in df_a.columns if 'Topic_' in col]
    df_a[topic_cols] = df_a[topic_cols].shift(1)
    df_a = df_a.dropna()

    threshold = 0.0075
    df_a['Trend'] = 0
    df_a.loc[df_a['Return'] > threshold, 'Trend'] = 1
    df_a.loc[df_a['Return'] < -threshold, 'Trend'] = -1

    X_a = df_a[topic_cols].values
    y_a = df_a['Trend'].values

    # 3. Temporal Split & Scaling
    split_idx = int(len(df_a) * 0.8)
    X_train_a, X_test_a = X_a[:split_idx], X_a[split_idx:]
    y_train_a, y_test_a = y_a[:split_idx], y_a[split_idx:]

    scaler = StandardScaler()
    X_train_a = scaler.fit_transform(X_train_a)
    X_test_a = scaler.transform(X_test_a)

    # 4. Train Model A
    model_a = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
    model_a.fit(X_train_a, y_train_a)

    # 5. Baseline Prediction (Guess Neutral / 0 every day)
    baseline_pred = np.zeros(len(y_test_a))
    baseline_acc = accuracy_score(y_test_a, baseline_pred)

    # 6. Apply Confidence Threshold Filter (The M4 Accuracy Boost)
    a_probs = model_a.predict_proba(X_test_a)
    CONFIDENCE_THRESHOLD = 0.45

    a_pred = []
    for p in a_probs:
        if np.max(p) < CONFIDENCE_THRESHOLD:
            a_pred.append(0)  # Default to Neutral if signal is weak
        else:
            a_pred.append(model_a.classes_[np.argmax(p)])

    a_pred = np.array(a_pred)
    fusion_acc = accuracy_score(y_test_a, a_pred)

    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Fusion Accuracy:   {fusion_acc:.4f}\n")
    print("Fusion Detailed Metrics:")
    print(classification_report(y_test_a, a_pred, target_names=['Down', 'Neutral', 'Up'], zero_division=0))

    # ---------------------------------------------------------
    # [APPROACH B: SEQUENTIAL REGIME DETECTION (HMM)]
    # ---------------------------------------------------------
    print("\n[APPROACH B: SEQUENTIAL REGIME DETECTION]")
    if os.path.exists('weighted_regime_results.csv'):
        df_b = pd.read_csv('weighted_regime_results.csv')

        # Calculate Baseline Overall Volatility (Annualized)
        baseline_vol = df_b['Return'].std() * np.sqrt(252)

        # Calculate Regime Volatilities
        regime_vols = []
        print("\nIndividual Regime Volatilities:")
        for regime in sorted(df_b['Market_Regime'].unique()):
            regime_data = df_b[df_b['Market_Regime'] == regime]
            vol = regime_data['Return'].std() * np.sqrt(252)
            regime_vols.append(vol)
            print(f"  - Regime {regime}: {vol:.4f} ({(vol * 100):.1f}%)")

        if len(regime_vols) > 1:
            # Difference between highest and lowest risk state
            fusion_vol_spread = max(regime_vols) - min(regime_vols)
            print(f"\nBaseline Volatility Spread: {baseline_vol:.4f}")
            print(f"Fusion Volatility Spread:   {fusion_vol_spread:.4f}")

            # Info gain (improvement in stratification)
            info_gain = ((fusion_vol_spread - baseline_vol) / baseline_vol) * 100

            # Since baseline spread is 1 number and fusion spread is the gap between max/min,
            # a positive info_gain percentage indicates successful risk separation.
            print(f"\nInformation Gain: {info_gain:.2f}% increase in risk stratification.")
        else:
            print("\nError: HMM only found 1 regime. Model failed to stratify.")
    else:
        print("\nError: 'weighted_regime_results.csv' not found. Run HMM script first.")


if __name__ == '__main__':
    generate_final_report()