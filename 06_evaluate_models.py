import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

if __name__ == '__main__':
    print("Starting Final Temporal Evaluation (80/20 Split)...")
    df = pd.read_csv('hmm_training_data.csv')

    # 1. SETUP TARGETS & FEATURES
    threshold = 0.0075
    df['Trend'] = 0
    df.loc[df['Return'] > threshold, 'Trend'] = 1
    df.loc[df['Return'] < -threshold, 'Trend'] = -1

    topic_cols = [col for col in df.columns if 'Topic_' in col]
    # Apply your Expert Weights (as defined in Phase 4)
    topic_weights = {'Topic_0': 0.5, 'Topic_1': 0.8, 'Topic_2': 1.5, 'Topic_3': 1.2, 'Topic_4': 2.0}
    for t, w in topic_weights.items():
        if t in df.columns:
            df[t] = df[t] * w

    # 2. TEMPORAL SPLIT (No Shuffling)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # ==========================================
    # APPROACH A: CAUSAL CLASSIFICATION
    # ==========================================
    X_train_a = train_df[topic_cols].values
    X_test_a = test_df[topic_cols].values
    y_train_a = train_df['Trend'].values
    y_test_a = test_df['Trend'].values

    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_train_a)
    X_test_a = scaler_a.transform(X_test_a)

    # Fusion Model vs Naive Baseline
    model_a = LogisticRegression(class_weight='balanced', max_iter=1000)
    model_a.fit(X_train_a, y_train_a)

    baseline_a = DummyClassifier(strategy='most_frequent')
    baseline_a.fit(X_train_a, y_train_a)

    a_pred = model_a.predict(X_test_a)
    a_base_pred = baseline_a.predict(X_test_a)

    # ==========================================
    # APPROACH B: SEQUENTIAL REGIME DETECTION
    # ==========================================
    # Feature Sets: Price-Only (Baseline) vs Price+News (Fusion)
    base_feats = ['Return']
    fuse_feats = ['Return'] + topic_cols

    scaler_base = StandardScaler()
    scaler_fuse = StandardScaler()

    X_train_base = scaler_base.fit_transform(train_df[base_feats].values)
    X_test_base = scaler_base.transform(test_df[base_feats].values)

    X_train_fuse = scaler_fuse.fit_transform(train_df[fuse_feats].values)
    X_test_fuse = scaler_fuse.transform(test_df[fuse_feats].values)

    # Train two HMMs to compare Volatility Stratification
    hmm_base = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    hmm_fuse = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)

    hmm_base.fit(X_train_base)
    hmm_fuse.fit(X_train_fuse)

    test_df['Base_Regime'] = hmm_base.predict(X_test_base)
    test_df['Fuse_Regime'] = hmm_fuse.predict(X_test_fuse)


    # Calculate Volatility Spread (Success Metric)
    def get_spread(data, regime_col):
        vols = data.groupby(regime_col)['Return'].std() * np.sqrt(252)
        return vols.max() - vols.min()


    spread_base = get_spread(test_df, 'Base_Regime')
    spread_fuse = get_spread(test_df, 'Fuse_Regime')

    # ==========================================
    # FINAL REPORTING
    # ==========================================
    print("\n" + "=" * 40)
    print("FINAL PGM PERFORMANCE REPORT")
    print("=" * 40)

    print("\n[APPROACH A: CAUSAL CLASSIFICATION]")
    print(f"Baseline Accuracy: {accuracy_score(y_test_a, a_base_pred):.4f}")
    print(f"Fusion Accuracy:   {accuracy_score(y_test_a, a_pred):.4f}")
    print("\nFusion Detailed Metrics:")
    print(classification_report(y_test_a, a_pred, target_names=['Down', 'Neutral', 'Up'], zero_division=0))

    print("\n[APPROACH B: SEQUENTIAL REGIME DETECTION]")
    print(f"Baseline Volatility Spread: {spread_base:.4f}")
    print(f"Fusion Volatility Spread:   {spread_fuse:.4f}")

    improvement = (spread_fuse / spread_base - 1) * 100 if spread_base > 0 else 0
    print(f"\nInformation Gain: {improvement:.2f}% increase in risk stratification.")

    if spread_fuse > spread_base:
        print("CONCLUSION: Sentiment Fusion successfully reduced state uncertainty.")
    else:
        print("CONCLUSION: Baseline remains dominant; topics require higher density.")