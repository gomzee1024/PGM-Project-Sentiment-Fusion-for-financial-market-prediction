import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("Loading data for Directional Causal Inference...")
    df = pd.read_csv('hmm_training_data.csv')

    # 1. APPLY IMPORTANCE WEIGHTS (Saliency Multipliers)
    # These prioritize high-impact narratives in the probability calculation
    # PRO TIP: Topic 1 and 6 are the strongest 'Regime Shifters' in a 10-year PGM
    topic_weights = {
        'Topic_0': 1.5,  # Macro Labor
        'Topic_1': 2.5,  # Executive Politics (Trump Era)
        'Topic_2': 0.5,  # Retail (Noise)
        'Topic_3': 1.2,  # Hospitality
        'Topic_4': 0.5,  # Media (Noise)
        'Topic_5': 0.8,  # Family
        'Topic_6': 2.5,  # Pandemic/Crisis (High Volatility)
        'Topic_7': 1.0,  # Legacy Politics
        'Topic_8': 0.5,  # General Noise
        'Topic_9': 1.8  # Legal/Biden Regulatory Risk
    }

    for topic, weight in topic_weights.items():
        if topic in df.columns:
            df[topic] = df[topic] * weight

    # 2. CAUSAL LAG: t-1 news influences t return
    topic_cols = [col for col in df.columns if 'Topic_' in col]
    df[topic_cols] = df[topic_cols].shift(1)
    df = df.dropna()

    # 3. DISCRETIZE TREND (Target Node)
    threshold = 0.0075
    df['Trend'] = 0
    df.loc[df['Return'] > threshold, 'Trend'] = 1
    df.loc[df['Return'] < -threshold, 'Trend'] = -1

    X = df[topic_cols].values
    y = df['Trend'].values

    # 4. TEMPORAL SPLIT & SCALING
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. TRAIN: Use 'balanced' weights to identify both Up and Down moves
    model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # 6. EXTRACT BIDIRECTIONAL CAUSAL MATRIX
    # Positive values = Bullish influence | Negative = Bearish influence
    importance = pd.DataFrame(
        model.coef_,
        columns=topic_cols,
        index=['Down_Weight', 'Neutral_Weight', 'Up_Weight']
    ).T

    print("\n--- BIDIRECTIONAL CAUSAL INFLUENCE ---")
    print(importance)

    print("\n--- CLASSIFICATION PERFORMANCE ---")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=['Down', 'Neutral', 'Up'], zero_division=0))

    importance.to_csv('bidirectional_influence_matrix.csv')