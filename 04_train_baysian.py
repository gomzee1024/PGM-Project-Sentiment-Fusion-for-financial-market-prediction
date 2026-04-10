import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("Loading data for Directional Causal Inference...")
    # This data was prepared in 03_align_and_filter.py
    df = pd.read_csv('hmm_training_data.csv')

    # 1. APPLY IMPORTANCE WEIGHTS (Saliency Multipliers)
    # These prioritize high-impact narratives in the probability calculation
    topic_weights = {
        'Topic_0': 1.5,  # Macro Labor
        'Topic_1': 2.5,  # Executive Politics
        'Topic_2': 0.5,  # Retail (Noise)
        'Topic_3': 1.2,  # Hospitality
        'Topic_4': 0.5,  # Media (Noise)
        'Topic_5': 0.8,  # Family
        'Topic_6': 2.5,  # Pandemic/Crisis
        'Topic_7': 1.0,  # Legacy Politics
        'Topic_8': 0.5,  # General Noise
        'Topic_9': 1.8  # Legal/Regulatory Risk
    }

    for topic, weight in topic_weights.items():
        if topic in df.columns:
            df[topic] = df[topic] * weight

    # 2. IMPLEMENT CAUSAL LAG (Yesterday's news -> Today's market trend)
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

    # 5. TRAIN
    # Using 'balanced' weights ensures we identify both Up and Down moves
    model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # =================================================================
    # 6. EXTRACT BIDIRECTIONAL CAUSAL MATRIX (RE-INSERTED)
    # =================================================================
    # These are the Beta coefficients for the research paper results
    influence_matrix = pd.DataFrame(
        model.coef_.T,
        index=topic_cols,
        columns=['Down_Weight', 'Neutral_Weight', 'Up_Weight']
    )
    print("\n--- BIDIRECTIONAL CAUSAL INFLUENCE ---")
    print(influence_matrix)
    # This CSV is used by 05_final_viz.py to create your charts
    influence_matrix.to_csv('approach_a_causal_influence.csv')

    # =================================================================
    # 7. EVALUATE WITH NEW CONFIDENCE FILTER (ACCURACY BOOST)
    # =================================================================
    print("\nApplying Confidence Thresholding for M4 Milestone Accuracy...")

    # Get raw probabilities for each class
    probs = model.predict_proba(X_test)
    class_labels = model.classes_  # Typically [-1, 0, 1]

    # ADJUSTABLE PARAMETER: Default to 0.45
    # High Threshold = High Accuracy (stays Neutral unless sure)
    # Low Threshold = High Recall (aggressive trend catching)
    CONFIDENCE_THRESHOLD = 0.45

    y_pred = []
    for prob_array in probs:
        max_prob = np.max(prob_array)
        predicted_label = class_labels[np.argmax(prob_array)]

        # If the strongest signal is below our threshold, default to Neutral (0)
        if max_prob < CONFIDENCE_THRESHOLD:
            y_pred.append(0)
        else:
            y_pred.append(predicted_label)

    y_pred = np.array(y_pred)

    print("\n--- CLASSIFICATION PERFORMANCE ---")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Neutral', 'Up'], zero_division=0))

    # Save Bayesian predictions for use in your final comparative graph
    df_results = pd.DataFrame({
        'Date': df.iloc[split_idx:]['date'].values,
        'Bayesian_Prediction': y_pred
    })
    # This ensures y_pred uses the standardized string names for the visualizer
    label_map = {-1: 'Down', 0: 'Neutral', 1: 'Up'}
    df_results['Bayesian_Prediction'] = df_results['Bayesian_Prediction'].map(label_map)
    df_results.to_csv('bayesian_predictions.csv', index=False)
    print("Saved Bayesian predictions to 'bayesian_predictions.csv'")