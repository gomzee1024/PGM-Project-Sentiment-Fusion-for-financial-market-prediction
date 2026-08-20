import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- GLOBAL COLOR PALETTE FOR CONSISTENCY ---
CRISIS_COLOR = '#e74c3c'  # Red (Regime 0 / Down / High Volatility)
NEUTRAL_COLOR = '#2980b9'  # Blue (Regime 1 / Neutral)
BULL_COLOR = '#27ae60'  # Green (Regime 2 / Up / Steady Growth)


def visualize_causal_influence(csv_file):
    """
    Visualizes the Influence Matrix from the Bayesian Classifier.
    """
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Run 04_train_baysian.py first.")
        return

    df_inf = pd.read_csv(csv_file, index_col=0)
    plot_cols = [c for c in ['Down_Weight', 'Up_Weight'] if c in df_inf.columns]

    plt.figure(figsize=(12, 6))
    # Apply standard colors: Red for Down, Green for Up
    df_inf[plot_cols].plot(kind='bar', ax=plt.gca(), color=[CRISIS_COLOR, BULL_COLOR])

    plt.title("Approach A: Bidirectional Causal Influence", fontsize=14)
    plt.ylabel("Influence Strength (Beta Coefficient)")
    plt.xlabel("Latent News Topics")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Market Direction")
    plt.tight_layout()
    plt.savefig('approach_a_causal_influence.png', dpi=300)
    print("Generated: approach_a_causal_influence.png")


def visualize_regime_separation(results_csv):
    """
    Visualizes the Volatility Spread from the HMM.
    """
    if not os.path.exists(results_csv):
        print(f"Warning: {results_csv} not found. Run 04_train_hmm_refined.py first.")
        return

    df = pd.read_csv(results_csv)
    vols = df.groupby('Market_Regime')['Return'].std() * np.sqrt(252)

    plt.figure(figsize=(10, 5))

    # Apply standard colors: 0=Red, 1=Blue, 2=Green
    colors = [CRISIS_COLOR, NEUTRAL_COLOR, BULL_COLOR]

    vols.plot(kind='bar', color=colors[:len(vols)])
    plt.title("Approach B: Regime Volatility Stratification", fontsize=14)
    plt.ylabel("Annualized Volatility (%)")
    plt.xlabel("Hidden Market State (Regime)")
    plt.xticks(rotation=0)

    for i, v in enumerate(vols):
        plt.text(i, v + 0.005, f"{v:.1%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('approach_b_volatility_spread.png', dpi=300)
    print("Generated: approach_b_volatility_spread.png")


def generate_comparative_plot():
    """
    Generates a 3-panel time-series chart comparing signals against the S&P 500 curve.
    """
    hmm_file = 'weighted_regime_results.csv'
    bayesian_file = 'bayesian_predictions.csv'

    if not os.path.exists(hmm_file):
        print(f"Warning: '{hmm_file}' not found. Cannot generate comparative plot.")
        return

    df = pd.read_csv(hmm_file)
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if os.path.exists(bayesian_file):
        df_bayes = pd.read_csv(bayesian_file)
        if 'date' in df_bayes.columns:
            df_bayes = df_bayes.rename(columns={'date': 'Date'})
        df_bayes['Date'] = pd.to_datetime(df_bayes['Date'])
        df = pd.merge(df, df_bayes[['Date', 'Bayesian_Prediction']], on='Date', how='left')
    else:
        df['Bayesian_Prediction'] = 'Neutral'

    if 'Close' not in df.columns and 'Return' in df.columns:
        df['Close'] = 1000 * (1 + df['Return']).cumprod()

    # Map signals: -1 (Red), 0 (Blue), 1 (Green)
    df['Signal_Baseline'] = 0
    bayes_map = {'Down': -1, 'Neutral': 0, 'Up': 1}
    df['Signal_A'] = df['Bayesian_Prediction'].map(bayes_map).fillna(0)
    hmm_map = {0: -1, 1: 0, 2: 1}
    df['Signal_B'] = df['Market_Regime'].map(hmm_map).fillna(0)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle('S&P 500 Sentiment Fusion: Model Comparison (2012-2022)', fontsize=18, fontweight='bold')

    models = [
        ('Baseline (Price-Only Neutral Bias)', 'Signal_Baseline', axes[0]),
        ('Approach A (Bayesian Belief Network)', 'Signal_A', axes[1]),
        ('Approach B (D-Separation HMM Regimes)', 'Signal_B', axes[2])
    ]

    for title, signal_col, ax in models:
        # Plot underlying S&P 500 line
        ax.plot(df['Date'], df['Close'], color='black', alpha=0.6, linewidth=1.5, label='S&P 500 Index')

        # Filter data by signal state
        bull_data = df[df[signal_col] == 1]
        bear_data = df[df[signal_col] == -1]
        neutral_data = df[df[signal_col] == 0]

        # Plot Neutral (Blue dots)
        ax.scatter(neutral_data['Date'], neutral_data['Close'], color=NEUTRAL_COLOR, marker='.', s=20, alpha=0.4,
                   label='Neutral / Normal', zorder=2)
        # Plot Bull (Green triangles)
        ax.scatter(bull_data['Date'], bull_data['Close'], color=BULL_COLOR, marker='^', s=55,
                   label='Bullish / Steady Growth', zorder=3)
        # Plot Bear (Red inverted triangles)
        ax.scatter(bear_data['Date'], bear_data['Close'], color=CRISIS_COLOR, marker='v', s=55,
                   label='Bearish / Crisis State', zorder=3)

        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Index Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add legend only to the top plot
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=11)

    axes[2].set_xlabel('Year', fontsize=14)
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig('compare_plot_1.png', dpi=300, bbox_inches='tight')
    print("Generated: compare_plot_1.png")


if __name__ == '__main__':
    print("Starting Phase 5: Result Visualization...")
    visualize_causal_influence('approach_a_causal_influence.csv')
    visualize_regime_separation('weighted_regime_results.csv')
    generate_comparative_plot()
    print("\nVisualization complete. Check your directory for PNG files.")