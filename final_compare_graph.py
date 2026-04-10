import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- GLOBAL COLOR PALETTE FOR CONSISTENCY ---
CRISIS_COLOR = '#e74c3c'  # Red (Regime 1 / Down / High Volatility)
NEUTRAL_COLOR = '#2980b9'  # Blue (Regime 0 / Neutral)
BULL_COLOR = '#27ae60'  # Green (Regime 2 / Up / Steady Growth)


def generate_comparative_plot():
    print("Loading project CSVs...")

    # 1. LOAD APPROACH B (HMM REGIMES)
    hmm_file = 'weighted_regime_results.csv'
    if not os.path.exists(hmm_file):
        print(f"Error: '{hmm_file}' not found in the current directory.")
        return

    df = pd.read_csv(hmm_file)

    # --- FIX FOR THE KEYERROR ---
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    elif 'DATE' in df.columns:
        df = df.rename(columns={'DATE': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. LOAD APPROACH A (BAYESIAN) AND BASELINE
    bayesian_file = 'bayesian_predictions.csv'

    if os.path.exists(bayesian_file):
        df_bayes = pd.read_csv(bayesian_file)
        if 'date' in df_bayes.columns:
            df_bayes = df_bayes.rename(columns={'date': 'Date'})
        df_bayes['Date'] = pd.to_datetime(df_bayes['Date'])

        # Merge on Date
        df = pd.merge(df, df_bayes[['Date', 'Bayesian_Prediction']], on='Date', how='left')
    else:
        print(f"Warning: '{bayesian_file}' not found. Defaulting to Neutral.")
        df['Bayesian_Prediction'] = 'Neutral'

    # 3. RECONSTRUCT S&P 500 PRICE
    if 'Close' not in df.columns and 'Return' in df.columns:
        print("Reconstructing S&P 500 price curve from daily returns...")
        df['Close'] = 1000 * (1 + df['Return']).cumprod()
    elif 'Close' not in df.columns:
        print("Error: Neither 'Close' nor 'Return' found in dataset.")
        return

    # 4. MAP YOUR SPECIFIC PROJECT LOGIC TO SIGNALS (-1, 0, 1)

    # Baseline: 53.7% accurate by guessing Neutral
    df['Signal_Baseline'] = 0

    # Approach A: Bayesian Directional Causal Inference
    bayes_map = {'Down': -1, 'Neutral': 0, 'Up': 1}
    df['Signal_A'] = df['Bayesian_Prediction'].map(bayes_map).fillna(0)

    # Approach B: D-Separation HMM Regimes (UPDATED FOR M4)
    # Regime 1 = Crisis (23.1% Vol) -> Bear Signal (-1)
    # Regime 0 = Neutral (14.5% Vol) -> Neutral Signal (0)
    # Regime 2 = Bull (14.1% Vol) -> Bull Signal (1)
    hmm_map = {1: -1, 0: 0, 2: 1}
    df['Signal_B'] = df['Market_Regime'].map(hmm_map).fillna(0)

    # 5. SET UP THE 3-PANEL VISUALIZATION
    print("Generating stacked visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle('S&P 500 Sentiment Fusion: Model Comparison (2012-2022)', fontsize=18, fontweight='bold')

    models = [
        ('Baseline (Price-Only Neutral Bias)', 'Signal_Baseline', axes[0]),
        ('Approach A (Bayesian Belief Network)', 'Signal_A', axes[1]),
        ('Approach B (D-Separation HMM Regimes)', 'Signal_B', axes[2])
    ]

    for title, signal_col, ax in models:
        # Plot the underlying S&P 500 curve
        ax.plot(df['Date'], df['Close'], color='black', alpha=0.6, linewidth=1.5, label='S&P 500 Index')

        # Filter the signals
        bull_data = df[df[signal_col] == 1]
        bear_data = df[df[signal_col] == -1]
        neutral_data = df[df[signal_col] == 0]

        # Plot Neutral (Blue dots)
        ax.scatter(neutral_data['Date'], neutral_data['Close'],
                   color=NEUTRAL_COLOR, marker='.', s=20, alpha=0.4, label='Neutral / Normal', zorder=2)

        # Plot Bull Signals (Green Up-Triangles)
        ax.scatter(bull_data['Date'], bull_data['Close'],
                   color=BULL_COLOR, marker='^', s=55, label='Bullish / Steady Growth', zorder=3)

        # Plot Bear Signals (Red Down-Triangles)
        ax.scatter(bear_data['Date'], bear_data['Close'],
                   color=CRISIS_COLOR, marker='v', s=55, label='Bearish / Crisis State', zorder=3)

        # Formatting
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Index Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add legend to the first plot
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=11)

    # 6. X-AXIS FORMATTING (Years)
    axes[2].set_xlabel('Year', fontsize=14)
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 7. EXPORT
    output_img = 'final_model_comparison.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Visualization saved to {output_img}")


if __name__ == '__main__':
    generate_comparative_plot()