import pandas as pd
import yfinance as yf
import numpy as np

if __name__ == '__main__':
    # 1. UPDATED SETTINGS FOR 11-YEAR RUN
    target_ticker = 'SPY'
    start_date = '2012-01-01'
    end_date = '2022-12-31'  # Ensure this covers your full 2022 JSON data

    print(f"Loading 10-Topic data from news_with_topics.csv...")
    df_news = pd.read_csv('news_with_topics.csv')

    # Dynamically find all 10 topics (Topic_0 to Topic_9)
    topic_cols = [col for col in df_news.columns if col.startswith('Topic_')]

    # 2. DAILY AGGREGATION
    df_news['date'] = pd.to_datetime(df_news['date']).dt.floor('D')
    df_news_daily = df_news.groupby('date')[topic_cols].mean().reset_index()

    # 3. NEW: TOPIC SMOOTHING (Evidence Accumulation)
    # We use a 3-day window to 'densify' the news signal.
    # This prevents the HMM from collapsing when news is sparse.
    print("Applying 3-day evidence smoothing...")
    df_news_daily[topic_cols] = df_news_daily[topic_cols].rolling(window=3, min_periods=1).mean()

    # 4. DOWNLOAD MARKET DATA
    print(f"Downloading {target_ticker} data ({start_date} to {end_date})...")
    stock = yf.download(target_ticker, start=start_date, end=end_date)

    # Robust multi-index cleaning for the latest yfinance versions
    stock = stock.reset_index()
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = [col[0] if col[0] else col[1] for col in stock.columns]

    stock.rename(columns={'Date': 'date'}, inplace=True)
    stock['date'] = pd.to_datetime(stock['date']).dt.floor('D')

    # 5. CALCULATE RETURNS
    stock['Return'] = stock['Close'].pct_change()

    # 6. MERGE NODES
    df_merged = pd.merge(stock[['date', 'Return']], df_news_daily, on='date', how='inner')

    # 7. IMPLEMENT CAUSAL LAG (Yesterday's smoothed news -> Today's return)
    # This creates the Directed Edge in your PGM: [News_t-1] ---> [Price_t]
    df_merged[topic_cols] = df_merged[topic_cols].shift(1)
    df_merged = df_merged.dropna()

    # 8. SAVE FINAL TRAINING DATA
    output_file = 'hmm_training_data.csv'
    df_merged.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Linked {len(df_merged)} days of Causal Evidence.")
    print(f"File saved as {output_file}. You are ready for Phase 4 Training.")