import pandas as pd
import spacy
import os
import concurrent.futures
import trafilatura
from tqdm import tqdm

# 1. GPU INITIALIZATION (Optimized for 4GB VRAM)
try:
    # Use require_gpu() to force CUDA usage; will error if GPU is not setup
    activated = spacy.require_gpu()
    if activated:
        print("✅ NVIDIA GPU Detected & Activated. Processing will be accelerated.")
except Exception as e:
    print(f"⚠️ GPU Activation failed. Falling back to CPU. Error: {e}")

# 2. SETTINGS & YEAR CONTROL
data_dir = 'DataSet/FinancialNews'
start_year = 2012
end_year = 2022

# Load spaCy (Small model is best for 4GB VRAM)
# Disabling parser and NER saves significant memory and speed
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

financial_noise = {
    'company', 'quarter', 'year', 'share', 'percent', 'management',
    'result', 'million', 'billion', 'dollar', 'forward', 'looking',
    'statement', 'include', 'expect', 'financial', 'operation', 'business',
    'period', 'ended', 'say', 'make', 'report', 'market', 'stock', 'price'
}
nlp.Defaults.stop_words.update(financial_noise)


def scrape_article_text(url):
    """Retrieves full article body text using trafilatura"""
    if pd.isna(url) or not str(url).startswith('http'):
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded, include_comments=False)
    except:
        return ""
    return ""


if __name__ == '__main__':
    all_dataframes = []

    # 3. SELECTIVE FILE LOADING & SUMMARY
    print(f"\n--- PHASE 1: DATA INGESTION ({start_year}-{end_year}) ---")
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f'data_{year}.json')
        if os.path.exists(file_path):
            temp_df = pd.read_json(file_path)
            temp_df['year_label'] = year  # Track which year we are in
            all_dataframes.append(temp_df)
            print(f"Loaded {year}: {len(temp_df):,} articles")
        else:
            print(f"Warning: {file_path} not found.")

    if not all_dataframes:
        print("No data found. Check your folder path.")
        exit()

    df = pd.concat(all_dataframes, ignore_index=True)
    total_articles = len(df)

    print("\n" + "=" * 40)
    print(f"PREPROCESSING TASK: {total_articles:,} Articles")
    print("=" * 40 + "\n")

    # 4. ENHANCED SCRAPING WITH THREADS & PROGRESS
    print(f"--- PHASE 2: WEB SCRAPING ARTICLES ---")
    scraped_results = [None] * total_articles

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        # Create a mapping of future to its original index
        future_to_idx = {executor.submit(scrape_article_text, row['link']): i
                         for i, row in df.iterrows()}

        # Progress bar for scraping
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=total_articles, desc="Scraping Web"):
            idx = future_to_idx[future]
            scraped_results[idx] = future.result()

    df['scraped_text'] = scraped_results
    df['content'] = (
            df['headline'].fillna('') + " " +
            df['short_description'].fillna('') + " " +
            df['scraped_text'].fillna('')
    )

    # 5. GPU-ACCELERATED NLP CLEANING (Batch Processing)
    print(f"\n--- PHASE 3: GPU LINGUISTIC CLEANING ---")

    processed_content = []
    texts = df['content'].fillna("").tolist()

    # SAFE_BATCH_SIZE: 64 is stable for 4GB VRAM
    # nlp.pipe processes articles in parallel on the GPU
    for doc in tqdm(nlp.pipe(texts, batch_size=64), total=total_articles, desc="GPU Processing"):
        # Lemmatize, remove stop words, and keep only alphabetic tokens
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        processed_content.append(" ".join(tokens))

    df['processed_content'] = processed_content

    # 6. SAVE FINAL OUTPUT
    output_file = 'processed_news.csv'
    df[['date', 'processed_content']].to_csv(output_file, index=False)
    print(f"\nSUCCESS: {total_articles:,} rows processed and saved to {output_file}")