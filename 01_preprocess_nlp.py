import pandas as pd
import spacy
import os
import concurrent.futures
import trafilatura

# 1. SETTINGS & YEAR CONTROL
data_dir = 'DataSet/FinancialNews'
start_year = 2012
end_year = 2022  # Adjust this range as needed

# Load spaCy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
financial_noise = {
    'company', 'quarter', 'year', 'share', 'percent', 'management',
    'result', 'million', 'billion', 'dollar', 'forward', 'looking',
    'statement', 'include', 'expect', 'financial', 'operation', 'business',
    'period', 'ended', 'say', 'make', 'report', 'market', 'stock', 'price'
}
nlp.Defaults.stop_words.update(financial_noise)


def scrape_article_text(url):
    if pd.isna(url) or not str(url).startswith('http'):
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded, include_comments=False)
    except:
        return ""
    return ""


def preprocess_text(text):
    doc = nlp(str(text).lower())
    # Keep only alphabetic tokens, remove stop words, and lemmatize
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


if __name__ == '__main__':
    all_dataframes = []

    # 2. SELECTIVE FILE LOADING
    print(f"Loading JSON data from {start_year} to {end_year}...")
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f'data_{year}.json')
        if os.path.exists(file_path):
            print(f"Reading {file_path}...")
            # Using lines=False because your example is a standard JSON list
            temp_df = pd.read_json(file_path)
            all_dataframes.append(temp_df)
        else:
            print(f"Warning: {file_path} not found.")

    if not all_dataframes:
        print("No data files found in the specified range. Check your 'data' folder.")
        exit()

    df = pd.concat(all_dataframes, ignore_index=True)

    # 3. CONTENT FUSION (For Stronger Topics)
    print("Scraping article text for enriched evidence...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        df['scraped_text'] = list(executor.map(scrape_article_text, df['link']))

    df['content'] = (
            df['headline'].fillna('') + " " +
            df['short_description'].fillna('') + " " +
            df['scraped_text'].fillna('')
    )

    # 4. CLEANING
    print("Cleaning text for LDA...")
    df['processed_content'] = df['content'].apply(preprocess_text)

    # 5. SAVE AS CSV (Crucial for the new LDA script compatibility)
    output_file = 'processed_news.csv'
    df[['date', 'processed_content']].to_csv(output_file, index=False)
    print(f"Pre-processing complete. {len(df)} rows saved to {output_file}")