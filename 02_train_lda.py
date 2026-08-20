import pandas as pd
from gensim import corpora, models
from gensim.models import Phrases
import re

if __name__ == '__main__':
    print("Loading processed news...")
    df = pd.read_csv('processed_news.csv').dropna()

    # 1. Convert text back to token lists
    data_words = [str(text).split() for text in df['processed_content']]

    # 2. BIGRAM MODELING: Capturing phrase dependencies (e.g., 'federal_reserve')
    # This is key for making topics 'Stronger'
    bigram = Phrases(data_words, min_count=5, threshold=10)
    data_bigrams = [bigram[doc] for doc in data_words]

    # 3. DICTIONARY & SELECTIVE FILTERING
    id2word = corpora.Dictionary(data_bigrams)

    # FORCE DEPENDENCE:
    # no_above=0.4: If a word is in 40%+ of news, it's 'Market Noise'. Delete it.
    # no_below=10: If a word is too rare, it's an outlier. Delete it.
    id2word.filter_extremes(no_below=10, no_above=0.4)

    corpus = [id2word.doc2bow(text) for text in data_bigrams]

    # 4. TRAIN LDA
    num_topics = 10
    print(f"Training LDA with {num_topics} topics...")
    lda_model = models.LdaMulticore(corpus=corpus, id2word=id2word,
                                    num_topics=num_topics, passes=20, workers=4)

    # 5. OUTPUT TOP 10 WORDS FOR NAMING
    print("\n" + "=" * 40)
    print("DISCOVERED TOPICS (Top 10 Words)")
    print("=" * 40)
    for idx, topic in lda_model.print_topics(-1, num_words=10):
        print(f"Topic {idx}: {topic}\n")


    # 6. SAVE TOPIC WEIGHTS
    def get_dist(text_list):
        bow = id2word.doc2bow(bigram[text_list])
        dist = lda_model.get_document_topics(bow, minimum_probability=0)
        return [v for _, v in dist]


    topic_cols = [f'Topic_{i}' for i in range(num_topics)]
    df[topic_cols] = pd.DataFrame(df['processed_content'].str.split().apply(get_dist).tolist(), index=df.index)

    df.to_csv('news_with_topics.csv', index=False)
    print("Topics saved to news_with_topics.csv")