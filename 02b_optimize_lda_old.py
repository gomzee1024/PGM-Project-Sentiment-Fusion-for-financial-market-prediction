import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


def compute_coherence_values(dictionary, corpus, texts, limit, start=5, step=5):
    """
    Trains multiple LDA models and computes the C_v coherence score for each.
    """
    coherence_values = []
    model_list = []

    print(f"Testing LDA models from K={start} to K={limit}...")
    for num_topics in range(start, limit + 1, step):
        print(f"  Training model with {num_topics} topics...")

        # Train the model
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            workers=4,
            passes=5,  # Reduced passes to speed up the search
            random_state=42,
            alpha='asymmetric'
        )
        model_list.append(model)

        # Calculate Coherence
        coherencemodel = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        score = coherencemodel.get_coherence()
        coherence_values.append(score)
        print(f"  -> Coherence Score: {score:.4f}")

    return model_list, coherence_values


if __name__ == '__main__':
    # 1. Load the preprocessed data from Phase 1
    print("Loading processed news...")
    try:
        df = pd.read_pickle('processed_news.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("Could not find 'processed_news.pkl'. Run Phase 1 first.")

    clean_corpus = df['clean_tokens'].tolist()

    # 2. Build the Dictionary and Bag-of-Words
    print("Building dictionary and BoW...")
    id2word = Dictionary(clean_corpus)
    id2word.filter_extremes(no_below=10, no_above=0.4)
    corpus = [id2word.doc2bow(text) for text in clean_corpus]

    # 3. Define the search grid
    start_k = 2
    limit_k = 30
    step_k = 2

    # 4. Run the optimization
    models, coherence_values = compute_coherence_values(
        dictionary=id2word,
        corpus=corpus,
        texts=clean_corpus,
        start=start_k,
        limit=limit_k,
        step=step_k
    )

    # 5. Plot the results for the academic report
    print("\nGenerating Coherence plot...")
    x = range(start_k, limit_k + 1, step_k)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='#2ca02c', linewidth=2)

    plt.title("LDA Topic Coherence Optimization ($C_v$)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Number of Topics ($K$)", fontsize=12)
    plt.ylabel("Coherence Score ($C_v$)", fontsize=12)

    # Highlight the maximum value
    max_idx = coherence_values.index(max(coherence_values))
    optimal_k = x[max_idx]
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal $K$ = {optimal_k}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lda_coherence_optimization.png', dpi=300)
    print("Optimization complete! Plot saved as 'lda_coherence_optimization.png'.")
    plt.show()

    print(f"\nCONCLUSION: Update '02_train_lda.py' to use num_topics = {optimal_k}")