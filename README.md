# PGM-Project-Sentiment-Fusion-for-financial-market-prediction

# Sentiment Fusion: A Sequential Probabilistic Graphical Model for Market Regime Detection and Causal Inference

**Author:** Gautam Kakadiya ([gxk384@case.edu](mailto:gxk384@case.edu))  
**Institution:** Case Western Reserve University  

---

## Executive Summary

Traditional quantitative forecasting models (such as ARIMA and deep learning architectures like LSTMs) rely primarily on price-action history. Because they map correlation rather than causality, these models remain inherently reactive—failing during Black Swan events and narrative-driven market panics.

**Sentiment Fusion** is an interpretable, "Glass-Box" Probabilistic Graphical Model (PGM) architecture that fuses **unstructured textual news** with **structured market time-series** across an 11-year dataset (2012–2022). By treating news topics as parent causal nodes in a Directed Acyclic Graph (DAG) and applying graph-theoretic **D-separation**, this framework:
1. Identifies the directional causal push of specific macro narratives ($\beta$ influence coefficients).
2. Overcomes the "Majority Class Trap" via a 0.45 posterior confidence filter, capturing 14% of major downward market crashes before they materialize.
3. Stratifies market volatility into three discrete regimes using a Gaussian Hidden Markov Model (HMM), isolating a high-risk **Crisis State at 23.1% annualized volatility**.

---

## System Architecture

The pipeline processes raw data through four modular probabilistic stages:

[Raw Financial News + URLs]
│
▼
┌───────────────────────────────────────┐
│ 1. Web Scraping & Text Preprocessing  │  ──> Extracts dense full-body article text (trafilatura)
└───────────────────────────────────────┘
│
▼
┌───────────────────────────────────────┐
│ 2. Unsupervised Feature Extraction    │  ──> LDA topic modeling (Low β prior) + Gemini semantic mapping
└───────────────────────────────────────┘
│
▼
┌───────────────────────────────────────┐
│ 3. Approach A: Bayesian Belief Net    │  ──> 5-day smoothed causal lag DAG + 0.45 confidence filter
└───────────────────────────────────────┘
│
▼
┌───────────────────────────────────────┐
│ 4. Approach B: Sequential Gaussian HMM│  ──> D-separation noise pruning + Baum-Welch / Viterbi decoding
└───────────────────────────────────────┘
│
▼
[Decoded Market Regimes: Crisis (23.1%), Neutral (14.5%), Steady Bull (14.1%)]

## Key Methodology

### 1. Dense Textual Evidence Scraping
Standard headlines (8–12 words) lack the textual density required to build robust word co-occurrence matrices. Using `trafilatura`, the pipeline dynamically scrapes full article bodies from URLs in the raw JSON repository, producing rich daily corpora.

### 2. Latent Dirichlet Allocation (LDA) & $\beta$ Prior Tuning
Text is transformed into 10 continuous probability vectors via LDA using Variational Bayes inference:

$$P(W, Z, \theta, \phi \mid \alpha, \beta) = \prod_{k=1}^{K} P(\phi_k \mid \beta)\ \prod_{d=1}^{M} P(\theta_d \mid \alpha)\ \prod_{n=1}^{N_d} P(z_{d,n} \mid \theta_d)\, P(W_{d,n} \mid \phi_{z_{d,n}})$$

A low Dirichlet prior $\beta$ is enforced to prevent a "uniform soup" of words, compelling the model to construct sharp, distinct topic boundaries. Gemini LLM is used as an expert semantic bridge to label the resulting clusters[cite: 2]:

### 3. Approach A: Bayesian Causal Inference
A directed causal lag is enforced: **Text Narrative at Day $t-1$** $\rightarrow$ **Market Trend at Day $t$**[cite: 2]. A 5-day rolling average filters daily noise[cite: 2]. The network learns directional Beta ($\beta$) Causal Influence Weights[cite: 2]:

$$\text{logit}(P(Y=k)) = \beta_0 + \sum_{i=1}^{10} \beta_i \cdot \text{Topic}_i$$[cite: 2]

### 4. Approach B: Gaussian HMM with D-Separation
To eliminate **Feature Drowning** caused by feeding all 10 noisy topics into the HMM, **D-separation** was applied to sever edges from 8 conditionally independent topics[cite: 2]. The HMM was trained exclusively on the strongest causal drivers: **Topic 0 (Labor)** and **Topic 8 (Sentiment)**[cite: 2].
* **Baum-Welch (EM):** Unsupervised parameter and transition estimation[cite: 2].
* **Viterbi Decoding:** Computes the optimal historical state sequence across 2012–2022[cite: 2].

---

# Execution Guide

Run the pipeline scripts sequentially in your terminal:

### Step 1: Text Preprocessing & NLP Pipeline
Cleans raw text, strips boilerplate noise, and executes tokenization and lemmatization.
```bash
python 01_preprocess_nlp.py
```

### Step 2: Unsupervised LDA Topic Modeling
Fits the Latent Dirichlet Allocation model with the tuned $\beta$ Dirichlet prior to extract daily 10-dimensional narrative topic distributions.
```bash
python 02_train_lda.py
```

### Step 3: Feature-Market Alignment & Evidence Smoothing
Aligns the daily topic probability vectors with next-day S&P 500 market returns ($t-1 \rightarrow t$) and applies the 5-day rolling evidence window.
```bash
python 03_align_and_filter.py
```

### Step 4: Train Probabilistic Graphical Models
* **Bayesian Belief Network (Approach A):** Learns directional $\beta$ causal influence weights and applies the 0.45 posterior confidence filter.
  ```bash
  python 04_train_baysian.py
  ```
* **Baseline Gaussian HMM (Approach B - Full 10 Topics):** Runs unpruned Baum-Welch learning and Viterbi state decoding.
  ```bash
  python 04_train_hmm.py
  ```
* **Refined Gaussian HMM (Approach B - D-Separation):** Prunes conditionally independent noise topics, training exclusively on the strongest causal drivers (Topic 0 & Topic 8).
  ```bash
  python 04_train_hmm_d_sep.py
  ```

### Step 5: Generate Final Visualizations
Exports the Beta influence weight charts, volatility stratification bar charts, and the 3-panel comparative time-series graph.
```bash
python 05_final_viz.py
```

### Step 6: Model Evaluation & Metric Reporting
Calculates classification metrics (Precision, Recall, F1-Score), HMM transition probabilities, and annualized volatility spread.
```bash
python 06_evaluate_models.py
```

---