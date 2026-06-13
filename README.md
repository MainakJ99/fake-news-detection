# Fake News Detection — A Comparative Study of Text Representations

A controlled comparison of four classic NLP text representations for classifying news
articles as **real** or **fake**, holding the downstream classifier fixed so that performance
differences reflect the *representation* rather than the model.

| Representation | Type | Captures |
|---|---|---|
| CountVectorizer | Sparse, frequency | Word / bigram counts |
| TF-IDF | Sparse, weighted | Counts down-weighted by document frequency |
| Word2Vec (mean-pooled) | Dense, 100-d | Distributional word semantics, averaged |
| Doc2Vec | Dense, 100-d | One learned vector per document |

## Key finding: the dataset has source leakage

The "real" articles in this dataset are Reuters wire copy and almost all begin with a dateline
such as `WASHINGTON (Reuters) -`; the fake articles never contain `(Reuters)`. A model can
therefore reach ~99.5% accuracy by memorising a single token instead of learning anything about
misinformation. This project **diagnoses** the leak, **strips** it (dateline, stray `(Reuters)`
mentions, URLs), de-duplicates, and reports the honest numbers below.

## Methodology

- **Single tokenizer** feeds every representation — sparse models consume the joined string,
  embedding models consume the token list — so the input is identical and only the representation
  changes.
- **One stratified 80/20 split**, shared across all four representations.
- **Classifier held fixed:** Logistic Regression, tuned over the same grid of `C` values via 5-fold CV.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC.

## Results (after removing source leakage)

| Representation | Accuracy | Precision | Recall | F1 | ROC-AUC | Best C |
|---|---|---|---|---|---|---|
| **TF-IDF** | 0.9827 | 0.9783 | 0.9901 | **0.9842** | **0.9985** | 10 |
| **CountVectorizer** | 0.9821 | 0.9781 | 0.9891 | 0.9836 | 0.9961 | 0.1 |
| **Doc2Vec** | 0.9633 | 0.9680 | 0.9641 | 0.9661 | 0.9930 | 0.01 |
| **Word2Vec** | 0.9616 | 0.9590 | 0.9707 | 0.9648 | 0.9928 | 0.1 |

### Takeaways

- **Sparse n-gram features win.** TF-IDF edges CountVectorizer, and both beat the dense
  embeddings by ~1.8 F1 points — once source artifacts are stripped, separating fake from real
  here is still largely a matter of vocabulary and phrasing, which sparse models preserve and
  mean-pooling washes out.
- **Doc2Vec ≈ Word2Vec**, with Doc2Vec marginally ahead — learning a document vector directly is
  only slightly better than averaging word vectors on this task.
- **The honest caveat:** even de-leaked, every model scores 96–98%. That reflects how separable
  the real/fake classes are *in this dataset* (formal wire copy vs. a different register), not the
  real-world difficulty of fake-news detection.

## Repository structure

```
fake-news-detection/
├── Fake_News_Detection_Comparative_Study.ipynb   # main notebook
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

1. Download the dataset (`True.csv`, `Fake.csv`) from
   [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
   and place both files in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run top to bottom:
   ```bash
   jupyter notebook Fake_News_Detection_Comparative_Study.ipynb
   ```

## Limitations & next steps

- The decisive test of generalisation is evaluation on a **different** fake-news source; high
  in-dataset scores say little about transfer.
- Mean-pooling is the weakest link in the Word2Vec pipeline — TF-IDF-weighted pooling or
  pretrained GloVe embeddings would be a fairer test of distributional semantics.
- The classifier is fixed to isolate the representation; repeating the comparison with a linear
  SVM is a cheap robustness check that the ranking holds.

## Author

Mainak Jana
