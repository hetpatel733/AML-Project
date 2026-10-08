### Table 8: LIAR Feature Representation & Tokenization Ablation Study

| Feature Representation           | Tokenization & Range                           | Vocabulary Limit   | Sublinear TF       | 5-Fold CV Accuracy   | 5-Fold CV F1   | Test Accuracy (tau=0.55)   | Error Disparity |FP - FN|   |
|:---------------------------------|:-----------------------------------------------|:-------------------|:-------------------|:---------------------|:---------------|:---------------------------|:----------------------------|
| Word TF-IDF (1, 2) [WINNING]     | Word unigrams + bigrams                        | 25,000 features    | True (1 + log(tf)) | 62.06% ± 1.64%       | 69.17% ± 1.31% | 62.60%                     | 15 (Optimal)                |
| Word TF-IDF (1, 3)               | Word unigrams, bigrams, trigrams               | 25,000 features    | True               | 61.36% ± 1.48%       | 68.74% ± 1.42% | 61.82%                     | 84                          |
| Char TF-IDF (3, 5)               | Character n-grams (3-5 within word boundaries) | 15,000 features    | True               | 60.75% ± 1.55%       | 67.89% ± 1.36% | 61.04%                     | 112                         |
| Word (1, 2) + Char (3, 5) Hybrid | FeatureUnion (Word 10k + Char 10k)             | 20,000 features    | True               | 60.88% ± 1.51%       | 68.02% ± 1.40% | 61.20%                     | 98                          |
