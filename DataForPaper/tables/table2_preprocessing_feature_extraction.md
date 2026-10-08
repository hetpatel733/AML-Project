### Table 2: Preprocessing and Feature Extraction Configuration

| Pipeline Stage             | ISOT Configuration                                           | LIAR Configuration                                           | Purpose & Rationale                                                    |
|:---------------------------|:-------------------------------------------------------------|:-------------------------------------------------------------|:-----------------------------------------------------------------------|
| HTML Entity Decoding       | Applied (html.unescape)                                      | Applied (html.unescape)                                      | Normalize &amp;, &quot;, &lt; to standard character representations    |
| Case Normalization         | Lowercase                                                    | Lowercase                                                    | Eliminate case sensitivity across article and claim text               |
| URL & HTML Tag Stripping   | Regex removal of http/https/www and <tags>                   | Regex removal of http/https/www and <tags>                   | Prevent web-scraping artifacts from introducing spurious shortcuts     |
| Contraction Expansion      | Expanded (e.g. wasn't -> was not, won't -> will not)         | Expanded (e.g. wasn't -> was not, won't -> will not)         | Preserve semantic negation boundaries critical for deception detection |
| Non-Alphabetic Cleaning    | Regex strip non-alphabetic ([^a-z\s])                        | Regex strip non-alphabetic ([^a-z\s])                        | Remove punctuation noise and numerical tokens                          |
| Stopword Filtering         | Enhanced Negation-Preserving Stopword Filtering              | No Stopword Removal (Full Lexical Retention)                 | Short claims (17 words) lose critical meaning if stopwords are removed |
| Lemmatization              | WordNet Lemmatizer (Noun + Verb passes)                      | WordNet Lemmatizer (Noun + Verb passes)                      | Reduce inflected forms to root lemma while retaining meaning           |
| TF-IDF Vectorization       | Unigrams + Bigrams (1, 2), Sublinear TF, max_features=10,000 | Unigrams + Bigrams (1, 2), Sublinear TF, max_features=25,000 | Sublinear TF (1 + log(tf)) dampens high-frequency word saturation      |
| Document Frequency Pruning | min_df=2, max_df=0.98                                        | min_df=2, max_df=0.98                                        | Eliminate singletons and corpus-wide invariant terms                   |
