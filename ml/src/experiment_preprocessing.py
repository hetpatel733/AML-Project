import re
import html
import string
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Any

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import (
    ISOT_FAKE_CSV,
    ISOT_TRUE_CSV,
    LIAR_TRAIN_TSV,
    LIAR_VALID_TSV,
    LIAR_TEST_TSV,
)

# Download resources quietly
for res, pkg in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")]:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(pkg, quiet=True)

NLTK_STOPWORDS = set(stopwords.words("english"))
NEGATION_WORDS = {
    "no", "nor", "not", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "scarcely", "barely", "without", "cannot", "cant",
    "wont", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "doesnt", "dont", "didnt", "couldnt", "shouldnt", "wouldnt", "against"
}
NEGATION_PRESERVED_STOPWORDS = NLTK_STOPWORDS - NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()

CONTRACTIONS = {
    r"won\'t": "will not",
    r"can\'t": "can not",
    r"cannot": "can not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}


@lru_cache(maxsize=300000)
def _lemmatize_token(token: str) -> str:
    # First lemmatize as verb if it ends with standard verb inflections, else noun
    noun_lemma = LEMMATIZER.lemmatize(token, pos="n")
    verb_lemma = LEMMATIZER.lemmatize(noun_lemma, pos="v")
    return verb_lemma


@lru_cache(maxsize=300000)
def _lemmatize_basic(token: str) -> str:
    return LEMMATIZER.lemmatize(token)


def clean_text_baseline(text: str) -> str:
    """Baseline paper preprocessing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [token for token in tokens if token not in NLTK_STOPWORDS]
    return " ".join(_lemmatize_basic(token) for token in tokens)


def clean_text_enhanced(text: str) -> str:
    """
    Enhanced Preprocessing Pipeline:
    1. Unescape HTML entities (e.g., &amp; -> &)
    2. Lowercase conversion
    3. Strip URLs and HTML tags
    4. Expand English contractions so negation is preserved ('didn't' -> 'did not')
    5. Remove non-alphabetic characters
    6. Normalize whitespace
    7. Filter stopwords while preserving critical negation words
    8. WordNet lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML
    text = html.unescape(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Strip URLs & HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 4. Expand Contractions
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
        
    # 5. Non-alphabetic filtering
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # 7. Tokenize and filter with negation preservation
    tokens = text.split()
    cleaned_tokens = [
        _lemmatize_token(t) for t in tokens 
        if t not in NEGATION_PRESERVED_STOPWORDS and len(t) > 1
    ]
    
    return " ".join(cleaned_tokens)


def load_raw_isot():
    fake_df = pd.read_csv(ISOT_FAKE_CSV)
    true_df = pd.read_csv(ISOT_TRUE_CSV)
    fake_df["label"] = 0
    true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["raw_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["raw_text", "label"]].drop_duplicates().dropna()
    return df["raw_text"].to_numpy(), df["label"].to_numpy(dtype=int)


def load_raw_liar():
    columns = [
        "id", "label", "statement", "subject", "speaker", "job", "state", "party",
        "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context",
    ]
    frames = [
        pd.read_csv(path, sep="\t", header=None, names=columns)
        for path in (LIAR_TRAIN_TSV, LIAR_VALID_TSV, LIAR_TEST_TSV)
    ]
    df = pd.concat(frames, ignore_index=True)
    label_mapping = {
        "true": 1,
        "mostly-true": 1,
        "half-true": 1,
        "false": 0,
        "barely-true": 0,
        "pants-fire": 0,
    }
    df["label"] = df["label"].map(label_mapping)
    df["raw_text"] = df["statement"].fillna("")
    df = df.dropna(subset=["label"])
    df = df[df["raw_text"].str.len() > 0]
    return df["raw_text"].to_numpy(), df["label"].to_numpy(dtype=int)


def run_experiments(dataset_name: str, raw_texts: np.ndarray, labels: np.ndarray):
    print(f"\n=======================================================")
    print(f" EXPERIMENTING ON {dataset_name.upper()} (80% Train Split Only)")
    print(f"=======================================================")
    
    # 1. 80/20 Stratified Split - Held out 20% test is untouched!
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        raw_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Total dataset: {len(labels)}, Train set: {len(y_train)}, Test set: {len(y_test)}")
    
    # Clean train texts with baseline vs enhanced
    print("Preprocessing training data (Baseline vs Enhanced)...")
    t0 = time.time()
    x_train_baseline = np.array([clean_text_baseline(t) for t in x_train_raw])
    t1 = time.time()
    print(f"Baseline cleaning done in {t1 - t0:.2f}s")
    
    x_train_enhanced = np.array([clean_text_enhanced(t) for t in x_train_raw])
    t2 = time.time()
    print(f"Enhanced cleaning done in {t2 - t1:.2f}s")
    
    # Grid of configurations to test using 5-Fold Stratified CV on the training partition only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    configs = [
        # (Name, clean_type, sublinear_tf, min_df, max_df, ngram_range, max_features)
        ("Baseline (Paper)", "baseline", False, 1, 1.0, (1, 2), 10000),
        ("Enhanced + sublinear=False, min_df=1, max_df=1.0, max_feat=10000", "enhanced", False, 1, 1.0, (1, 2), 10000),
        ("Enhanced + sublinear=True, min_df=2, max_df=0.98, max_feat=10000", "enhanced", True, 2, 0.98, (1, 2), 10000),
        ("Enhanced + sublinear=True, min_df=3, max_df=0.95, max_feat=10000", "enhanced", True, 3, 0.95, (1, 2), 10000),
        ("Enhanced + sublinear=True, min_df=5, max_df=0.95, max_feat=10000", "enhanced", True, 5, 0.95, (1, 2), 10000),
        ("Enhanced + sublinear=True, min_df=2, max_df=0.98, max_feat=15000", "enhanced", True, 2, 0.98, (1, 2), 15000),
        ("Enhanced + sublinear=True, min_df=2, max_df=0.98, max_feat=20000", "enhanced", True, 2, 0.98, (1, 2), 20000),
    ]
    
    # We will evaluate across all 5 models
    models_to_test = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Multinomial NB": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    results = []
    
    for name, clean_type, sublinear_tf, min_df, max_df, ngrams, max_feat in configs:
        texts = x_train_baseline if clean_type == "baseline" else x_train_enhanced
        
        vec = TfidfVectorizer(
            ngram_range=ngrams,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            max_features=max_feat
        )
        x_train_vec = vec.fit_transform(texts)
        
        scores = {}
        for m_name, model in models_to_test.items():
            cv_scores = cross_val_score(model, x_train_vec, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
            scores[m_name] = np.mean(cv_scores)
            
        avg_score = np.mean(list(scores.values()))
        results.append({
            "config": name,
            "sublinear_tf": sublinear_tf,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_feat,
            "scores": scores,
            "avg_accuracy": avg_score
        })
        print(f"\nConfig: {name}")
        for m_name, score in scores.items():
            print(f"  {m_name}: {score * 100:.2f}%")
        print(f"  --> Mean CV Accuracy: {avg_score * 100:.2f}%")
        
    best_config = max(results, key=lambda x: x["avg_accuracy"])
    print(f"\n>>> BEST CONFIGURATION FOR {dataset_name.upper()}: {best_config['config']} (Mean CV Acc: {best_config['avg_accuracy'] * 100:.2f}%)")
    return best_config


if __name__ == "__main__":
    isot_texts, isot_labels = load_raw_isot()
    isot_best = run_experiments("isot", isot_texts, isot_labels)
    
    liar_texts, liar_labels = load_raw_liar()
    liar_best = run_experiments("liar", liar_texts, liar_labels)
