import html
import re
import string
from functools import lru_cache
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from src.config import (
    ISOT_FAKE_CSV,
    ISOT_TRUE_CSV,
    LIAR_TRAIN_TSV,
    LIAR_VALID_TSV,
    LIAR_TEST_TSV,
)


def _nltk_resource(resource: str, package: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


_nltk_resource("corpora/stopwords", "stopwords")
_nltk_resource("corpora/wordnet", "wordnet")
_nltk_resource("corpora/omw-1.4", "omw-1.4")

# Standard English stopwords
BASE_STOPWORDS = set(stopwords.words("english"))

# Critical negation tokens to preserve in fake news deception detection
NEGATION_WORDS = {
    "no", "nor", "not", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "scarcely", "barely", "without", "cannot", "cant",
    "wont", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "doesnt", "dont", "didnt", "couldnt", "shouldnt", "wouldnt", "against"
}

# Negation-preserving stopword set
ENHANCED_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()

CONTRACTION_MAP = {
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
def _lemmatize(token: str) -> str:
    """Lemmatize token with WordNet (noun first, then verb)."""
    noun_lemma = LEMMATIZER.lemmatize(token, pos="n")
    return LEMMATIZER.lemmatize(noun_lemma, pos="v")


def clean_text(text: str) -> str:
    """
    Enhanced Text Preprocessing Pipeline:
    1. Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
    2. Lowercase transformation
    3. Remove URLs (http://, https://, www.) and HTML tags (<...>)
    4. Expand contractions to preserve negation semantics ('wasn't' -> 'was not')
    5. Clean non-alphabetic characters
    6. Normalize whitespace
    7. Filter stopwords while preserving critical negation words
    8. WordNet lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip URLs & HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Expand contractions
    for pattern, replacement in CONTRACTION_MAP.items():
        text = re.sub(pattern, replacement, text)

    # 5. Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenize and filter with negation preservation & length check
    tokens = text.split()
    cleaned_tokens = [
        _lemmatize(token)
        for token in tokens
        if token not in ENHANCED_STOPWORDS and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


def _split(texts: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )


def load_isot_dataset(test_size: float = 0.2):
    fake_df = pd.read_csv(ISOT_FAKE_CSV)
    true_df = pd.read_csv(ISOT_TRUE_CSV)
    fake_df["label"] = 0
    true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].drop_duplicates().dropna()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    return _split(df["text"].to_numpy(), df["label"].to_numpy(dtype=int), test_size)


def load_liar_dataset(test_size: float = 0.2):
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
    df["text"] = df["statement"].fillna("").map(clean_text)
    df = df.dropna(subset=["label"])
    df = df[df["text"].str.len() > 0]
    return _split(df["text"].to_numpy(), df["label"].to_numpy(dtype=int), test_size)


def load_dataset(dataset: str, test_size: float = 0.2):
    if dataset.lower() == "isot":
        return load_isot_dataset(test_size)
    if dataset.lower() == "liar":
        return load_liar_dataset(test_size)
    raise ValueError(f"Unknown dataset: {dataset}")
