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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


def clean_text_liar(text: str) -> str:
    """
    LIAR Dedicated Preprocessing Pipeline:
    1. Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
    2. Lowercase transformation
    3. Remove URLs and HTML tags
    4. Expand contractions (preserving negation words)
    5. Clean non-alphabetic characters
    6. Normalize whitespace
    7. Retain short claim tokens without aggressive stopword removal
    8. WordNet lemmatization
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    for pattern, replacement in CONTRACTION_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    cleaned_tokens = [_lemmatize(token) for token in tokens if len(token) > 1]
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


def load_liar_dataframe():
    """Load raw LIAR dataframe with all columns and standard binary label mapping."""
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
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def load_liar_dataset(test_size: float = 0.2, as_dataframe: bool = False):
    """Load LIAR dataset returning train/test splits (80/20)."""
    df = load_liar_dataframe()
    df["clean_statement"] = df["statement"].apply(clean_text_liar)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"],
    )
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    if as_dataframe:
        return train_df, test_df, y_train, y_test
    return train_df["clean_statement"].to_numpy(), test_df["clean_statement"].to_numpy(), y_train, y_test


def load_dataset(dataset: str, test_size: float = 0.2):
    if dataset.lower() == "isot":
        return load_isot_dataset(test_size)
    if dataset.lower() == "liar":
        return load_liar_dataset(test_size, as_dataframe=False)
    raise ValueError(f"Unknown dataset: {dataset}")


class LiarFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Unified Feature Extraction Pipeline for LIAR:
    - Word TF-IDF (1,2) on cleaned statements with sublinear TF
    - Char-WB TF-IDF (3,5) on cleaned statements with sublinear TF
    - Categorical CountVectors for Subject, Speaker, Party, State, Job
    - Context TF-IDF (1,2) on Venue/Location
    - Scaled Speaker Credit History counts
    Compatible with both DataFrame (training/batch evaluation) and list/string inputs (inference).
    """

    def __init__(
        self,
        word_ngram_range=(1, 2),
        word_max_features=12000,
        char_ngram_range=(3, 5),
        char_max_features=8000,
        context_max_features=3000,
    ):
        self.word_ngram_range = tuple(word_ngram_range)
        self.word_max_features = word_max_features
        self.char_ngram_range = tuple(char_ngram_range)
        self.char_max_features = char_max_features
        self.context_max_features = context_max_features

        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=self.word_ngram_range,
                        sublinear_tf=True,
                        min_df=2,
                        max_df=0.98,
                        max_features=self.word_max_features,
                    ),
                    "clean_statement",
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=self.char_ngram_range,
                        sublinear_tf=True,
                        min_df=3,
                        max_features=self.char_max_features,
                    ),
                    "clean_statement",
                ),
                ("subject_vec", CountVectorizer(binary=True, min_df=2), "subject_clean"),
                ("speaker_vec", CountVectorizer(binary=True, min_df=2), "speaker_clean"),
                ("party_vec", CountVectorizer(binary=True), "party_clean"),
                ("state_vec", CountVectorizer(binary=True, min_df=2), "state_clean"),
                ("job_vec", CountVectorizer(binary=True, min_df=2), "job_clean"),
                (
                    "context_vec",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=self.context_max_features,
                    ),
                    "context_clean",
                ),
                (
                    "credit_counts",
                    MinMaxScaler(),
                    ["barely_true", "false", "half_true", "mostly_true", "pants_fire"],
                ),
            ],
            remainder="drop",
        )

    def _prepare_df(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, (list, tuple, np.ndarray)):
            df = pd.DataFrame({"statement": list(X)})
        else:
            df = pd.DataFrame({"statement": [str(X)]})

        if "clean_statement" not in df.columns:
            df["clean_statement"] = df["statement"].fillna("").map(clean_text_liar)
        for col in ["subject", "speaker", "party", "state", "context", "job"]:
            clean_col = f"{col}_clean"
            if clean_col not in df.columns:
                val = df[col] if col in df.columns else "unknown"
                df[clean_col] = pd.Series(val, index=df.index).fillna("unknown").astype(str).str.lower()
        for col in ["barely_true", "false", "half_true", "mostly_true", "pants_fire"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
        return df

    def fit(self, X, y=None):
        df = self._prepare_df(X)
        self.column_transformer.fit(df, y)
        return self

    def transform(self, X):
        df = self._prepare_df(X)
        return self.column_transformer.transform(df)

    def fit_transform(self, X, y=None):
        df = self._prepare_df(X)
        return self.column_transformer.fit_transform(df, y)
