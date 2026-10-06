import re
import string
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split

from src.config import (
    ISOT_FAKE_CSV,
    ISOT_TRUE_CSV,
    LIAR_TRAIN_TSV,
    LIAR_VALID_TSV,
    LIAR_TEST_TSV
)

# Safely verify/download NLTK stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except (LookupError, AttributeError):
    try:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"])

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

def load_isot_dataset():
    fake_df = pd.read_csv(ISOT_FAKE_CSV)
    true_df = pd.read_csv(ISOT_TRUE_CSV)
    
    fake_df["label"] = 0
    true_df["label"] = 1
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]]
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates()
    
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""]
    
    X = df["text"].values
    y = df["label"].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_liar_dataset():
    columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context"]
    
    train_df = pd.read_csv(LIAR_TRAIN_TSV, sep="\t", header=None, names=columns)
    val_df = pd.read_csv(LIAR_VALID_TSV, sep="\t", header=None, names=columns)
    test_df = pd.read_csv(LIAR_TEST_TSV, sep="\t", header=None, names=columns)
    
    # Mapping: true, mostly-true, half-true -> 1 (True)
    # false, barely-true, pants-fire -> 0 (Fake)
    label_mapping = {
        "true": 1,
        "mostly-true": 1,
        "half-true": 1,
        "false": 0,
        "barely-true": 0,
        "pants-fire": 0
    }
    
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].map(label_mapping)
        df["text"] = df["statement"].fillna("")
        df["text"] = df["text"].apply(clean_text)
        
    train_df = train_df.dropna(subset=["text", "label"])
    val_df = val_df.dropna(subset=["text", "label"])
    test_df = test_df.dropna(subset=["text", "label"])
    
    train_df = train_df[train_df["text"].str.strip() != ""]
    val_df = val_df[val_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]
    
    return (
        train_df["text"].values, val_df["text"].values, test_df["text"].values,
        train_df["label"].values.astype(int), val_df["label"].values.astype(int), test_df["label"].values.astype(int)
    )

