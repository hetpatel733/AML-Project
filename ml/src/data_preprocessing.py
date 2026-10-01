import re
import string
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split

from src.config import (
    FAKE_CSV_PATH,
    TRUE_CSV_PATH,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    LABEL_MAPPING
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
        # Fallback standard English stopword set if offline
        STOPWORDS = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
            "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
            "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves"
        }

def clean_text(text: str) -> str:
    """
    Standard NLP preprocessing pipeline for news text.
    
    Steps:
    1. Lowercase normalization
    2. Removal of URLs, web links, and email addresses
    3. Removal of HTML/XML tags
    4. Removal of news wire agency prefixes (e.g. 'WASHINGTON (Reuters) -') to prevent source leakage
    5. Removal of punctuation and non-alphabetic noise
    6. Removal of standard English stopwords
    7. Whitespace normalization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs & hyperlinks
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove Reuters/AP style dateline prefixes (prevents trivial wire leakage)
    text = re.sub(r"^[a-z\s]+(?:\(reuters\)|\(ap\))\s*-\s*", " ", text)

    # 5. Remove punctuation and numbers, keeping alphabetic word tokens
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 6. Tokenize & filter stopwords
    tokens = text.split()
    filtered_tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

    # 7. Rejoin and normalize whitespace
    return " ".join(filtered_tokens)


def load_raw_datasets(fake_path=FAKE_CSV_PATH, true_path=TRUE_CSV_PATH) -> pd.DataFrame:
    """
    Loads Fake.csv and True.csv datasets, assigns binary labels,
    merges them, and removes duplicates and incomplete entries.
    
    Label Convention:
    - 0 -> FAKE
    - 1 -> REAL
    """
    print(f"[Data Loader] Loading Fake dataset from: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 0
    fake_df["label_name"] = "FAKE"

    print(f"[Data Loader] Loading Real dataset from: {true_path}")
    true_df = pd.read_csv(true_path)
    true_df["label"] = 1
    true_df["label_name"] = "REAL"

    # Combine datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    initial_count = len(combined_df)

    # Drop nulls in core text columns
    combined_df["title"] = combined_df["title"].fillna("")
    combined_df["text"] = combined_df["text"].fillna("")

    # Combine title + text to maximize lexical context
    combined_df["combined_text"] = combined_df["title"].str.strip() + " " + combined_df["text"].str.strip()

    # Drop rows where combined text is empty or blank
    combined_df = combined_df[combined_df["combined_text"].str.strip() != ""]

    # Drop duplicate combined articles
    combined_df = combined_df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    deduped_count = len(combined_df)

    print(f"[Data Loader] Loaded {initial_count} total articles. After deduplication & null cleaning: {deduped_count} articles.")
    print(f"[Data Loader] Class Distribution:\n{combined_df['label_name'].value_counts()}")

    return combined_df


def prepare_and_split_data(
    df: pd.DataFrame = None,
    test_size: float = TEST_RATIO,
    random_state: int = RANDOM_STATE
):
    """
    Cleans raw text, extracts features (X) and labels (y), and performs
    a stratified train-test split to avoid class imbalance across splits.
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    if df is None:
        df = load_raw_datasets()

    print("[Data Preprocessing] Applying NLP cleaning pipeline to text corpus...")
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Filter out any samples that became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    X = df["clean_text"].values
    y = df["label"].values

    print(f"[Data Preprocessing] Total clean samples: {len(X)}")
    print(f"[Train/Test Split] Splitting with test_size={test_size}, random_state={random_state}, stratified...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Train/Test Split] Train size: {len(X_train)} samples | Test size: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, df


def prepare_and_split_data_3way(
    df: pd.DataFrame = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE
):
    """
    Cleans raw text, extracts features (X) and labels (y), and performs
    a 3-way stratified split: Train (70%), Validation (15%), Test (15%).
    
    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test, df
    """
    if df is None:
        df = load_raw_datasets()

    print("[Data Preprocessing] Applying NLP cleaning pipeline to text corpus...")
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Filter out any samples that became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    X = df["clean_text"].values
    y = df["label"].values

    total_samples = len(X)
    print(f"[Data Preprocessing] Total clean samples: {total_samples}")
    print(f"[3-Way Split] Splitting: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%, random_state={random_state}, stratified...")

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )

    # Second split: Val vs Test from Temp
    val_rel_ratio = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_rel_ratio),
        random_state=random_state,
        stratify=y_temp
    )

    print(f"[3-Way Split] Train: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%) | "
          f"Val: {len(X_val)} ({len(X_val)/total_samples*100:.1f}%) | "
          f"Test: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, df
