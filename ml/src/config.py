import os
from pathlib import Path

# Base ML Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
API_DIR = BASE_DIR / "api"

# Dataset Files
FAKE_CSV_PATH = DATA_DIR / "Fake.csv"
TRUE_CSV_PATH = DATA_DIR / "True.csv"

# Saved Model & Artifact Paths
BOW_VECTORIZER_PATH = MODELS_DIR / "bow_vectorizer.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
VECTORIZER_PATH = TFIDF_VECTORIZER_PATH

BOW_LOGISTIC_REGRESSION_PATH = MODELS_DIR / "bow_logistic_regression.joblib"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "tfidf_logistic_regression.joblib"
NAIVE_BAYES_PATH = MODELS_DIR / "tfidf_naive_bayes.joblib"
SVM_PATH = MODELS_DIR / "tfidf_svm.joblib"
RANDOM_FOREST_PATH = MODELS_DIR / "tfidf_random_forest.joblib"
PASSIVE_AGGRESSIVE_PATH = MODELS_DIR / "tfidf_passive_aggressive.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_config.json"

# Evaluation Result Artifacts
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
METRICS_CSV_PATH = RESULTS_DIR / "metrics_comparison.csv"
CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrix.png"
MODEL_COMPARISON_PNG = RESULTS_DIR / "model_comparison.png"
ROC_CURVE_PNG = RESULTS_DIR / "roc_curve.png"
PRECISION_RECALL_PNG = RESULTS_DIR / "precision_recall_curve.png"

# Reproducibility and Training Parameters
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CV_FOLDS = 5

# Binary Classification Label Mapping
# 0 -> FAKE (Deceptive / Fabricated / Low-credibility patterns)
# 1 -> REAL (Verified / High-credibility journalistic patterns)
LABEL_MAPPING = {
    0: "FAKE",
    1: "REAL"
}
REVERSE_LABEL_MAPPING = {
    "FAKE": 0,
    "REAL": 1
}

# Feature Extraction Hyperparameters
BOW_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.85,
    "strip_accents": "unicode"
}

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.85,
    "sublinear_tf": True,
    "strip_accents": "unicode"
}

# Ensure artifact output directories exist
for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR, API_DIR]:
    path.mkdir(parents=True, exist_ok=True)
