import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    MODEL_DECISION_TREE,
    MODEL_DISPLAY_NAMES,
    MODEL_LINEAR_SVM,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_MULTINOMIAL_NB,
    MODEL_NAMES,
    MODEL_RANDOM_FOREST,
    get_benchmark_json_path,
    get_metadata_path,
    get_model_dir,
    get_model_path,
    get_vectorizer_path,
)
from src.data_preprocessing import LiarFeatureExtractor, clean_text_liar, load_dataset

RANDOM_STATE = 42

# Selected optimal TF-IDF parameters for ISOT
ISOT_TFIDF_CONFIG = {
    "ngram_range": [1, 2],
    "sublinear_tf": True,
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 10000,
}

# Baseline paper accuracies for replication comparison
PAPER_BASELINES = {
    "isot": {
        MODEL_LOGISTIC_REGRESSION: 0.98836,
        MODEL_MULTINOMIAL_NB: 0.94910,
        MODEL_LINEAR_SVM: 0.99373,
        MODEL_DECISION_TREE: 0.99437,
        MODEL_RANDOM_FOREST: 0.99604,
    },
    "liar": {
        MODEL_LOGISTIC_REGRESSION: 0.62800,
        MODEL_MULTINOMIAL_NB: 0.62300,
        MODEL_LINEAR_SVM: 0.62600,
        MODEL_DECISION_TREE: 0.57100,
        MODEL_RANDOM_FOREST: 0.62500,
    },
}


def build_model(dataset: str, model_name: str):
    if dataset == "liar":
        if model_name == MODEL_LOGISTIC_REGRESSION:
            return LogisticRegression(C=0.5, solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)
        if model_name == MODEL_MULTINOMIAL_NB:
            return MultinomialNB(alpha=3.0, fit_prior=True)
        if model_name == MODEL_LINEAR_SVM:
            return LinearSVC(C=0.05, max_iter=3000, random_state=RANDOM_STATE)
        if model_name == MODEL_DECISION_TREE:
            return DecisionTreeClassifier(
                criterion="entropy",
                max_depth=8,
                min_samples_split=30,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
            )
        if model_name == MODEL_RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=50,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    else:
        if model_name == MODEL_LOGISTIC_REGRESSION:
            return LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        if model_name == MODEL_MULTINOMIAL_NB:
            return MultinomialNB(alpha=1.0)
        if model_name == MODEL_LINEAR_SVM:
            return LinearSVC(C=1.0, random_state=RANDOM_STATE)
        if model_name == MODEL_DECISION_TREE:
            return DecisionTreeClassifier(random_state=RANDOM_STATE)
        if model_name == MODEL_RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    raise ValueError(f"Unknown model: {model_name} for dataset: {dataset}")


def build_vectorizer(dataset: str):
    if dataset == "liar":
        return LiarFeatureExtractor(
            word_ngram_range=(1, 2),
            word_max_features=12000,
            char_ngram_range=(3, 5),
            char_max_features=8000,
            context_max_features=3000,
        )
    return TfidfVectorizer(
        ngram_range=(ISOT_TFIDF_CONFIG["ngram_range"][0], ISOT_TFIDF_CONFIG["ngram_range"][1]),
        sublinear_tf=ISOT_TFIDF_CONFIG["sublinear_tf"],
        min_df=ISOT_TFIDF_CONFIG["min_df"],
        max_df=ISOT_TFIDF_CONFIG["max_df"],
        max_features=ISOT_TFIDF_CONFIG["max_features"],
    )


def _scores(model: Any, features) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        positive_index = list(model.classes_).index(1)
        return probabilities[:, positive_index]
    return np.asarray(model.decision_function(features), dtype=float)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, scores)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "rocAuc": float(roc_auc),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def train_one(dataset: str, model_name: str, x_train, x_test, y_train, y_test) -> Dict[str, Any]:
    started = time.perf_counter()

    # Preprocessing & Vectorization with selected configuration
    vectorizer = build_vectorizer(dataset)

    # Fit vectorizer strictly on training partition
    train_features = vectorizer.fit_transform(x_train)
    test_features = vectorizer.transform(x_test)

    # 5-Fold Stratified Cross Validation on training features only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_model = build_model(dataset, model_name)
    cv_results = cross_validate(
        cv_model,
        train_features,
        y_train,
        cv=skf,
        scoring=["accuracy", "f1", "precision", "recall"],
        n_jobs=-1,
    )

    # Final Model Training on full 80% train split
    model = build_model(dataset, model_name)
    model.fit(train_features, y_train)

    # Final Evaluation on untouched 20% test partition
    predictions = model.predict(test_features)
    scores = _scores(model, test_features)
    metrics = calculate_metrics(y_test, predictions, scores)
    training_time = time.perf_counter() - started

    # Comparison against Paper Baseline
    paper_baseline = PAPER_BASELINES.get(dataset, {}).get(model_name, None)
    if paper_baseline is not None:
        delta = metrics["accuracy"] - paper_baseline
        gain_pct = (delta / paper_baseline) * 100
        paper_comparison = {
            "accuracy": paper_baseline,
            "source": "Paper Baseline Replication",
            "delta": float(delta),
            "percentageGain": f"{gain_pct:+.2f}%",
            "improved": bool(delta >= 0),
        }
    else:
        paper_comparison = None

    # Model persistence
    model_dir = get_model_dir(dataset, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, get_model_path(dataset, model_name))
    joblib.dump(vectorizer, get_vectorizer_path(dataset, model_name))

    cv_summary = {
        "folds": 5,
        "strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on 80% train partition only",
        "mean": {
            "accuracy": float(np.mean(cv_results["test_accuracy"])),
            "f1": float(np.mean(cv_results["test_f1"])),
            "precision": float(np.mean(cv_results["test_precision"])),
            "recall": float(np.mean(cv_results["test_recall"])),
        },
        "std": {
            "accuracy": float(np.std(cv_results["test_accuracy"])),
            "f1": float(np.std(cv_results["test_f1"])),
            "precision": float(np.std(cv_results["test_precision"])),
            "recall": float(np.std(cv_results["test_recall"])),
        },
        "scores": {
            "accuracy": [float(s) for s in cv_results["test_accuracy"]],
            "f1": [float(s) for s in cv_results["test_f1"]],
        },
    }

    if dataset == "liar":
        preprocessing_steps = [
            "HTML entity decoding",
            "lowercase conversion",
            "URL and HTML tag removal",
            "contraction expansion for negation retention",
            "non-alphabetic character cleaning",
            "whitespace normalization",
            "WordNet lemmatization without aggressive stopword removal",
            "Word TF-IDF (1,2) with sublinear TF",
            "Char-WB TF-IDF (3,5) with sublinear TF",
            "Subject, Speaker, Party, State, Job binary count vectors",
            "Context venue/location TF-IDF (1,2)",
            "Speaker credit history scaled continuous features",
        ]
        feature_repr = "Multi-modal Feature Extractor (Word (1,2) + Char-WB (3,5) TF-IDF + Metadata Categoricals & Scaled Credit History)"
    else:
        preprocessing_steps = [
            "HTML entity decoding",
            "lowercase conversion",
            "URL and HTML tag removal",
            "contraction expansion for negation retention",
            "non-alphabetic character cleaning",
            "whitespace normalization",
            "negation-preserving English stopword filtering",
            "WordNet lemmatization",
        ]
        feature_repr = "TF-IDF (unigrams + bigrams, sublinear TF)"

    metadata = {
        "dataset": dataset,
        "model": model_name,
        "name": MODEL_DISPLAY_NAMES[model_name],
        "training_time_seconds": training_time,
        "parameters": model.get_params(),
        "representation": feature_repr,
        "preprocessing": preprocessing_steps,
        "train_count": len(y_train),
        "test_count": len(y_test),
        "metrics": metrics,
        "crossValidation": cv_summary,
        "paperReference": paper_comparison,
    }
    with get_metadata_path(dataset, model_name).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {
        "id": model_name,
        "name": MODEL_DISPLAY_NAMES[model_name],
        "type": "classical_ml",
        "representation": feature_repr,
        "status": "trained",
        "parameters": metadata["parameters"],
        "trainingTime": training_time,
        "training": {"count": len(y_train), "percentage": 80.0},
        "validation": {"used": False, "count": 0, "percentage": 0.0},
        "test": {"count": len(y_test), "percentage": 20.0, "metrics": metrics},
        "metrics": metrics,
        "crossValidation": cv_summary,
        "paperReference": paper_comparison,
        "improvementAgainstPaper": {
            "paperBaselineAccuracy": paper_baseline,
            "accuracyDelta": float(metrics["accuracy"] - paper_baseline) if paper_baseline else 0.0,
            "percentageGain": f"{((metrics['accuracy'] - paper_baseline) / paper_baseline) * 100:+.2f}%" if paper_baseline else "0.00%",
            "improved": bool((metrics["accuracy"] - paper_baseline) >= 0) if paper_baseline else True,
        },
        "confusionMatrix": {
            "labels": ["FAKE", "REAL"],
            "matrix": metrics["confusion_matrix"],
            "truePositive": metrics["tp"],
            "trueNegative": metrics["tn"],
            "falsePositive": metrics["fp"],
            "falseNegative": metrics["fn"],
        },
        "artifactPath": str(model_dir),
    }


def _class_distribution(labels: np.ndarray) -> Dict[str, int]:
    return {"FAKE": int(np.sum(labels == 0)), "REAL": int(np.sum(labels == 1))}


def write_benchmark(dataset: str, x_train, x_test, y_train, y_test, models) -> Path:
    total = len(y_train) + len(y_test)
    if dataset == "liar":
        preprocessing_dict = {
            "lowercase": True,
            "removePunctuation": True,
            "removeSpecialCharacters": True,
            "removeUrlsAndHtml": True,
            "normalizeWhitespace": True,
            "expandContractions": True,
            "preserveNegationWords": True,
            "removeStopwords": "retained (short-claim preservation)",
            "lemmatization": "WordNet",
            "tokenization": "word & character n-grams with contraction expansion",
            "vectorizer": "LiarFeatureExtractor (Word TF-IDF + Char-WB TF-IDF + Metadata + Credit Counts)",
            "wordNgramRange": [1, 2],
            "wordMaxFeatures": 12000,
            "charNgramRange": [3, 5],
            "charMaxFeatures": 8000,
            "metadataFeatures": ["subject", "speaker", "party", "state", "job", "context", "credit_history"],
        }
        exp_name = "Multi-Modal Feature Engineering Pipeline with Tuned Hyperparameters"
    else:
        preprocessing_dict = {
            "lowercase": True,
            "removePunctuation": True,
            "removeSpecialCharacters": True,
            "removeUrlsAndHtml": True,
            "normalizeWhitespace": True,
            "expandContractions": True,
            "preserveNegationWords": True,
            "removeStopwords": "custom (negation-preserving)",
            "lemmatization": "WordNet",
            "tokenization": "whitespace & alphabetic tokens with contraction expansion",
            "vectorizer": "TF-IDF",
            "ngramRange": ISOT_TFIDF_CONFIG["ngram_range"],
            "sublinearTf": ISOT_TFIDF_CONFIG["sublinear_tf"],
            "minDf": ISOT_TFIDF_CONFIG["min_df"],
            "maxDf": ISOT_TFIDF_CONFIG["max_df"],
            "maxFeatures": ISOT_TFIDF_CONFIG["max_features"],
        }
        exp_name = "Improved Preprocessing Pipeline with Negation Preservation & Sublinear TF-IDF"

    benchmark = {
        "dataset": {
            "id": dataset,
            "name": dataset.upper(),
            "description": "Binary fake-news classification dataset.",
            "totalRecords": total,
            "classes": ["FAKE", "REAL"],
            "classDistribution": _class_distribution(np.concatenate([y_train, y_test])),
            "labelMapping": {"0": "FAKE", "1": "REAL"},
            "preprocessing": preprocessing_dict,
        },
        "experiment": {
            "version": "improved-liar-v2" if dataset == "liar" else "improved-preprocessing-v2",
            "name": exp_name,
            "trainingTimestamp": datetime.now(timezone.utc).isoformat(),
            "randomSeed": RANDOM_STATE,
            "split": "stratified 80% training / 20% testing",
            "validation": {
                "used": False,
                "reason": "No separate validation split. Preprocessing and tuning conducted on 80% train split via 5-fold CV; final evaluation on untouched 20% test partition.",
            },
            "crossValidationOnTrain": {
                "folds": 5,
                "strategy": "StratifiedKFold on 80% training partition only",
            },
        },
        "splits": {
            "training": {"count": len(y_train), "percentage": 80.0, "classDistribution": _class_distribution(y_train)},
            "validation": {"count": 0, "percentage": 0.0, "used": False},
            "testing": {"count": len(y_test), "percentage": 20.0, "classDistribution": _class_distribution(y_test)},
        },
        "models": models,
        "ensemble": {"status": "not_used", "reason": "Models were trained and evaluated separately."},
    }
    path = get_benchmark_json_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(benchmark, file, indent=2, allow_nan=False)
    return path


def train_dataset(dataset: str) -> Path:
    print(f"\n=======================================================")
    print(f" TRAINING AND EVALUATING ON {dataset.upper()}")
    print(f"=======================================================")
    x_train, x_test, y_train, y_test = load_dataset(dataset, test_size=0.2)
    print(f"Loaded {dataset.upper()}: Train={len(y_train)}, Test={len(y_test)}")
    results = []
    for model_name in MODEL_NAMES:
        print(f"  Training {MODEL_DISPLAY_NAMES[model_name]}...")
        res = train_one(dataset, model_name, x_train, x_test, y_train, y_test)
        acc = res["metrics"]["accuracy"]
        f1 = res["metrics"]["f1"]
        paper_acc = PAPER_BASELINES.get(dataset, {}).get(model_name, 0.0)
        delta = acc - paper_acc
        print(f"    -> Test Accuracy: {acc*100:.2f}% (Paper: {paper_acc*100:.2f}%, Delta: {delta*100:+.2f}%), F1: {f1*100:.2f}%")
        results.append(res)
    benchmark_path = write_benchmark(dataset, x_train, x_test, y_train, y_test, results)
    return benchmark_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark fake news classification models.")
    parser.add_argument(
        "--dataset",
        choices=["isot", "liar", "all"],
        default="liar",
        help="Dataset to train models on. Default is 'liar'.",
    )
    args = parser.parse_args()

    datasets = ["isot", "liar"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        benchmark_path = train_dataset(dataset)
        print(f"Wrote {benchmark_path}")


if __name__ == "__main__":
    main()
