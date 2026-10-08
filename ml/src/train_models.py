import json
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
from src.data_preprocessing import load_dataset

RANDOM_STATE = 42

# Selected optimal TF-IDF parameters from training-set CV exploration
TFIDF_CONFIG = {
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
        MODEL_LOGISTIC_REGRESSION: 0.61079,
        MODEL_MULTINOMIAL_NB: 0.60688,
        MODEL_LINEAR_SVM: 0.57562,
        MODEL_DECISION_TREE: 0.56311,
        MODEL_RANDOM_FOREST: 0.60883,
    },
}


def build_model(model_name: str):
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
    raise ValueError(f"Unknown model: {model_name}")


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
    vectorizer = TfidfVectorizer(
        ngram_range=(TFIDF_CONFIG["ngram_range"][0], TFIDF_CONFIG["ngram_range"][1]),
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        min_df=TFIDF_CONFIG["min_df"],
        max_df=TFIDF_CONFIG["max_df"],
        max_features=TFIDF_CONFIG["max_features"],
    )
    
    # Fit vectorizer strictly on training partition
    train_features = vectorizer.fit_transform(x_train)
    test_features = vectorizer.transform(x_test)
    
    # 5-Fold Stratified Cross Validation on training features only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_model = build_model(model_name)
    cv_results = cross_validate(
        cv_model,
        train_features,
        y_train,
        cv=skf,
        scoring=["accuracy", "f1", "precision", "recall"],
        n_jobs=-1,
    )
    
    # Final Model Training on full 80% train split
    model = build_model(model_name)
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

    metadata = {
        "dataset": dataset,
        "model": model_name,
        "name": MODEL_DISPLAY_NAMES[model_name],
        "training_time_seconds": training_time,
        "parameters": model.get_params(),
        "tfidf": TFIDF_CONFIG,
        "preprocessing": [
            "HTML entity decoding",
            "lowercase conversion",
            "URL and HTML tag removal",
            "contraction expansion for negation retention",
            "non-alphabetic character cleaning",
            "whitespace normalization",
            "negation-preserving English stopword filtering",
            "WordNet lemmatization",
        ],
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
        "representation": "TF-IDF (unigrams + bigrams, sublinear TF)",
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
    benchmark = {
        "dataset": {
            "id": dataset,
            "name": dataset.upper(),
            "description": "Binary fake-news classification dataset.",
            "totalRecords": total,
            "classes": ["FAKE", "REAL"],
            "classDistribution": _class_distribution(np.concatenate([y_train, y_test])),
            "labelMapping": {"0": "FAKE", "1": "REAL"},
            "preprocessing": {
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
                "ngramRange": TFIDF_CONFIG["ngram_range"],
                "sublinearTf": TFIDF_CONFIG["sublinear_tf"],
                "minDf": TFIDF_CONFIG["min_df"],
                "maxDf": TFIDF_CONFIG["max_df"],
                "maxFeatures": TFIDF_CONFIG["max_features"],
            },
        },
        "experiment": {
            "version": "improved-preprocessing-v2",
            "name": "Improved Preprocessing Pipeline with Negation Preservation & Sublinear TF-IDF",
            "trainingTimestamp": datetime.now(timezone.utc).isoformat(),
            "randomSeed": RANDOM_STATE,
            "split": "stratified 80% training / 20% testing",
            "validation": {
                "used": False,
                "reason": "No separate validation split. Preprocessing experiments conducted on 80% train split via 5-fold CV; final evaluation on untouched 20% test partition.",
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
    for dataset in ("isot", "liar"):
        benchmark_path = train_dataset(dataset)
        print(f"Wrote {benchmark_path}")


if __name__ == "__main__":
    main()
