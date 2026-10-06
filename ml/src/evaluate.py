"""
Evaluation module for computing metrics and writing benchmarks.
Handles 5-fold CV results, final test metrics, and benchmark files.
"""
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    get_benchmarks_json_path,
    get_benchmarks_csv_path,
    get_predictions_dir,
    MODEL_TFIDF_PAC,
    MODEL_TFIDF_RF,
    MODEL_TFIDF_LR,
    MODEL_GLOVE_CNN_BILSTM
)


def _metric_block(metrics):
    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "rocAuc": metrics.get("roc_auc"),
        "tp": metrics.get("tp"),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn")
    }


def write_dataset_benchmark(dataset, experiment_version, timestamp, splits,
                             label_mapping, preprocessing, model_results,
                             ensemble=None):
    """Serialize one authoritative benchmark document from executed results."""
    X_train, y_train = splits["training"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["testing"]
    class_distribution = {
        "FAKE": int(np.sum(y_train == 0) + np.sum(y_val == 0) + np.sum(y_test == 0)),
        "REAL": int(np.sum(y_train == 1) + np.sum(y_val == 1) + np.sum(y_test == 1))
    }
    total_records = len(y_train) + len(y_val) + len(y_test)

    models = []
    for result in model_results:
        metrics = result.get("test_metrics", {})
        folds = result.get("cv_folds", [])
        model_entry = {
            "id": result["model"],
            "name": result.get("name", result["model"]),
            "type": result.get("type", "classical_ml"),
            "representation": result.get("representation", "TF-IDF"),
            "status": result.get("status", "trained"),
            "training": result.get("training", {"count": len(y_train), "percentage": len(y_train) / total_records * 100}),
            "crossValidation": {
                "folds": folds,
                "mean": {key.replace("_mean", ""): value for key, value in result.get("cv_stats", {}).items() if key.endswith("_mean")},
                "std": {key.replace("_std", ""): value for key, value in result.get("cv_stats", {}).items() if key.endswith("_std")}
            },
            "validation": result.get("validation", {"count": len(y_val), "percentage": len(y_val) / total_records * 100}),
            "test": {"count": len(y_test), "percentage": len(y_test) / total_records * 100, "metrics": _metric_block(metrics)},
            "confusionMatrix": {
                "labels": ["FAKE", "REAL"],
                "matrix": [[metrics.get("tn"), metrics.get("fp")], [metrics.get("fn"), metrics.get("tp")]],
                "truePositive": metrics.get("tp"),
                "trueNegative": metrics.get("tn"),
                "falsePositive": metrics.get("fp"),
                "falseNegative": metrics.get("fn")
            },
            "metrics": _metric_block(metrics),
            "hyperparameters": result.get("hyperparameters", {}),
            "trainingTime": result.get("training_time"),
            "artifactPath": result.get("artifact_path"),
            "predictionArchive": result.get("prediction_archive"),
            "trainingHistory": result.get("training_history")
        }
        models.append(model_entry)

    benchmark = {
        "dataset": {
            "id": dataset,
            "name": dataset.upper(),
            "description": "Binary fake-news classification dataset.",
            "totalRecords": total_records,
            "classes": ["FAKE", "REAL"],
            "classDistribution": class_distribution,
            "labelMapping": label_mapping,
            "preprocessing": preprocessing
        },
        "experiment": {
            "version": experiment_version,
            "trainingTimestamp": timestamp,
            "randomSeed": 42,
            "crossValidation": {"folds": 5, "strategy": "StratifiedKFold"}
        },
        "splits": {
            "training": {"count": len(y_train), "percentage": len(y_train) / total_records * 100, "classDistribution": {"FAKE": int(np.sum(y_train == 0)), "REAL": int(np.sum(y_train == 1))}},
            "validation": {"count": len(y_val), "percentage": len(y_val) / total_records * 100, "classDistribution": {"FAKE": int(np.sum(y_val == 0)), "REAL": int(np.sum(y_val == 1))}},
            "testing": {"count": len(y_test), "percentage": len(y_test) / total_records * 100, "classDistribution": {"FAKE": int(np.sum(y_test == 0)), "REAL": int(np.sum(y_test == 1))}}
        },
        "models": models,
        "ensemble": ensemble or {"status": "unavailable", "reason": "Ensemble metrics were not produced by the executed experiment."},
        "predictionArchive": {"path": f"ml/results/{dataset}/predictions", "count": len(y_test) * sum(model.get("status", "trained") == "trained" for model in models)}
    }
    filepath = get_benchmarks_json_path(dataset)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(benchmark, file, indent=2, allow_nan=False)
    return filepath


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate all required metrics."""
    metrics = {}
    
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="binary", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="binary", zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    
    # ROC-AUC if probabilities available
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["roc_auc"] = None
    
    return metrics


def calculate_cv_statistics(fold_metrics: list):
    """Calculate mean and standard deviation from fold results."""
    if not fold_metrics:
        return {}
    
    keys = fold_metrics[0].keys()
    stats = {}
    
    for key in keys:
        values = [m[key] for m in fold_metrics if m.get(key) is not None]
        if values:
            stats[f"{key}_mean"] = np.mean(values)
            stats[f"{key}_std"] = np.std(values)
    
    return stats


def save_predictions(dataset: str, model_name: str, experiment_version: str,
                     X_test, y_true, y_pred, y_score, predictions_dir: Path):
    """Save raw predictions to JSON lines file."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{experiment_version}_{model_name}_predictions.jsonl"
    filepath = predictions_dir / filename
    
    with open(filepath, "w") as f:
        for i, (text, true_label, pred_label, score) in enumerate(zip(X_test, y_true, y_pred, y_score)):
            record = {
                "dataset": dataset,
                "model": model_name,
                "experiment_version": experiment_version,
                "input_id": i,
                "text": text[:500] if len(text) > 500 else text,
                "prediction": int(pred_label) if isinstance(pred_label, (int, np.integer)) else pred_label,
                "confidence": float(score) if score is not None else None,
                "true_label": int(true_label) if isinstance(true_label, (int, np.integer)) else true_label,
                "correct": bool(pred_label == true_label),
                "timestamp": datetime.now().isoformat()
            }
            f.write(json.dumps(record) + "\n")
    
    return filepath


def write_benchmarks_json(dataset: str, model_name: str, experiment_version: str,
                          cv_metrics: list, cv_stats: dict, test_metrics: dict,
                          config: dict):
    """Write benchmark JSON file."""
    benchmark = {
        "dataset": dataset,
        "model": model_name,
        "experiment_version": experiment_version,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "cv_folds": cv_metrics,
        "cv_statistics": cv_stats,
        "test_metrics": test_metrics
    }
    
    filepath = get_benchmarks_json_path(dataset)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    benchmarks = {}
    if filepath.exists():
        with open(filepath, "r") as f:
            existing = json.load(f)
        if "models" in existing:
            benchmarks = existing
        else:
            benchmarks = {"models": [existing]}

    models = benchmarks.setdefault("models", [])
    models[:] = [item for item in models if item.get("model") != model_name]
    models.append(benchmark)

    # Keep the latest result at the top level for existing consumers.
    benchmarks.update(benchmark)
    
    with open(filepath, "w") as f:
        json.dump(benchmarks, f, indent=2)
    
    return filepath


def write_benchmarks_csv(dataset: str, model_name: str, experiment_version: str,
                         cv_stats: dict, test_metrics: dict):
    """Write benchmark CSV file (summary row)."""
    filepath = get_benchmarks_csv_path(dataset)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    row = {
        "dataset": dataset,
        "model": model_name,
        "experiment_version": experiment_version,
        "timestamp": datetime.now().isoformat(),
        "cv_accuracy_mean": cv_stats.get("accuracy_mean"),
        "cv_accuracy_std": cv_stats.get("accuracy_std"),
        "cv_f1_mean": cv_stats.get("f1_mean"),
        "cv_f1_std": cv_stats.get("f1_std"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "test_tp": test_metrics.get("tp"),
        "test_tn": test_metrics.get("tn"),
        "test_fp": test_metrics.get("fp"),
        "test_fn": test_metrics.get("fn")
    }
    
    file_exists = filepath.exists()
    
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return filepath


def evaluate_model(dataset: str, model_name: str, experiment_version: str,
                   X_test, y_test, y_pred, y_score=None, cv_results=None):
    """
    Main evaluation pipeline.
    
    Args:
        dataset: isot or liar
        model_name: Model identifier
        experiment_version: Version string
        X_test: Test texts
        y_test: True labels
        y_pred: Predicted labels
        y_score: Prediction probabilities/scores
        cv_results: List of fold metrics from CV
    """
    test_metrics = calculate_metrics(y_test, y_pred, y_score)
    
    cv_stats = {}
    if cv_results:
        cv_stats = calculate_cv_statistics(cv_results)
    
    config = {
        "model": model_name,
        "dataset": dataset,
        "note": "Training pending - config to be determined"
    }
    
    predictions_dir = get_predictions_dir(dataset)
    save_predictions(dataset, model_name, experiment_version, X_test, y_test, 
                     y_pred, y_score if y_score is not None else [None]*len(y_pred), predictions_dir)
    
    write_benchmarks_json(dataset, model_name, experiment_version, 
                          cv_results or [], cv_stats, test_metrics, config)
    write_benchmarks_csv(dataset, model_name, experiment_version, 
                         cv_stats, test_metrics)
    
    return test_metrics, cv_stats

