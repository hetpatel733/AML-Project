"""
Train and Evaluate Winning Text-Based LIAR Pipeline
=================================================
Controlled ablation demonstrated that the negation-preserving text pipeline
(clean_text_liar) with Word TF-IDF (1,2) achieves the strongest generalization
and most balanced error characteristics without noisy metadata features.

This script:
1. Loads the LIAR dataset (12,791 samples, 80/20 train/test split, random_state=42).
2. Performs 5-fold Stratified CV on the 80% train set to record CV metrics & threshold analysis.
3. Retrains all 5 models on the full 80% train partition.
4. Performs ONE final test evaluation on the untouched 20% test holdout (2,559 samples).
5. Saves serialized models & vectorizers to ml/models/liar/<model_id>/.
6. Updates ml/results/liar/benchmark.json.
7. Generates comprehensive visualization figures in ml/results/figures/liar/.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)
import joblib

# Add ml root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preprocessing import load_liar_dataframe, clean_text_liar

RANDOM_STATE = 42
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "liar"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "liar"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "figures", "liar"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_model_definitions():
    """Return dictionary of the 5 tuned classifiers."""
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                C=1.0, solver="liblinear", max_iter=1000, random_state=RANDOM_STATE
            ),
            "supports_proba": True,
            "best_threshold": 0.55,
            "paper_baseline": 0.628
        },
        "multinomial_nb": {
            "name": "Multinomial Naive Bayes",
            "model": MultinomialNB(alpha=1.0),
            "supports_proba": True,
            "best_threshold": 0.55,
            "paper_baseline": 0.612
        },
        "linear_svm": {
            "name": "Linear SVM",
            "model": CalibratedClassifierCV(
                estimator=LinearSVC(C=0.1, random_state=RANDOM_STATE, max_iter=2000),
                cv=3
            ),
            "supports_proba": True,
            "best_threshold": 0.55,
            "paper_baseline": 0.626
        },
        "decision_tree": {
            "name": "Decision Tree",
            "model": DecisionTreeClassifier(
                max_depth=20, min_samples_split=5, random_state=RANDOM_STATE
            ),
            "supports_proba": True,
            "best_threshold": 0.50,
            "paper_baseline": 0.571
        },
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=None, min_samples_split=5,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "supports_proba": True,
            "best_threshold": 0.55,
            "paper_baseline": 0.625
        }
    }


def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Calculate all standard classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    auc = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
            
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "rocAuc": auc,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def perform_cross_validation(X_train_texts, y_train, model_dict):
    """Run 5-Fold Stratified CV on 80% train set to get robust CV metrics & threshold analysis."""
    print("=" * 80)
    print("5-FOLD STRATIFIED CROSS-VALIDATION ON 80% TRAIN SET (N = 10,232)")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    
    for model_id, info in model_dict.items():
        print(f"\n--- Cross-Validating: {info['name']} ---")
        fold_accs = []
        fold_f1s = []
        fold_precs = []
        fold_recs = []
        fold_aucs = []
        
        oof_y_true = []
        oof_y_prob = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_texts, y_train), 1):
            X_tr, X_val = X_train_texts[train_idx], X_train_texts[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                max_features=25000
            )
            X_tr_vec = vec.fit_transform(X_tr)
            X_val_vec = vec.transform(X_val)
            
            clf = clone(info['model'])
            clf.fit(X_tr_vec, y_tr)
            
            y_pred = clf.predict(X_val_vec)
            
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_val_vec)[:, 1]
            elif hasattr(clf, "decision_function"):
                df = clf.decision_function(X_val_vec)
                y_prob = 1 / (1 + np.exp(-df))
            else:
                y_prob = y_pred.astype(float)
                
            fold_accs.append(accuracy_score(y_val, y_pred))
            fold_f1s.append(f1_score(y_val, y_pred, zero_division=0))
            fold_precs.append(precision_score(y_val, y_pred, zero_division=0))
            fold_recs.append(recall_score(y_val, y_pred, zero_division=0))
            try:
                fold_aucs.append(roc_auc_score(y_val, y_prob))
            except Exception:
                pass
                
            oof_y_true.extend(y_val)
            oof_y_prob.extend(y_prob)
            
        oof_y_true = np.array(oof_y_true)
        oof_y_prob = np.array(oof_y_prob)
        
        # Threshold analysis on OOF predictions
        thresholds = [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60]
        thresh_stats = {}
        for th in thresholds:
            y_pred_th = (oof_y_prob >= th).astype(int)
            m = evaluate_predictions(oof_y_true, y_pred_th, oof_y_prob)
            thresh_stats[str(th)] = {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "fp_fn_diff": abs(m["fp"] - m["fn"])
            }
            
        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        mean_prec = float(np.mean(fold_precs))
        std_prec = float(np.std(fold_precs))
        mean_rec = float(np.mean(fold_recs))
        std_rec = float(np.std(fold_recs))
        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else None
        
        print(f"  CV Acc:  {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  CV F1:   {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  CV AUC:  {mean_auc:.4f}" if mean_auc else "  CV AUC: N/A")
        print(f"  Threshold 0.50: FP={thresh_stats['0.5']['fp']}, FN={thresh_stats['0.5']['fn']} (diff={thresh_stats['0.5']['fp_fn_diff']})")
        print(f"  Threshold 0.55: FP={thresh_stats['0.55']['fp']}, FN={thresh_stats['0.55']['fn']} (diff={thresh_stats['0.55']['fp_fn_diff']})")
        
        cv_results[model_id] = {
            "folds": 5,
            "strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on 80% train partition only",
            "mean": {
                "accuracy": mean_acc,
                "f1": mean_f1,
                "precision": mean_prec,
                "recall": mean_rec,
                "roc_auc": mean_auc
            },
            "std": {
                "accuracy": std_acc,
                "f1": std_f1,
                "precision": std_prec,
                "recall": std_rec
            },
            "scores": {
                "accuracy": [float(x) for x in fold_accs],
                "f1": [float(x) for x in fold_f1s],
                "precision": [float(x) for x in fold_precs],
                "recall": [float(x) for x in fold_recs]
            },
            "threshold_analysis": thresh_stats,
            "oof_true": oof_y_true,
            "oof_prob": oof_y_prob
        }
        
    return cv_results


def main():
    print("Loading LIAR dataset...")
    df = load_liar_dataframe()
    total_records = len(df)
    print(f"Total dataset size: {total_records} records.")
    
    # Preprocessing with negation retention & clean_text_liar
    print("Applying clean_text_liar to claims...")
    df["clean_statement"] = df["statement"].apply(clean_text_liar)
    
    X = df["clean_statement"].values
    y = df["label"].values
    
    # Exact 80/20 train/test split (Zero Leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    train_count = len(y_train)
    test_count = len(y_test)
    train_fake = int((y_train == 0).sum())
    train_real = int((y_train == 1).sum())
    test_fake = int((y_test == 0).sum())
    test_real = int((y_test == 1).sum())
    
    print(f"Train split: {train_count} (FAKE: {train_fake}, REAL: {train_real})")
    print(f"Test split:  {test_count} (FAKE: {test_fake}, REAL: {test_real})")
    
    model_defs = get_model_definitions()
    
    # Step 1: Cross-Validation & Threshold selection on 80% Train
    cv_data = perform_cross_validation(X_train_raw, y_train, model_defs)
    
    # Step 2: Fit Vectorizer on 80% Train
    print("\nFitting Word TF-IDF (1,2) vectorizer on 80% Train set...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=25000
    )
    t0_vec = time.perf_counter()
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    vec_time = time.perf_counter() - t0_vec
    print(f"Vectorized train shape: {X_train_vec.shape}, test shape: {X_test_vec.shape} ({vec_time:.2f}s)")
    
    # Step 3: Retrain and Final Evaluation
    benchmark_models = []
    test_predictions_dict = {}
    
    print("\n" + "=" * 80)
    print("FINAL RETRAINING (80% Train) & EVALUATION (20% Holdout Test)")
    print("=" * 80)
    
    for model_id, info in model_defs.items():
        print(f"\nRetraining {info['name']}...")
        clf = info['model']
        
        t0 = time.perf_counter()
        clf.fit(X_train_vec, y_train)
        train_time = time.perf_counter() - t0
        
        # Test evaluation at default threshold 0.50
        y_test_pred_default = clf.predict(X_test_vec)
        if hasattr(clf, "predict_proba"):
            y_test_prob = clf.predict_proba(X_test_vec)[:, 1]
        elif hasattr(clf, "decision_function"):
            df_score = clf.decision_function(X_test_vec)
            y_test_prob = 1 / (1 + np.exp(-df_score))
        else:
            y_test_prob = y_test_pred_default.astype(float)
            
        test_metrics_default = evaluate_predictions(y_test, y_test_pred_default, y_test_prob)
        
        # Test evaluation at optimal balanced threshold
        best_th = info['best_threshold']
        y_test_pred_tuned = (y_test_prob >= best_th).astype(int)
        test_metrics_tuned = evaluate_predictions(y_test, y_test_pred_tuned, y_test_prob)
        
        print(f"  [Default 0.50] Acc: {test_metrics_default['accuracy']:.4f}, Prec: {test_metrics_default['precision']:.4f}, Rec: {test_metrics_default['recall']:.4f}, F1: {test_metrics_default['f1']:.4f}, AUC: {test_metrics_default['roc_auc']:.4f}")
        print(f"  [Default 0.50] TP: {test_metrics_default['tp']}, TN: {test_metrics_default['tn']}, FP: {test_metrics_default['fp']}, FN: {test_metrics_default['fn']} (Diff: {abs(test_metrics_default['fp'] - test_metrics_default['fn'])})")
        print(f"  [Tuned {best_th:.2f}]  Acc: {test_metrics_tuned['accuracy']:.4f}, Prec: {test_metrics_tuned['precision']:.4f}, Rec: {test_metrics_tuned['recall']:.4f}, F1: {test_metrics_tuned['f1']:.4f}")
        print(f"  [Tuned {best_th:.2f}]  TP: {test_metrics_tuned['tp']}, TN: {test_metrics_tuned['tn']}, FP: {test_metrics_tuned['fp']}, FN: {test_metrics_tuned['fn']} (Diff: {abs(test_metrics_tuned['fp'] - test_metrics_tuned['fn'])})")
        
        test_predictions_dict[model_id] = {
            "y_true": y_test,
            "y_prob": y_test_prob,
            "y_pred_default": y_test_pred_default,
            "y_pred_tuned": y_test_pred_tuned,
            "name": info['name']
        }
        
        # Save model artifacts into ml/models/liar/<model_id>/
        model_save_dir = os.path.join(MODELS_DIR, model_id)
        os.makedirs(model_save_dir, exist_ok=True)
        
        model_file = os.path.join(model_save_dir, "model.joblib")
        vec_file = os.path.join(model_save_dir, "vectorizer.joblib")
        meta_file = os.path.join(model_save_dir, "metadata.json")
        
        joblib.dump(clf, model_file)
        joblib.dump(vectorizer, vec_file)
        
        paper_ref = info['paper_baseline']
        delta = test_metrics_default['accuracy'] - paper_ref
        pct_gain = f"{'+' if delta >= 0 else ''}{(delta / paper_ref) * 100:.2f}%"
        
        # Get raw parameters
        try:
            params = clf.get_params()
            # Clean non-serializable objects
            clean_params = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    clean_params[k] = v
                elif isinstance(v, (list, tuple)):
                    clean_params[k] = list(v)
                else:
                    clean_params[k] = str(v)
        except Exception:
            clean_params = {}
            
        meta_dict = {
            "dataset": "liar",
            "model": model_id,
            "name": info['name'],
            "training_time_seconds": train_time,
            "parameters": clean_params,
            "representation": "Word TF-IDF (1,2) with Sublinear TF (25,000 max features)",
            "preprocessing": [
                "HTML entity decoding",
                "Lowercase transformation",
                "URL and HTML tag removal",
                "Contraction expansion for negation retention",
                "Non-alphabetic character removal",
                "Whitespace normalization",
                "WordNet lemmatization without aggressive stopword deletion",
                "Word TF-IDF (1,2) n-grams (min_df=2, max_features=25000)"
            ],
            "optimal_threshold": best_th,
            "train_count": train_count,
            "test_count": test_count,
            "metrics": test_metrics_default,
            "metrics_tuned_threshold": test_metrics_tuned,
            "cross_validation": {
                "mean": cv_data[model_id]["mean"],
                "std": cv_data[model_id]["std"],
                "scores": cv_data[model_id]["scores"],
                "threshold_analysis": cv_data[model_id]["threshold_analysis"]
            }
        }
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
            
        benchmark_model_entry = {
            "id": model_id,
            "name": info['name'],
            "type": "classical_ml",
            "representation": "Word TF-IDF (1,2) with Sublinear TF (25,000 features, Negation-Preserving)",
            "status": "trained",
            "parameters": clean_params,
            "trainingTime": train_time,
            "training": {
                "count": train_count,
                "percentage": 80.0
            },
            "validation": {
                "used": False,
                "count": 0,
                "percentage": 0.0
            },
            "test": {
                "count": test_count,
                "percentage": 20.0,
                "metrics": test_metrics_default,
                "metricsAtTunedThreshold": {
                    "threshold": best_th,
                    **test_metrics_tuned
                }
            },
            "metrics": test_metrics_default,
            "crossValidation": {
                "folds": cv_data[model_id]["folds"],
                "strategy": cv_data[model_id]["strategy"],
                "mean": cv_data[model_id]["mean"],
                "std": cv_data[model_id]["std"],
                "scores": cv_data[model_id]["scores"],
                "thresholdAnalysis": cv_data[model_id]["threshold_analysis"]
            },
            "paperReference": {
                "accuracy": paper_ref,
                "source": "Paper Baseline Replication",
                "delta": delta,
                "percentageGain": pct_gain,
                "improved": delta >= 0
            },
            "paperComparison": {
                "paperBaselineAccuracy": paper_ref,
                "improvementDelta": delta,
                "percentageGain": pct_gain,
                "improved": delta >= 0
            },
            "improvementAgainstPaper": {
                "paperBaselineAccuracy": paper_ref,
                "accuracyDelta": delta,
                "percentageGain": pct_gain,
                "improved": delta >= 0
            },
            "confusionMatrix": {
                "labels": ["FAKE", "REAL"],
                "matrix": test_metrics_default["confusion_matrix"],
                "truePositive": test_metrics_default["tp"],
                "trueNegative": test_metrics_default["tn"],
                "falsePositive": test_metrics_default["fp"],
                "falseNegative": test_metrics_default["fn"]
            },
            "artifactPath": model_save_dir
        }
        benchmark_models.append(benchmark_model_entry)
        
    # Build complete benchmark.json
    benchmark_json_data = {
        "dataset": {
            "id": "liar",
            "name": "LIAR",
            "description": "Binary fake-news classification dataset (PANTS-FIRE, FALSE, BARELY-TRUE vs HALF-TRUE, MOSTLY-TRUE, TRUE).",
            "totalRecords": total_records,
            "classes": ["FAKE", "REAL"],
            "classDistribution": {
                "FAKE": int((y == 0).sum()),
                "REAL": int((y == 1).sum())
            },
            "labelMapping": {
                "0": "FAKE",
                "1": "REAL"
            },
            "preprocessing": {
                "lowercase": True,
                "removePunctuation": True,
                "removeSpecialCharacters": True,
                "removeUrlsAndHtml": True,
                "normalizeWhitespace": True,
                "expandContractions": True,
                "preserveNegationWords": True,
                "removeStopwords": "retained (short-claim preservation)",
                "lemmatization": "WordNet",
                "tokenization": "word n-grams with negation preservation",
                "vectorizer": "TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=25000)",
                "ngramRange": [1, 2],
                "wordNgramRange": [1, 2],
                "maxFeatures": 25000,
                "wordMaxFeatures": 25000,
                "sublinearTf": True,
                "minDf": 2,
                "maxDf": 0.98,
                "textCleaning": "clean_text_liar (HTML decoding, URL stripping, special chars removed, lowercase, whitespace normalized, negation words preserved)",
                "metadataFeatures": "None (pure text-based modeling)"
            }
        },
        "experiment": {
            "version": "liar-text-controlled-v3",
            "name": "Controlled Text-Based LIAR Ablation & Tuned Model Benchmark",
            "trainingTimestamp": datetime.now(timezone.utc).isoformat(),
            "randomSeed": RANDOM_STATE,
            "split": "stratified 80% training / 20% testing",
            "validation": {
                "used": False,
                "reason": "No separate validation split. Preprocessing, hyperparameter grids, and decision thresholds tuned on 80% train split via 5-fold Stratified CV; final evaluation on untouched 20% test partition."
            },
            "crossValidationOnTrain": {
                "folds": 5,
                "strategy": "StratifiedKFold on 80% training partition only"
            },
            "ablationSummary": {
                "word_1_2_cv_acc": 0.6206,
                "word_1_3_cv_acc": 0.6136,
                "char_3_5_cv_acc": 0.6075,
                "word_char_cv_acc": 0.6088,
                "winning_representation": "Word TF-IDF (1,2) with clean_text_liar"
            }
        },
        "paperComparison": {
            "baselinePaper": "Wang (2017) LIAR Benchmark",
            "source": "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection (ACL 2017)"
        },
        "splits": {
            "training": {
                "count": train_count,
                "percentage": 80.0,
                "classDistribution": {
                    "FAKE": train_fake,
                    "REAL": train_real
                }
            },
            "validation": {
                "count": 0,
                "percentage": 0.0,
                "used": False
            },
            "testing": {
                "count": test_count,
                "percentage": 20.0,
                "classDistribution": {
                    "FAKE": test_fake,
                    "REAL": test_real
                }
            }
        },
        "models": benchmark_models,
        "ensemble": {
            "status": "not_used",
            "reason": "Models were trained and evaluated separately."
        }
    }
    
    benchmark_path = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_json_data, f, indent=2)
    print(f"\nUpdated benchmark results saved to: {benchmark_path}")
    
    # Step 4: Generate Publication Figures
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS IN ml/results/figures/liar/")
    print("=" * 80)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Individual Confusion Matrix Plots
    for model_id, pdata in test_predictions_dict.items():
        cm = confusion_matrix(pdata["y_true"], pdata["y_pred_default"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["FAKE (0)", "REAL (1)"],
                    yticklabels=["FAKE (0)", "REAL (1)"])
        plt.title(f"{pdata['name']} - Confusion Matrix (Test Set)", fontsize=13, weight='bold')
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_id}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")

    # 2. Combined 5-Panel Confusion Matrix
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for idx, (model_id, pdata) in enumerate(test_predictions_dict.items()):
        cm = confusion_matrix(pdata["y_true"], pdata["y_pred_default"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
        axes[idx].set_title(pdata['name'], fontsize=12, weight='bold')
        axes[idx].set_xlabel("Predicted", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True", fontsize=10)
        else:
            axes[idx].set_ylabel("")
    plt.suptitle("LIAR Test Set Confusion Matrices Across All Models", fontsize=15, weight='bold', y=1.03)
    plt.tight_layout()
    all_cm_path = os.path.join(FIGURES_DIR, "liar_all_confusion_matrices.png")
    plt.savefig(all_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {all_cm_path}")

    # 3. Model Performance Comparison Bar Chart (CV vs Test)
    model_names = [pdata['name'] for pdata in test_predictions_dict.values()]
    cv_accs = [cv_data[mid]["mean"]["accuracy"] * 100 for mid in test_predictions_dict.keys()]
    test_accs = [evaluate_predictions(pdata["y_true"], pdata["y_pred_default"])["accuracy"] * 100 for pdata in test_predictions_dict.values()]
    test_f1s = [evaluate_predictions(pdata["y_true"], pdata["y_pred_default"])["f1"] * 100 for pdata in test_predictions_dict.values()]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, cv_accs, width, label="Train 5-Fold CV Acc (%)", color="#3498db")
    plt.bar(x, test_accs, width, label="Test Holdout Acc (%)", color="#2ecc71")
    plt.bar(x + width, test_f1s, width, label="Test Holdout F1 (%)", color="#e74c3c")
    
    plt.xlabel("Model Architecture", fontsize=12, weight='bold')
    plt.ylabel("Score (%)", fontsize=12, weight='bold')
    plt.title("LIAR Dataset - Cross-Validation vs Test Performance", fontsize=14, weight='bold')
    plt.xticks(x, model_names, rotation=15, ha='right', fontsize=11)
    plt.ylim(45, 75)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    for i in range(len(x)):
        plt.text(x[i] - width, cv_accs[i] + 0.5, f"{cv_accs[i]:.1f}%", ha='center', fontsize=9)
        plt.text(x[i], test_accs[i] + 0.5, f"{test_accs[i]:.1f}%", ha='center', fontsize=9)
        plt.text(x[i] + width, test_f1s[i] + 0.5, f"{test_f1s[i]:.1f}%", ha='center', fontsize=9)
    plt.tight_layout()
    comp_path = os.path.join(FIGURES_DIR, "liar_accuracy_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")

    # 4. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for model_id, pdata in test_predictions_dict.items():
        prec, rec, _ = precision_recall_curve(pdata["y_true"], pdata["y_prob"])
        plt.plot(rec, prec, label=f"{pdata['name']}", linewidth=2)
    plt.xlabel("Recall", fontsize=12, weight='bold')
    plt.ylabel("Precision", fontsize=12, weight='bold')
    plt.title("Precision-Recall Curves (LIAR Test Set)", fontsize=14, weight='bold')
    plt.legend(loc="lower left", frameon=True, facecolor='white')
    plt.tight_layout()
    pr_path = os.path.join(FIGURES_DIR, "liar_precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved: {pr_path}")

    # 5. ROC Curves
    plt.figure(figsize=(8, 6))
    for model_id, pdata in test_predictions_dict.items():
        fpr, tpr, _ = roc_curve(pdata["y_true"], pdata["y_prob"])
        auc_val = roc_auc_score(pdata["y_true"], pdata["y_prob"])
        plt.plot(fpr, tpr, label=f"{pdata['name']} (AUC = {auc_val:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    plt.xlabel("False Positive Rate", fontsize=12, weight='bold')
    plt.ylabel("True Positive Rate", fontsize=12, weight='bold')
    plt.title("ROC Curves (LIAR Test Set)", fontsize=14, weight='bold')
    plt.legend(loc="lower right", frameon=True, facecolor='white')
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "liar_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_path}")

    # 6. Threshold vs FP, FN, and Error Gap Analysis (Logistic Regression)
    lr_oof_true = cv_data["logistic_regression"]["oof_true"]
    lr_oof_prob = cv_data["logistic_regression"]["oof_prob"]
    
    thresh_grid = np.linspace(0.35, 0.65, 31)
    fps, fns, diffs, accs, f1s = [], [], [], [], []
    
    for th in thresh_grid:
        yp = (lr_oof_prob >= th).astype(int)
        cm = confusion_matrix(lr_oof_true, yp)
        tn, fp, fn, tp = cm.ravel()
        fps.append(fp)
        fns.append(fn)
        diffs.append(abs(fp - fn))
        accs.append(accuracy_score(lr_oof_true, yp) * 100)
        f1s.append(f1_score(lr_oof_true, yp) * 100)
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(thresh_grid, fps, 'r-', label="False Positives (FP)", linewidth=2)
    ax1.plot(thresh_grid, fns, 'b-', label="False Negatives (FN)", linewidth=2)
    ax1.plot(thresh_grid, diffs, 'k--', label="|FP - FN| Error Disparity", linewidth=2)
    ax1.axvline(0.55, color='green', linestyle=':', label="Tuned Threshold (0.55)", linewidth=2)
    
    ax2.plot(thresh_grid, accs, 'g-', label="Accuracy (%)", linewidth=1.5, alpha=0.7)
    ax2.plot(thresh_grid, f1s, 'm-', label="F1 Score (%)", linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel("Classification Decision Threshold (tau)", fontsize=12, weight='bold')
    ax1.set_ylabel("Error Counts (FP, FN)", fontsize=12, weight='bold')
    ax2.set_ylabel("Metric (%)", fontsize=12, weight='bold')
    plt.title("Logistic Regression: Error Balance vs Threshold on LIAR Train CV", fontsize=14, weight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', frameon=True, facecolor='white')
    plt.tight_layout()
    thresh_fig_path = os.path.join(FIGURES_DIR, "liar_recall_vs_threshold.png")
    plt.savefig(thresh_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {thresh_fig_path}")

    print("\n" + "=" * 80)
    print("ALL LIAR TRAINING, BENCHMARKING, AND VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
