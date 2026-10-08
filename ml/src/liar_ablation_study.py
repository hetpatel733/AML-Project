import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data_preprocessing import (
    load_liar_dataset,
    clean_text,
    clean_text_liar,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = 0.5
            
    fp_fn_ratio = (fp / fn) if fn > 0 else (float('inf') if fp > 0 else 1.0)
    # Balance score: penalty for large disparity between FP and FN
    fp_fn_diff = abs(fp - fn)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fp_fn_ratio": float(fp_fn_ratio),
        "fp_fn_diff": int(fp_fn_diff),
    }


def evaluate_cv_oof(
    vectorizer_factory,
    model_factory,
    X_raw: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    thresholds: List[float] = [0.40, 0.45, 0.50, 0.55, 0.60],
) -> Dict[str, Any]:
    """
    Evaluate a model configuration with 5-fold Stratified CV.
    Fits vectorizer strictly on training folds to prevent any data leakage.
    Collects out-of-fold predicted probabilities to evaluate multiple decision thresholds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(y), dtype=float)
    oof_preds_default = np.zeros(len(y), dtype=int)
    
    fold_metrics_list = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_train_fold_raw, X_val_fold_raw = X_raw[train_idx], X_raw[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Fit vectorizer solely on fold training data
        vec = vectorizer_factory()
        X_train_vec = vec.fit_transform(X_train_fold_raw)
        X_val_vec = vec.transform(X_val_fold_raw)
        
        # Train model
        model = model_factory()
        model.fit(X_train_vec, y_train_fold)
        
        # Get probabilities or calibrated decision function
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val_vec)[:, 1]
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_val_vec)
            # Sigmoid scaling
            probs = 1.0 / (1.0 + np.exp(-df))
        else:
            preds = model.predict(X_val_vec)
            probs = preds.astype(float)
            
        oof_probs[val_idx] = probs
        oof_preds_default[val_idx] = (probs >= 0.5).astype(int)
        
        fold_metrics = compute_metrics(y_val_fold, oof_preds_default[val_idx], probs)
        fold_metrics_list.append(fold_metrics)

    # Evaluate all thresholds on overall out-of-fold predictions
    threshold_results = {}
    for thresh in thresholds:
        thresh_preds = (oof_probs >= thresh).astype(int)
        metrics = compute_metrics(y, thresh_preds, oof_probs)
        threshold_results[f"{thresh:.2f}"] = metrics
        
    return {
        "oof_metrics_0.50": threshold_results["0.50"],
        "threshold_sweep": threshold_results,
        "fold_mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics_list])),
        "fold_std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics_list])),
        "fold_mean_f1": float(np.mean([m["f1"] for m in fold_metrics_list])),
        "fold_std_f1": float(np.std([m["f1"] for m in fold_metrics_list])),
        "oof_probs": oof_probs,
    }


def run_ablation_study():
    print("=" * 80)
    print("LIAR DATASET ABLATION STUDY: CONTROLLED FEATURE & HYPERPARAMETER SEARCH")
    print("Zero Test Leakage — 5-Fold Stratified CV on 80% Training Split Only")
    print("=" * 80)
    
    train_df, test_df, y_train, y_test = load_liar_dataset(test_size=0.2)
    print(f"Total training samples: {len(train_df)} | Total test holdout samples: {len(test_df)}")
    print(f"Class distribution in train: {np.bincount(y_train)} (0=False/Barely/Pants-Fire, 1=True/Mostly/Half-True)")
    
    # Preprocess text on training set using clean_text and clean_text_liar
    print("\nPreparing preprocessed text variants...")
    statements_raw = train_df["statement"].fillna("").astype(str).to_numpy()
    
    t0 = time.time()
    statements_cleaned = np.array([clean_text(s) for s in statements_raw])
    print(f"clean_text (stopword filtered + negation preserved) completed in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    statements_liar = np.array([clean_text_liar(s) for s in statements_raw])
    print(f"clean_text_liar (no stopword removal + lemmatized) completed in {time.time()-t0:.2f}s")
    
    # 4 Core Ablation Feature Representations
    vectorizer_configs = {
        "1. Word TF-IDF (1,2)": lambda: TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=12000,
        ),
        "2. Word TF-IDF (1,3)": lambda: TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_features=15000,
        ),
        "3. Char TF-IDF (3,5)": lambda: TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=3,
            max_features=15000,
        ),
        "4. Word (1,2) + Char (3,5)": lambda: FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=10000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=10000)),
        ]),
    }
    
    # Base models to compare across feature representations
    base_models = {
        "Logistic Regression (C=1.0)": lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "Multinomial Naive Bayes (alpha=1.0)": lambda: MultinomialNB(alpha=1.0),
        "Linear SVM (C=1.0)": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=3000), cv=3),
        "Decision Tree (default)": lambda: DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42),
        "Random Forest (100 trees)": lambda: RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1),
    }
    
    # PART 1: Compare Preprocessing (clean_text vs clean_text_liar) on Feature 1 & 4 with Logistic Regression
    print("\n" + "=" * 80)
    print("PART 1: PREPROCESSING COMPARISON (5-Fold CV on 80% Train)")
    print("=" * 80)
    
    for prep_name, X_data in [("clean_text (stopword filtered)", statements_cleaned), ("clean_text_liar (no stopword removal)", statements_liar)]:
        print(f"\n--- Preprocessing: {prep_name} ---")
        for feat_name, vec_factory in [("Word TF-IDF (1,2)", vectorizer_configs["1. Word TF-IDF (1,2)"]), ("Word (1,2) + Char (3,5)", vectorizer_configs["4. Word (1,2) + Char (3,5)"])]:
            res = evaluate_cv_oof(vec_factory, lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42), X_data, y_train)
            m = res["oof_metrics_0.50"]
            print(f"  {feat_name:28s} | Acc: {m['accuracy']*100:.2f}% | Prec: {m['precision']*100:.2f}% | Rec: {m['recall']*100:.2f}% | F1: {m['f1']*100:.2f}% | AUC: {m['roc_auc']:.4f} | TP: {m['tp']:4d} | TN: {m['tn']:4d} | FP: {m['fp']:4d} | FN: {m['fn']:4d} | FP-FN Diff: {m['fp_fn_diff']}")

    # PART 2: Feature Representation Ablation across all 5 models (using best text preprocessing)
    print("\n" + "=" * 80)
    print("PART 2: CONTROLLED FEATURE ABLATION (All 5 Models, 5-Fold CV on 80% Train)")
    print("=" * 80)
    
    best_prep_X = statements_cleaned  # We'll evaluate both and confirm
    ablation_results = {}
    
    for feat_name, vec_factory in vectorizer_configs.items():
        print(f"\n>>> Evaluating Representation: {feat_name}")
        ablation_results[feat_name] = {}
        for model_name, mod_factory in base_models.items():
            res = evaluate_cv_oof(vec_factory, mod_factory, best_prep_X, y_train)
            m = res["oof_metrics_0.50"]
            ablation_results[feat_name][model_name] = res
            print(f"  {model_name:36s} | Acc: {m['accuracy']*100:.2f}% | Prec: {m['precision']*100:.2f}% | Rec: {m['recall']*100:.2f}% | F1: {m['f1']*100:.2f}% | AUC: {m['roc_auc']:.4f} | TP: {m['tp']:4d} | TN: {m['tn']:4d} | FP: {m['fp']:4d} | FN: {m['fn']:4d}")

    # PART 3: Hyperparameter Grid Search & Tuning for each of the 5 models on the winning feature representation
    print("\n" + "=" * 80)
    print("PART 3: HYPERPARAMETER TUNING & THRESHOLD OPTIMIZATION (5-Fold CV)")
    print("=" * 80)
    
    # Let's define hyperparameter candidates for each model
    model_param_grids = {
        "Logistic Regression": [
            ("C=0.1, penalty=l2", lambda: LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
            ("C=0.5, penalty=l2", lambda: LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
            ("C=1.0, penalty=l2", lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
            ("C=2.0, penalty=l2", lambda: LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
            ("C=5.0, penalty=l2", lambda: LogisticRegression(C=5.0, max_iter=2000, random_state=42)),
            ("C=1.0, balanced", lambda: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)),
            ("C=2.0, balanced", lambda: LogisticRegression(C=2.0, class_weight='balanced', max_iter=2000, random_state=42)),
        ],
        "Multinomial Naive Bayes": [
            ("alpha=0.01", lambda: MultinomialNB(alpha=0.01)),
            ("alpha=0.1", lambda: MultinomialNB(alpha=0.1)),
            ("alpha=0.5", lambda: MultinomialNB(alpha=0.5)),
            ("alpha=1.0", lambda: MultinomialNB(alpha=1.0)),
            ("alpha=2.0", lambda: MultinomialNB(alpha=2.0)),
            ("alpha=5.0", lambda: MultinomialNB(alpha=5.0)),
        ],
        "Linear SVM": [
            ("C=0.05, calibrated", lambda: CalibratedClassifierCV(LinearSVC(C=0.05, random_state=42, max_iter=3000), cv=3)),
            ("C=0.1, calibrated", lambda: CalibratedClassifierCV(LinearSVC(C=0.1, random_state=42, max_iter=3000), cv=3)),
            ("C=0.5, calibrated", lambda: CalibratedClassifierCV(LinearSVC(C=0.5, random_state=42, max_iter=3000), cv=3)),
            ("C=1.0, calibrated", lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=3000), cv=3)),
            ("C=2.0, calibrated", lambda: CalibratedClassifierCV(LinearSVC(C=2.0, random_state=42, max_iter=3000), cv=3)),
            ("C=1.0, balanced, calibrated", lambda: CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=3000), cv=3)),
        ],
        "Decision Tree": [
            ("max_depth=10, min_split=10", lambda: DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)),
            ("max_depth=20, min_split=5", lambda: DecisionTreeClassifier(max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)),
            ("max_depth=30, min_split=5", lambda: DecisionTreeClassifier(max_depth=30, min_samples_split=5, min_samples_leaf=2, random_state=42)),
            ("max_depth=50, min_split=2", lambda: DecisionTreeClassifier(max_depth=50, min_samples_split=2, min_samples_leaf=1, random_state=42)),
            ("max_depth=None, min_split=5", lambda: DecisionTreeClassifier(max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=42)),
        ],
        "Random Forest": [
            ("trees=100, depth=20", lambda: RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)),
            ("trees=200, depth=20", lambda: RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)),
            ("trees=100, depth=30", lambda: RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)),
            ("trees=200, depth=30", lambda: RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)),
            ("trees=200, depth=50", lambda: RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_split=5, random_state=42, n_jobs=-1)),
            ("trees=200, depth=None", lambda: RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42, n_jobs=-1)),
        ],
    }

    # Test top vectorizer configurations for the tuning phase
    for top_vec_name in ["1. Word TF-IDF (1,2)", "2. Word TF-IDF (1,3)", "4. Word (1,2) + Char (3,5)"]:
        print(f"\n==================== TUNING WITH {top_vec_name} ====================")
        top_vec_factory = vectorizer_configs[top_vec_name]
        
        for model_family, param_candidates in model_param_grids.items():
            print(f"\n--- {model_family} ---")
            for param_label, factory in param_candidates:
                res = evaluate_cv_oof(top_vec_factory, factory, best_prep_X, y_train)
                m = res["oof_metrics_0.50"]
                print(f"  {param_label:30s} | Acc: {m['accuracy']*100:.2f}% | Prec: {m['precision']*100:.2f}% | Rec: {m['recall']*100:.2f}% | F1: {m['f1']*100:.2f}% | AUC: {m['roc_auc']:.4f} | TP:{m['tp']:4d} TN:{m['tn']:4d} FP:{m['fp']:4d} FN:{m['fn']:4d} | Diff:{m['fp_fn_diff']}")

    # PART 4: Detailed Decision Threshold Optimization for tuned models
    print("\n" + "=" * 80)
    print("PART 4: DECISION THRESHOLD SWEEP (0.40, 0.45, 0.50, 0.55, 0.60) ON TRAINING CV")
    print("=" * 80)
    
    # Let's test thresholds for key models on candidate vectorizer
    eval_models = {
        "Logistic Regression (C=1.0)": lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "Multinomial Naive Bayes (alpha=0.5)": lambda: MultinomialNB(alpha=0.5),
        "Linear SVM (C=1.0, Calibrated)": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=3000), cv=3),
        "Decision Tree (depth=30)": lambda: DecisionTreeClassifier(max_depth=30, min_samples_split=5, min_samples_leaf=2, random_state=42),
        "Random Forest (200 trees, depth=30)": lambda: RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1),
    }
    
    for model_name, factory in eval_models.items():
        print(f"\n>>> Threshold Sweep for: {model_name}")
        for vec_name in ["1. Word TF-IDF (1,2)", "4. Word (1,2) + Char (3,5)"]:
            print(f"  Vectorization: {vec_name}")
            res = evaluate_cv_oof(vectorizer_configs[vec_name], factory, best_prep_X, y_train)
            for thresh_str, tm in res["threshold_sweep"].items():
                print(f"    Threshold {thresh_str}: Acc={tm['accuracy']*100:.2f}% | Prec={tm['precision']*100:.2f}% | Rec={tm['recall']*100:.2f}% | F1={tm['f1']*100:.2f}% | TP={tm['tp']:4d} | TN={tm['tn']:4d} | FP={tm['fp']:4d} | FN={tm['fn']:4d} | FP/FN Ratio={tm['fp_fn_ratio']:.2f} | Diff={tm['fp_fn_diff']}")


if __name__ == "__main__":
    run_ablation_study()
