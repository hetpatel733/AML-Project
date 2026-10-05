import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    BOW_PARAMS,
    TFIDF_PARAMS,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    CV_FOLDS,
    BOW_VECTORIZER_PATH,
    TFIDF_VECTORIZER_PATH,
    VECTORIZER_PATH,
    BOW_LOGISTIC_REGRESSION_PATH,
    LOGISTIC_REGRESSION_PATH,
    NAIVE_BAYES_PATH,
    SVM_PATH,
    RANDOM_FOREST_PATH,
    PASSIVE_AGGRESSIVE_PATH,
    FINAL_MODEL_PATH,
    MODEL_METADATA_PATH,
    METRICS_JSON_PATH,
    METRICS_CSV_PATH
)
from src.data_preprocessing import prepare_and_split_data_3way, clean_text


def train_models():
    """
    Complete Multi-Model ML Training Pipeline:
    1. Loads and cleans raw news data.
    2. Performs 3-way stratified train (70%) / validation (15%) / test (15%) split.
    3. Fits CountVectorizer (BoW) and TfidfVectorizer ONLY on X_train (preventing data leakage).
    4. Trains 6 candidate classifiers:
       - BoW + Logistic Regression
       - TF-IDF + Logistic Regression
       - TF-IDF + Multinomial Naive Bayes
       - TF-IDF + Linear SVM (Calibrated)
       - TF-IDF + Random Forest
       - TF-IDF + Passive Aggressive Classifier (Calibrated)
    5. Computes 5-fold Stratified Cross-Validation on Training set.
    6. Computes Validation set metrics to derive STRICT validation weights (w_i = val_f1).
    7. Evaluates Test set performance for all 6 models + 2 Ensembles (Majority Vote & Val-Weighted).
    8. Serializes all artifacts (Joblib) and metadata (JSON, CSV).
    """
    print("=" * 80)
    print("       FAKE NEWS DETECTION - MULTI-MODEL EXPERIMENTAL TRAINING PIPELINE")
    print("=" * 80)

    # 1. 3-Way Stratified Data Split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, df = prepare_and_split_data_3way(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE
    )

    # 2. Fit Vectorizers ONLY on Training Corpus
    print("\n[Feature Extraction] Fitting CountVectorizer (BoW) on training corpus...")
    bow_vectorizer = CountVectorizer(**BOW_PARAMS)
    X_train_bow = bow_vectorizer.fit_transform(X_train_raw)
    X_val_bow = bow_vectorizer.transform(X_val_raw)
    X_test_bow = bow_vectorizer.transform(X_test_raw)
    joblib.dump(bow_vectorizer, BOW_VECTORIZER_PATH)
    print(f"  -> BoW Vocabulary: {len(bow_vectorizer.vocabulary_)} features | Saved: {BOW_VECTORIZER_PATH}")

    print("\n[Feature Extraction] Fitting TfidfVectorizer (TF-IDF) on training corpus...")
    tfidf_vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
    X_val_tfidf = tfidf_vectorizer.transform(X_val_raw)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    print(f"  -> TF-IDF Vocabulary: {len(tfidf_vectorizer.vocabulary_)} features | Saved: {TFIDF_VECTORIZER_PATH}")

    # 3. Model Definitions
    model_definitions = [
        {
            "id": "bow_lr",
            "name": "BoW + Logistic Regression",
            "description": "Bag-of-Words word occurrence representation paired with L2-regularized Logistic Regression.",
            "feature_type": "bow",
            "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
            "param_grid": {"C": [0.5, 1.0, 2.0]},
            "path": BOW_LOGISTIC_REGRESSION_PATH
        },
        {
            "id": "tfidf_lr",
            "name": "TF-IDF + Logistic Regression",
            "description": "Term Frequency-Inverse Document Frequency sublinear weighting with L2 Logistic Regression.",
            "feature_type": "tfidf",
            "clf": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
            "param_grid": {"C": [0.5, 1.0, 2.0]},
            "path": LOGISTIC_REGRESSION_PATH
        },
        {
            "id": "tfidf_nb",
            "name": "TF-IDF + Multinomial Naive Bayes",
            "description": "Probabilistic generative classifier using Bayes theorem with Laplace smoothing (alpha=0.1).",
            "feature_type": "tfidf",
            "clf": MultinomialNB(alpha=0.1),
            "param_grid": {"alpha": [0.05, 0.1, 0.5]},
            "path": NAIVE_BAYES_PATH
        },
        {
            "id": "tfidf_svm",
            "name": "TF-IDF + Linear SVM",
            "description": "Maximum-margin hyperplane classifier (LinearSVC) calibrated with 3-fold sigmoid scaling.",
            "feature_type": "tfidf",
            "clf": CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
                cv=3
            ),
            "param_grid": {"estimator__C": [0.5, 1.0, 2.0]},
            "path": SVM_PATH
        },
        {
            "id": "tfidf_rf",
            "name": "TF-IDF + Random Forest",
            "description": "Ensemble of 100 decorrelated decision trees with max depth 25 for non-linear boundary capture.",
            "feature_type": "tfidf",
            "clf": RandomForestClassifier(
                n_estimators=100,
                max_depth=25,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "param_grid": {"n_estimators": [100, 200], "max_depth": [None, 25]},
            "path": RANDOM_FOREST_PATH
        },
        {
            "id": "tfidf_pac",
            "name": "TF-IDF + Passive Aggressive Classifier",
            "description": "Online margin-based incremental learning algorithm calibrated with 3-fold cross-validation.",
            "feature_type": "tfidf",
            "clf": CalibratedClassifierCV(
                estimator=PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
                cv=3
            ),
            "param_grid": {"estimator__C": [0.5, 1.0, 2.0]},
            "path": PASSIVE_AGGRESSIVE_PATH
        }
    ]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    model_results = {}
    val_probs = {}
    test_probs = {}
    val_preds = {}
    test_preds = {}

    print("\n" + "=" * 80)
    print("                     TRAINING & CROSS-VALIDATION")
    print("=" * 80)

    for m in model_definitions:
        m_id = m["id"]
        m_name = m["name"]
        feature_type = m["feature_type"]
        clf = m["clf"]
        save_path = m["path"]

        print(f"\n--- [{m_id.upper()}] Training {m_name} ---")

        X_tr = X_train_bow if feature_type == "bow" else X_train_tfidf
        X_v = X_val_bow if feature_type == "bow" else X_val_tfidf
        X_te = X_test_bow if feature_type == "bow" else X_test_tfidf

        print("  [Tuning] Selecting hyperparameters using training data only...")
        tuner = GridSearchCV(
            estimator=clf,
            param_grid=m["param_grid"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )
        clf = tuner.fit(X_tr, y_train).best_estimator_
        print(f"  [Tuning] Best parameters: {tuner.best_params_}")

        # 5-Fold Stratified Cross-Validation on Training Data
        print(f"  [CV] Running {CV_FOLDS}-fold stratified cross-validation...")
        cv_scores = cross_validate(
            clf,
            X_tr,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            n_jobs=-1,
            error_score="raise"
        )
        cv_metrics = {
            metric: {
                "mean": round(float(np.mean(cv_scores[f"test_{metric}"])), 4),
                "std": round(float(np.std(cv_scores[f"test_{metric}"])), 4),
                "folds": [round(float(score), 4) for score in cv_scores[f"test_{metric}"]]
            }
            for metric in ["accuracy", "precision", "recall", "f1"]
        }
        cv_mean_f1 = cv_metrics["f1"]["mean"]
        cv_std_f1 = cv_metrics["f1"]["std"]
        print(f"  [CV] 5-Fold Mean F1: {cv_mean_f1 * 100:.2f}% (+/- {cv_std_f1 * 100:.2f}%)")

        # Fit on full training set
        print("  [Fit] Fitting model on full training set (70%)...")
        clf.fit(X_tr, y_train)

        # Validation set evaluation (Strictly used for ensemble weighting)
        val_pred = clf.predict(X_v)
        val_preds[m_id] = val_pred

        if hasattr(clf, "predict_proba"):
            val_prob = clf.predict_proba(X_v)[:, 1]
            test_prob = clf.predict_proba(X_te)[:, 1]
        elif hasattr(clf, "decision_function"):
            decision_v = clf.decision_function(X_v)
            val_prob = 1.0 / (1.0 + np.exp(-decision_v))
            decision_te = clf.decision_function(X_te)
            test_prob = 1.0 / (1.0 + np.exp(-decision_te))
        else:
            val_prob = val_pred.astype(float)
            test_prob = clf.predict(X_te).astype(float)

        val_probs[m_id] = val_prob
        test_probs[m_id] = test_prob

        # Val metrics
        val_acc = float(accuracy_score(y_val, val_pred))
        val_prec = float(precision_score(y_val, val_pred, zero_division=0))
        val_rec = float(recall_score(y_val, val_pred, zero_division=0))
        val_f1 = float(f1_score(y_val, val_pred, zero_division=0))
        val_roc = float(roc_auc_score(y_val, val_prob))
        val_cm = confusion_matrix(y_val, val_pred).tolist()

        # Test set evaluation
        test_pred = clf.predict(X_te)
        test_preds[m_id] = test_pred

        test_acc = float(accuracy_score(y_test, test_pred))
        test_prec = float(precision_score(y_test, test_pred, zero_division=0))
        test_rec = float(recall_score(y_test, test_pred, zero_division=0))
        test_f1 = float(f1_score(y_test, test_pred, zero_division=0))
        test_roc = float(roc_auc_score(y_test, test_prob))
        test_cm = confusion_matrix(y_test, test_pred).tolist()

        print(f"  [Validation] Acc: {val_acc*100:.2f}% | Prec: {val_prec*100:.2f}% | Rec: {val_rec*100:.2f}% | F1: {val_f1*100:.2f}% | ROC-AUC: {val_roc:.4f}")
        print(f"  [Test Split] Acc: {test_acc*100:.2f}% | Prec: {test_prec*100:.2f}% | Rec: {test_rec*100:.2f}% | F1: {test_f1*100:.2f}% | ROC-AUC: {test_roc:.4f}")

        # Save model artifact
        joblib.dump(clf, save_path)
        print(f"  [Persistence] Saved to: {save_path}")

        model_results[m_id] = {
            "id": m_id,
            "name": m_name,
            "description": m["description"],
            "feature_type": feature_type,
            "selected_hyperparameters": tuner.best_params_,
            "cv_f1_mean": round(cv_mean_f1, 4),
            "cv_f1_std": round(cv_std_f1, 4),
            "cv_metrics": {
                "accuracy_mean": cv_metrics["accuracy"]["mean"],
                "accuracy_std": cv_metrics["accuracy"]["std"],
                "precision_mean": cv_metrics["precision"]["mean"],
                "precision_std": cv_metrics["precision"]["std"],
                "recall_mean": cv_metrics["recall"]["mean"],
                "recall_std": cv_metrics["recall"]["std"],
                "f1_mean": cv_metrics["f1"]["mean"],
                "f1_std": cv_metrics["f1"]["std"]
            },
            "val_metrics": {
                "accuracy": round(val_acc, 4),
                "precision": round(val_prec, 4),
                "recall": round(val_rec, 4),
                "f1_score": round(val_f1, 4),
                "roc_auc": round(val_roc, 4),
                "confusion_matrix": val_cm
            },
            "test_metrics": {
                "accuracy": round(test_acc, 4),
                "precision": round(test_prec, 4),
                "recall": round(test_rec, 4),
                "f1_score": round(test_f1, 4),
                "roc_auc": round(test_roc, 4),
                "confusion_matrix": test_cm
            },
            # Validation F1 score for ensemble weight w_i
            "validation_weight_raw": round(val_f1, 4)
        }

    # 4. Calculate Validation Weights for Validation-Weighted Ensemble
    print("\n" + "=" * 80)
    print("                  ENSEMBLE WEIGHT COMPUTATION (STRICT VAL-F1)")
    print("=" * 80)
    raw_weights = {m_id: model_results[m_id]["val_metrics"]["f1_score"] for m_id in model_results}
    sum_weights = sum(raw_weights.values())
    normalized_weights = {m_id: round(w / sum_weights, 4) for m_id, w in raw_weights.items()}

    for m_id, norm_w in normalized_weights.items():
        model_results[m_id]["ensemble_weight_normalized"] = norm_w
        print(f"  Model '{m_id}': Raw Val-F1={raw_weights[m_id]:.4f} -> Normalized Weight={norm_w:.4f} ({norm_w*100:.1f}%)")

    # 5. Evaluate Ensembles on Validation and Test Sets
    # 5a. Validation Weighted Ensemble
    val_weighted_probs = np.zeros(len(y_val))
    test_weighted_probs = np.zeros(len(y_test))
    for m_id, w in normalized_weights.items():
        val_weighted_probs += w * val_probs[m_id]
        test_weighted_probs += w * test_probs[m_id]

    val_weighted_preds = (val_weighted_probs >= 0.5).astype(int)
    test_weighted_preds = (test_weighted_probs >= 0.5).astype(int)

    val_weighted_metrics = {
        "accuracy": round(float(accuracy_score(y_val, val_weighted_preds)), 4),
        "precision": round(float(precision_score(y_val, val_weighted_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, val_weighted_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_val, val_weighted_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, val_weighted_probs)), 4),
        "confusion_matrix": confusion_matrix(y_val, val_weighted_preds).tolist()
    }

    test_weighted_metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_weighted_preds)), 4),
        "precision": round(float(precision_score(y_test, test_weighted_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_weighted_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, test_weighted_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_weighted_probs)), 4),
        "confusion_matrix": confusion_matrix(y_test, test_weighted_preds).tolist()
    }

    # 5b. Majority Voting Ensemble
    val_vote_matrix = np.column_stack([val_preds[m_id] for m_id in model_results])
    test_vote_matrix = np.column_stack([test_preds[m_id] for m_id in model_results])

    # Real class votes count (1)
    val_maj_votes = np.sum(val_vote_matrix, axis=1)
    test_maj_votes = np.sum(test_vote_matrix, axis=1)

    val_maj_preds = (val_maj_votes >= 3).astype(int) # 3 or more out of 6 votes for REAL
    test_maj_preds = (test_maj_votes >= 3).astype(int)
    val_maj_probs = val_maj_votes / len(model_results)
    test_maj_probs = test_maj_votes / len(model_results)

    val_maj_metrics = {
        "accuracy": round(float(accuracy_score(y_val, val_maj_preds)), 4),
        "precision": round(float(precision_score(y_val, val_maj_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, val_maj_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_val, val_maj_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, val_maj_probs)), 4),
        "confusion_matrix": confusion_matrix(y_val, val_maj_preds).tolist()
    }

    test_maj_metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_maj_preds)), 4),
        "precision": round(float(precision_score(y_test, test_maj_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_maj_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, test_maj_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_maj_probs)), 4),
        "confusion_matrix": confusion_matrix(y_test, test_maj_preds).tolist()
    }

    print("\n[Ensemble Test Evaluation]")
    print(f"  -> Validation-Weighted Ensemble: Acc={test_weighted_metrics['accuracy']*100:.2f}%, F1={test_weighted_metrics['f1_score']*100:.2f}%, ROC={test_weighted_metrics['roc_auc']:.4f}")
    print(f"  -> Majority Voting Ensemble:    Acc={test_maj_metrics['accuracy']*100:.2f}%, F1={test_maj_metrics['f1_score']*100:.2f}%, ROC={test_maj_metrics['roc_auc']:.4f}")

    # 6. Model Selection: Select top-performing individual model and also save final model
    best_single_id = max(model_results.keys(), key=lambda k: model_results[k]["val_metrics"]["f1_score"])
    best_single_name = model_results[best_single_id]["name"]
    best_single_f1 = model_results[best_single_id]["val_metrics"]["f1_score"]
    print(f"\n[Model Selection] Best Individual Model on Validation: '{best_single_name}' (Val F1: {best_single_f1*100:.2f}%)")

    # Save best single model to FINAL_MODEL_PATH
    best_clf_obj = joblib.load(model_definitions[[m["id"] for m in model_definitions].index(best_single_id)]["path"])
    joblib.dump(best_clf_obj, FINAL_MODEL_PATH)
    print(f"[Model Selection] Persisted champion model to: {FINAL_MODEL_PATH}")

    # 7. Comprehensive Model Config & Experiment Metadata
    metadata = {
        "dataset_metadata": {
            "total_samples": len(df),
            "train_samples": len(X_train_raw),
            "val_samples": len(X_val_raw),
            "test_samples": len(X_test_raw),
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_state": RANDOM_STATE,
            "bow_vocabulary_size": len(bow_vectorizer.vocabulary_),
            "tfidf_vocabulary_size": len(tfidf_vectorizer.vocabulary_)
        },
        "experiment": {
            "version": "v2",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "ISOT Fake and Real News files available in ml/data",
            "external_validation": {
                "status": "not_performed",
                "dataset": None,
                "reason": "No independent, compatible fake/real labeled dataset is included in the repository."
            },
            "leakage_controls": [
                "Stratified split before vectorizer fitting",
                "Vectorizers fitted on training text only",
                "Ensemble weights derived from validation F1 only",
                "Final test set used only for evaluation"
            ]
        },
        "data_quality": {
            "source_rows": int(df.attrs.get("source_rows", len(df))),
            "processed_rows": int(len(df)),
            "empty_clean_text_rows_removed": int(df.attrs.get("empty_clean_text_rows_removed", 0)),
            "duplicate_rows_removed": int(df.attrs.get("duplicate_rows_removed", 0)),
            "missing_title_count": int(df.attrs.get("missing_title_count", 0)),
            "missing_text_count": int(df.attrs.get("missing_text_count", 0)),
            "class_distribution": {
                str(label): int(count)
                for label, count in df["label_name"].value_counts().items()
            }
        },
        "models": model_results,
        "ensemble_weights": {
            "raw_validation_f1": raw_weights,
            "normalized_weights": normalized_weights
        },
        "ensembles": {
            "validation_weighted": {
                "name": "Validation-Weighted Soft Ensemble",
                "description": "Weighted probability aggregation using validation F1 scores as weights: P_ens = sum(w_i * P_i) / sum(w_i).",
                "val_metrics": val_weighted_metrics,
                "test_metrics": test_weighted_metrics
            },
            "majority_voting": {
                "name": "Majority Voting Hard Ensemble",
                "description": "Democratic consensus where each model casts one discrete prediction vote. Class with >= 50% votes wins.",
                "val_metrics": val_maj_metrics,
                "test_metrics": test_maj_metrics
            }
        },
        "champion_model": {
            "id": best_single_id,
            "name": best_single_name,
            "val_f1": best_single_f1,
            "test_metrics": model_results[best_single_id]["test_metrics"]
        },
        "hyperparameters": {
            "bow": BOW_PARAMS,
            "tfidf": TFIDF_PARAMS,
            "cv_folds": CV_FOLDS
        }
    }

    # Write model config and metrics JSON
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Metadata] Exported configuration to: {MODEL_METADATA_PATH}")

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] Exported metrics to: {METRICS_JSON_PATH}")

    # Export comparison CSV for academic reporting
    comparison_rows = []
    for m_id, m_data in model_results.items():
        comparison_rows.append({
            "Model": m_data["name"],
            "Type": "Individual",
            "Representation": m_data["feature_type"].upper(),
            "CV 5-Fold F1 (%)": f"{m_data['cv_f1_mean']*100:.2f} +/- {m_data['cv_f1_std']*100:.2f}",
            "Val F1 (%)": f"{m_data['val_metrics']['f1_score']*100:.2f}",
            "Test Accuracy (%)": f"{m_data['test_metrics']['accuracy']*100:.2f}",
            "Test Precision (%)": f"{m_data['test_metrics']['precision']*100:.2f}",
            "Test Recall (%)": f"{m_data['test_metrics']['recall']*100:.2f}",
            "Test F1 (%)": f"{m_data['test_metrics']['f1_score']*100:.2f}",
            "Test ROC-AUC": f"{m_data['test_metrics']['roc_auc']:.4f}",
            "Normalized Ensemble Weight (%)": f"{normalized_weights[m_id]*100:.2f}"
        })

    # Add Ensembles to CSV
    comparison_rows.append({
        "Model": "Majority Voting Ensemble (6 Models)",
        "Type": "Ensemble (Hard)",
        "Representation": "BoW + TF-IDF Hybrid",
        "CV 5-Fold F1 (%)": "N/A",
        "Val F1 (%)": f"{val_maj_metrics['f1_score']*100:.2f}",
        "Test Accuracy (%)": f"{test_maj_metrics['accuracy']*100:.2f}",
        "Test Precision (%)": f"{test_maj_metrics['precision']*100:.2f}",
        "Test Recall (%)": f"{test_maj_metrics['recall']*100:.2f}",
        "Test F1 (%)": f"{test_maj_metrics['f1_score']*100:.2f}",
        "Test ROC-AUC": f"{test_maj_metrics['roc_auc']:.4f}",
        "Normalized Ensemble Weight (%)": "N/A"
    })

    comparison_rows.append({
        "Model": "Validation-Weighted Ensemble (6 Models)",
        "Type": "Ensemble (Soft)",
        "Representation": "BoW + TF-IDF Hybrid",
        "CV 5-Fold F1 (%)": "N/A",
        "Val F1 (%)": f"{val_weighted_metrics['f1_score']*100:.2f}",
        "Test Accuracy (%)": f"{test_weighted_metrics['accuracy']*100:.2f}",
        "Test Precision (%)": f"{test_weighted_metrics['precision']*100:.2f}",
        "Test Recall (%)": f"{test_weighted_metrics['recall']*100:.2f}",
        "Test F1 (%)": f"{test_weighted_metrics['f1_score']*100:.2f}",
        "Test ROC-AUC": f"{test_weighted_metrics['roc_auc']:.4f}",
        "Normalized Ensemble Weight (%)": "100.00"
    })

    pd.DataFrame(comparison_rows).to_csv(METRICS_CSV_PATH, index=False)
    print(f"[Metadata] Exported comparison table to: {METRICS_CSV_PATH}")

    # Save evaluation split references
    joblib.dump(X_test_raw, DATA_DIR / "X_test_raw.joblib")
    np.save(DATA_DIR / "y_test.npy", y_test)
    joblib.dump(X_val_raw, DATA_DIR / "X_val_raw.joblib")
    np.save(DATA_DIR / "y_val.npy", y_val)

    print("\n" + "=" * 80)
    print("      ALL 6 MODELS & DUAL ENSEMBLES TRAINED & PERSISTED SUCCESSFULLY!")
    print("=" * 80)
    return metadata


if __name__ == "__main__":
    train_models()
