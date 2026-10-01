import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    VECTORIZER_PATH,
    LOGISTIC_REGRESSION_PATH,
    NAIVE_BAYES_PATH,
    SVM_PATH,
    FINAL_MODEL_PATH,
    METRICS_JSON_PATH,
    METRICS_CSV_PATH,
    CONFUSION_MATRIX_PNG,
    MODEL_COMPARISON_PNG,
    ROC_CURVE_PNG,
    PRECISION_RECALL_PNG,
    LABEL_MAPPING
)
from src.data_preprocessing import prepare_and_split_data, clean_text


# Set publication-quality visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def run_full_evaluation():
    """
    Comprehensive Academic Model Evaluation:
    - Measures Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices.
    - Saves metrics.json and metrics_comparison.csv.
    - Generates and saves 4 publication-quality visualization figures.
    """
    print("=" * 75)
    print("       FAKE NEWS DETECTION - ACADEMIC MODEL EVALUATION")
    print("=" * 75)

    # 1. Load Vectorizer and Test Data
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Run train.py first!")

    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Reload test set
    if (DATA_DIR / "X_test_raw.joblib").exists() and (DATA_DIR / "y_test.npy").exists():
        X_test_raw = joblib.load(DATA_DIR / "X_test_raw.joblib")
        y_test = np.load(DATA_DIR / "y_test.npy")
        X_test_vec = vectorizer.transform(X_test_raw)
    else:
        _, X_test_raw, _, y_test, _ = prepare_and_split_data()
        X_test_vec = vectorizer.transform(X_test_raw)

    # 2. Define Models to Evaluate
    model_paths = {
        "Logistic Regression": LOGISTIC_REGRESSION_PATH,
        "Multinomial Naive Bayes": NAIVE_BAYES_PATH,
        "Support Vector Machine": SVM_PATH,
        "Random Forest": MODELS_DIR / "random_forest.joblib"
    }

    metrics_dict = {}
    curves_data = {}
    confusion_matrices = {}

    for name, path in model_paths.items():
        if not path.exists():
            print(f"[Warning] Model file not found: {path}. Skipping...")
            continue

        print(f"\n[Evaluating] Processing: {name}")
        clf = joblib.load(path)

        # Predictions
        y_pred = clf.predict(X_test_vec)

        # Probability scores for label 1 (REAL)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_vec)[:, 1]
        elif hasattr(clf, "decision_function"):
            decision = clf.decision_function(X_test_vec)
            y_proba = 1 / (1 + np.exp(-decision))
        else:
            y_proba = y_pred

        # Quantitative Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cls_rep = classification_report(
            y_test,
            y_pred,
            target_names=["FAKE (0)", "REAL (1)"],
            output_dict=True,
            zero_division=0
        )

        metrics_dict[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "total_test_samples": int(len(y_test))
            },
            "classification_report": cls_rep
        }

        confusion_matrices[name] = cm
        curves_data[name] = {
            "y_proba": y_proba,
            "roc_auc": roc_auc
        }

        print(f"  -> Accuracy:  {acc * 100:.2f}%")
        print(f"  -> Precision: {prec * 100:.2f}%")
        print(f"  -> Recall:    {rec * 100:.2f}%")
        print(f"  -> F1-Score:  {f1 * 100:.2f}%")
        print(f"  -> ROC-AUC:   {roc_auc:.4f}")
        print(f"  -> Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 3. Save JSON Metrics
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n[Persistence] Metrics JSON saved to: {METRICS_JSON_PATH}")

    # 4. Save Tabular CSV Metrics
    csv_rows = []
    for name, data in metrics_dict.items():
        csv_rows.append({
            "Model": name,
            "Accuracy": data["accuracy"],
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1-Score": data["f1_score"],
            "ROC-AUC": data["roc_auc"],
            "True Negatives": data["confusion_matrix"]["true_negative"],
            "False Positives": data["confusion_matrix"]["false_positive"],
            "False Negatives": data["confusion_matrix"]["false_negative"],
            "True Positives": data["confusion_matrix"]["true_positive"]
        })
    metrics_df = pd.DataFrame(csv_rows)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"[Persistence] Comparison CSV saved to: {METRICS_CSV_PATH}")

    # 5. Generate Figure 1: Confusion Matrices
    print("\n[Visualization] Generating Confusion Matrices Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Predicted FAKE", "Predicted REAL"],
            yticklabels=["Actual FAKE", "Actual REAL"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{name}", fontsize=13, pad=10, weight="bold")
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Classifier Output")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {CONFUSION_MATRIX_PNG}")

    # 6. Generate Figure 2: Model Comparison Bar Chart
    print("[Visualization] Generating Model Comparison Chart...")
    plot_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
    chart = sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette
    )
    plt.title("Empirical Performance Comparison Across NLP Classifiers", fontsize=14, weight="bold", pad=15)
    plt.ylim(0.70, 1.02)
    plt.ylabel("Score (0.00 - 1.00)")
    plt.xlabel("Classifier")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add numeric labels on bars
    for p in chart.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            chart.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {MODEL_COMPARISON_PNG}")

    # 7. Generate Figure 3: ROC Curves
    print("[Visualization] Generating ROC Curves Plot...")
    plt.figure(figsize=(9, 7))
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    for idx, (name, data) in enumerate(curves_data.items()):
        fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {data['roc_auc']:.4f})",
            color=colors[idx % len(colors)],
            linewidth=2.2
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", alpha=0.6)
    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, weight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {ROC_CURVE_PNG}")

    # 8. Generate Figure 4: Precision-Recall Curves
    print("[Visualization] Generating Precision-Recall Curves Plot...")
    plt.figure(figsize=(9, 7))

    for idx, (name, data) in enumerate(curves_data.items()):
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, data["y_proba"])
        plt.plot(
            rec_curve,
            prec_curve,
            label=f"{name}",
            color=colors[idx % len(colors)],
            linewidth=2.2
        )

    plt.xlim([0.0, 1.02])
    plt.ylim([0.70, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Across NLP Classifiers", fontsize=14, weight="bold", pad=15)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(PRECISION_RECALL_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {PRECISION_RECALL_PNG}")

    print("\n" + "=" * 75)
    print("       ALL EVALUATIONS AND ARTIFACTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return metrics_dict


if __name__ == "__main__":
    run_full_evaluation()
