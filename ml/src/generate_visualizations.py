import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)

from src.config import (
    MODEL_NAMES,
    MODEL_DISPLAY_NAMES,
    get_benchmark_json_path,
    get_model_path,
    get_vectorizer_path,
)
from src.data_preprocessing import load_dataset

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
})

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ISOT_FIG_DIR = FIGURES_DIR / "isot"
LIAR_FIG_DIR = FIGURES_DIR / "liar"
COMP_FIG_DIR = FIGURES_DIR / "comparisons"

for d in [FIGURES_DIR, ISOT_FIG_DIR, LIAR_FIG_DIR, COMP_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_model_scores(model: Any, features) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        positive_index = list(model.classes_).index(1)
        return probabilities[:, positive_index]
    scores = np.asarray(model.decision_function(features), dtype=float)
    # Sigmoid normalization for threshold analysis
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))


def plot_single_confusion_matrix(cm: np.ndarray, model_name: str, dataset_name: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Calculate percentages
    total = np.sum(cm)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            annot_matrix[i, j] = f"{labels[i][j]}\n{count:,}\n({pct:.1f}%)"
            
    cmap = "Blues" if dataset_name.lower() == "isot" else "Purples"
    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        cbar=False,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Predicted FAKE", "Predicted REAL"],
        yticklabels=["Actual FAKE", "Actual REAL"],
        linewidths=1.5,
        linecolor="white",
    )
    
    ax.set_title(f"Confusion Matrix: {model_name}\n({dataset_name.upper()} Test Set)", pad=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_all_confusion_matrices_grid(cms: Dict[str, np.ndarray], dataset_name: str, save_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    cmap = "Blues" if dataset_name.lower() == "isot" else "Purples"
    labels_tag = [["TN", "FP"], ["FN", "TP"]]
    
    for idx, (m_id, m_name) in enumerate(MODEL_DISPLAY_NAMES.items()):
        ax = axes[idx]
        cm = cms[m_id]
        total = np.sum(cm)
        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = (count / total) * 100
                annot_matrix[i, j] = f"{labels_tag[i][j]}\n{count:,}\n({pct:.1f}%)"
                
        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
            xticklabels=["Pred FAKE", "Pred REAL"],
            yticklabels=["Act FAKE", "Act REAL"],
            linewidths=1.2,
            linecolor="white",
        )
        ax.set_title(m_name, weight="bold", fontsize=13, pad=10)
        
    # Hide the 6th unused subplot
    axes[5].axis("off")
    
    fig.suptitle(f"Confusion Matrices for All 5 Models ({dataset_name.upper()} 20% Test Partition)", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {save_path}")


def plot_accuracy_comparison(benchmark_data: Dict[str, Any], dataset_name: str, save_path: Path):
    models = benchmark_data.get("models", [])
    model_names = [m["name"] for m in models]
    test_accs = [m["metrics"]["accuracy"] * 100 for m in models]
    paper_accs = [m.get("paperReference", {}).get("accuracy", 0) * 100 for m in models]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    primary_color = "#2b5c8f" if dataset_name.lower() == "isot" else "#6a4c93"
    paper_color = "#999999"
    
    rects1 = ax.bar(x - width/2, paper_accs, width, label="Paper Baseline", color=paper_color, edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, test_accs, width, label="Enhanced Model (Ours)", color=primary_color, edgecolor="black")
    
    ax.set_ylabel("Accuracy (%)", fontsize=13, weight="bold")
    ax.set_title(f"Model Accuracy Comparison: Enhanced vs. Paper Baseline ({dataset_name.upper()} Dataset)", fontsize=14, weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", weight="bold")
    ax.legend(frameon=True, loc="lower right" if dataset_name.lower() == "liar" else "lower left")
    
    # Set y limits nicely
    min_val = min(min(test_accs), min(paper_accs))
    max_val = max(max(test_accs), max(paper_accs))
    y_min = max(0, min_val - (5 if dataset_name.lower() == "liar" else 2))
    y_max = min(100, max_val + (8 if dataset_name.lower() == "liar" else 2))
    ax.set_ylim(y_min, y_max)
    
    # Annotate bars
    def autolabel(rects, is_enhanced=False):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            delta = test_accs[idx] - paper_accs[idx]
            if is_enhanced:
                delta_str = f" ({delta:+.2f}%)" if abs(delta) > 0.001 else ""
                ax.annotate(
                    f"{height:.2f}%{delta_str}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9.5, weight="bold",
                    color=primary_color
                )
            else:
                ax.annotate(
                    f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9.5, color="#555555"
                )
                
    autolabel(rects1)
    autolabel(rects2, is_enhanced=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_side_by_side_accuracy(isot_bm: Dict[str, Any], liar_bm: Dict[str, Any], save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    
    for ax, bm, d_name, col in [(ax1, isot_bm, "ISOT", "#2b5c8f"), (ax2, liar_bm, "LIAR", "#6a4c93")]:
        models = bm.get("models", [])
        m_names = [m["name"] for m in models]
        t_accs = [m["metrics"]["accuracy"] * 100 for m in models]
        p_accs = [m.get("paperReference", {}).get("accuracy", 0) * 100 for m in models]
        
        x = np.arange(len(m_names))
        width = 0.35
        
        r1 = ax.bar(x - width/2, p_accs, width, label="Paper Baseline", color="#999999", edgecolor="black", alpha=0.85)
        r2 = ax.bar(x + width/2, t_accs, width, label="Enhanced Model", color=col, edgecolor="black")
        
        ax.set_ylabel("Accuracy (%)", fontsize=12, weight="bold")
        ax.set_title(f"{d_name} Dataset (5 Classifiers)", fontsize=14, weight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(m_names, rotation=20, ha="right", fontsize=10, weight="bold")
        ax.legend(loc="lower left" if d_name == "ISOT" else "lower right")
        
        y_min = 90 if d_name == "ISOT" else 50
        y_max = 101 if d_name == "ISOT" else 67
        ax.set_ylim(y_min, y_max)
        
        for idx, rect in enumerate(r2):
            h = rect.get_height()
            delta = t_accs[idx] - p_accs[idx]
            delta_str = f" ({delta:+.2f}%)" if abs(delta) > 0.001 else ""
            ax.annotate(
                f"{h:.2f}%\n{delta_str}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, weight="bold",
                color=col
            )
            
    fig.suptitle("Overall Accuracy Comparison across ISOT and LIAR Datasets (80/20 Test Split)", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side: {save_path}")


def plot_precision_recall_curves(
    y_test: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    dataset_name: str,
    save_path: Path
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    
    colors = {
        "logistic_regression": "#1f77b4",
        "multinomial_nb": "#ff7f0e",
        "linear_svm": "#2ca02c",
        "decision_tree": "#d62728",
        "random_forest": "#9467bd",
    }
    
    # Calculate baseline (prevalence of positive class)
    baseline = np.sum(y_test == 1) / len(y_test)
    ax.plot([0, 1], [baseline, baseline], "k--", alpha=0.6, label=f"No Skill Baseline ({baseline:.2f})")
    
    for m_id, m_name in MODEL_DISPLAY_NAMES.items():
        scores = model_scores[m_id]
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        
        ax.plot(
            rec,
            prec,
            lw=2.2,
            color=colors.get(m_id, "blue"),
            label=f"{m_name} (AP = {ap:.4f})"
        )
        
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=13, weight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=13, weight="bold")
    ax.set_title(f"Precision-Recall Curves: {dataset_name.upper()} Dataset (20% Test Set)", fontsize=14, weight="bold", pad=14)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0 if dataset_name.lower() == "liar" else 0.8, 1.03])
    ax.legend(loc="lower left" if dataset_name.lower() == "isot" else "upper right", frameon=True, fontsize=10.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_recall_vs_threshold_curves(
    y_test: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    dataset_name: str,
    save_path: Path
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    
    colors = {
        "logistic_regression": "#1f77b4",
        "multinomial_nb": "#ff7f0e",
        "linear_svm": "#2ca02c",
        "decision_tree": "#d62728",
        "random_forest": "#9467bd",
    }
    
    thresholds_range = np.linspace(0.01, 0.99, 100)
    
    for m_id, m_name in MODEL_DISPLAY_NAMES.items():
        scores = model_scores[m_id]
        recalls = []
        for th in thresholds_range:
            preds_at_th = (scores >= th).astype(int)
            tp = np.sum((y_test == 1) & (preds_at_th == 1))
            actual_pos = np.sum(y_test == 1)
            recalls.append(tp / actual_pos if actual_pos > 0 else 0)
            
        ax.plot(thresholds_range, recalls, lw=2.2, color=colors.get(m_id, "blue"), label=m_name)
        
    ax.axvline(0.5, color="red", linestyle=":", alpha=0.7, label="Default Threshold (0.50)")
    ax.set_xlabel("Classification Decision Threshold", fontsize=13, weight="bold")
    ax.set_ylabel("Recall (True Positive Rate)", fontsize=13, weight="bold")
    ax.set_title(f"Recall vs. Decision Threshold: {dataset_name.upper()} Dataset", fontsize=14, weight="bold", pad=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower left", frameon=True, fontsize=10.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Recall vs Threshold: {save_path}")


def plot_side_by_side_recall_threshold(
    isot_y_test: np.ndarray,
    isot_scores: Dict[str, np.ndarray],
    liar_y_test: np.ndarray,
    liar_scores: Dict[str, np.ndarray],
    save_path: Path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))
    
    colors = {
        "logistic_regression": "#1f77b4",
        "multinomial_nb": "#ff7f0e",
        "linear_svm": "#2ca02c",
        "decision_tree": "#d62728",
        "random_forest": "#9467bd",
    }
    
    thresholds_range = np.linspace(0.01, 0.99, 100)
    
    for ax, y_t, s_dict, d_name in [(ax1, isot_y_test, isot_scores, "ISOT"), (ax2, liar_y_test, liar_scores, "LIAR")]:
        actual_pos = np.sum(y_t == 1)
        for m_id, m_name in MODEL_DISPLAY_NAMES.items():
            scores = s_dict[m_id]
            recalls = [np.sum((y_t == 1) & (scores >= th)) / actual_pos for th in thresholds_range]
            ax.plot(thresholds_range, recalls, lw=2.2, color=colors.get(m_id, "blue"), label=m_name)
            
        ax.axvline(0.5, color="red", linestyle=":", alpha=0.7, label="Default Threshold (0.50)")
        ax.set_xlabel("Decision Threshold", fontsize=12, weight="bold")
        ax.set_ylabel("Recall (Sensitivity)", fontsize=12, weight="bold")
        ax.set_title(f"{d_name} Dataset Recall vs Threshold", fontsize=14, weight="bold", pad=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.legend(loc="lower left", fontsize=10)
        
    fig.suptitle("Recall Sensitivity Curves Across Classification Thresholds (20% Test Sets)", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side recall threshold: {save_path}")


def plot_side_by_side_pr_curves(
    isot_y_test: np.ndarray,
    isot_scores: Dict[str, np.ndarray],
    liar_y_test: np.ndarray,
    liar_scores: Dict[str, np.ndarray],
    save_path: Path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))
    
    colors = {
        "logistic_regression": "#1f77b4",
        "multinomial_nb": "#ff7f0e",
        "linear_svm": "#2ca02c",
        "decision_tree": "#d62728",
        "random_forest": "#9467bd",
    }
    
    # 1. ISOT PR curve
    base_isot = np.sum(isot_y_test == 1) / len(isot_y_test)
    ax1.plot([0, 1], [base_isot, base_isot], "k--", alpha=0.6, label=f"No Skill ({base_isot:.2f})")
    for m_id, m_name in MODEL_DISPLAY_NAMES.items():
        scores = isot_scores[m_id]
        prec, rec, _ = precision_recall_curve(isot_y_test, scores)
        ap = average_precision_score(isot_y_test, scores)
        ax1.plot(rec, prec, lw=2.2, color=colors.get(m_id, "blue"), label=f"{m_name} (AP={ap:.4f})")
    ax1.set_xlabel("Recall", fontsize=12, weight="bold")
    ax1.set_ylabel("Precision", fontsize=12, weight="bold")
    ax1.set_title("ISOT Dataset Precision-Recall Curves", fontsize=14, weight="bold", pad=12)
    ax1.set_xlim([0.0, 1.02])
    ax1.set_ylim([0.85, 1.02])
    ax1.legend(loc="lower left", fontsize=10)
    
    # 2. LIAR PR curve
    base_liar = np.sum(liar_y_test == 1) / len(liar_y_test)
    ax2.plot([0, 1], [base_liar, base_liar], "k--", alpha=0.6, label=f"No Skill ({base_liar:.2f})")
    for m_id, m_name in MODEL_DISPLAY_NAMES.items():
        scores = liar_scores[m_id]
        prec, rec, _ = precision_recall_curve(liar_y_test, scores)
        ap = average_precision_score(liar_y_test, scores)
        ax2.plot(rec, prec, lw=2.2, color=colors.get(m_id, "blue"), label=f"{m_name} (AP={ap:.4f})")
    ax2.set_xlabel("Recall", fontsize=12, weight="bold")
    ax2.set_ylabel("Precision", fontsize=12, weight="bold")
    ax2.set_title("LIAR Dataset Precision-Recall Curves", fontsize=14, weight="bold", pad=12)
    ax2.set_xlim([0.0, 1.02])
    ax2.set_ylim([0.45, 0.95])
    ax2.legend(loc="upper right", fontsize=10)
    
    fig.suptitle("Comparative Precision-Recall & Recall Trajectories on 20% Test Sets", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side PR curves: {save_path}")


def main():
    print("Generating comprehensive evaluation figures and curves...")
    
    # Load benchmarks
    with get_benchmark_json_path("isot").open("r", encoding="utf-8") as f:
        isot_bm = json.load(f)
    with get_benchmark_json_path("liar").open("r", encoding="utf-8") as f:
        liar_bm = json.load(f)
        
    # 1. Accuracy Comparisons
    plot_accuracy_comparison(isot_bm, "isot", ISOT_FIG_DIR / "isot_accuracy_comparison.png")
    plot_accuracy_comparison(liar_bm, "liar", LIAR_FIG_DIR / "liar_accuracy_comparison.png")
    plot_side_by_side_accuracy(isot_bm, liar_bm, COMP_FIG_DIR / "both_datasets_accuracy_comparison.png")
    
    # 2. Confusion Matrices & Scores Evaluation
    datasets_data = {}
    for d_name, d_dir, bm_data in [("isot", ISOT_FIG_DIR, isot_bm), ("liar", LIAR_FIG_DIR, liar_bm)]:
        models_available = all(get_model_path(d_name, m_id).exists() for m_id in MODEL_DISPLAY_NAMES)
        cms = {}
        scores_dict = {}

        if models_available:
            x_train, x_test, y_train, y_test = load_dataset(d_name, test_size=0.2)
            for m_id, m_name in MODEL_DISPLAY_NAMES.items():
                model = joblib.load(get_model_path(d_name, m_id))
                vec = joblib.load(get_vectorizer_path(d_name, m_id))

                x_test_vec = vec.transform(x_test)
                preds = model.predict(x_test_vec)
                scores = get_model_scores(model, x_test_vec)

                cm = confusion_matrix(y_test, preds, labels=[0, 1])
                cms[m_id] = cm
                scores_dict[m_id] = scores

                # Individual CM plot
                plot_single_confusion_matrix(cm, m_name, d_name, d_dir / f"confusion_matrix_{m_id}.png")

            # Grid CM plot for this dataset
            plot_all_confusion_matrices_grid(cms, d_name, d_dir / f"{d_name}_all_confusion_matrices.png")

            # PR Curve for this dataset
            plot_precision_recall_curves(y_test, scores_dict, d_name, d_dir / f"{d_name}_precision_recall_curves.png")

            # Recall vs Threshold Curve for this dataset
            plot_recall_vs_threshold_curves(y_test, scores_dict, d_name, d_dir / f"{d_name}_recall_vs_threshold.png")

            datasets_data[d_name] = {"y_test": y_test, "scores": scores_dict}
        else:
            # Reconstruct confusion matrices from benchmark json
            bm_models = {m["id"]: m for m in bm_data.get("models", [])}
            for m_id, m_name in MODEL_DISPLAY_NAMES.items():
                if m_id in bm_models and "confusion_matrix" in bm_models[m_id].get("metrics", {}):
                    cm = np.array(bm_models[m_id]["metrics"]["confusion_matrix"])
                    cms[m_id] = cm
                    plot_single_confusion_matrix(cm, m_name, d_name, d_dir / f"confusion_matrix_{m_id}.png")
            if cms:
                plot_all_confusion_matrices_grid(cms, d_name, d_dir / f"{d_name}_all_confusion_matrices.png")

    if "isot" in datasets_data and "liar" in datasets_data:
        # Side-by-side PR curves
        plot_side_by_side_pr_curves(
            datasets_data["isot"]["y_test"],
            datasets_data["isot"]["scores"],
            datasets_data["liar"]["y_test"],
            datasets_data["liar"]["scores"],
            COMP_FIG_DIR / "both_datasets_recall_pr_curves.png"
        )

        # Side-by-side Recall vs Threshold curves
        plot_side_by_side_recall_threshold(
            datasets_data["isot"]["y_test"],
            datasets_data["isot"]["scores"],
            datasets_data["liar"]["y_test"],
            datasets_data["liar"]["scores"],
            COMP_FIG_DIR / "both_datasets_recall_vs_threshold.png"
        )
    
    print("\nAll visualization images generated successfully!")


if __name__ == "__main__":
    main()
