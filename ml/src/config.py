from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

ISOT_DIR = DATASETS_DIR / "ISOT"
LIAR_DIR = DATASETS_DIR / "LIAR"
ISOT_FAKE_CSV = ISOT_DIR / "Fake.csv"
ISOT_TRUE_CSV = ISOT_DIR / "True.csv"
LIAR_TRAIN_TSV = LIAR_DIR / "train.tsv"
LIAR_VALID_TSV = LIAR_DIR / "valid.tsv"
LIAR_TEST_TSV = LIAR_DIR / "test.tsv"

MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_MULTINOMIAL_NB = "multinomial_nb"
MODEL_LINEAR_SVM = "linear_svm"
MODEL_DECISION_TREE = "decision_tree"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_NAMES = [
    MODEL_LOGISTIC_REGRESSION,
    MODEL_MULTINOMIAL_NB,
    MODEL_LINEAR_SVM,
    MODEL_DECISION_TREE,
    MODEL_RANDOM_FOREST,
]

MODEL_DISPLAY_NAMES = {
    MODEL_LOGISTIC_REGRESSION: "Logistic Regression",
    MODEL_MULTINOMIAL_NB: "Multinomial Naive Bayes",
    MODEL_LINEAR_SVM: "Linear Support Vector Machine",
    MODEL_DECISION_TREE: "Decision Tree",
    MODEL_RANDOM_FOREST: "Random Forest",
}


def get_model_dir(dataset: str, model_name: str) -> Path:
    return MODELS_DIR / dataset / model_name


def get_model_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "model.joblib"


def get_vectorizer_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "vectorizer.joblib"


def get_metadata_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "metadata.json"


def get_benchmark_json_path(dataset: str) -> Path:
    return RESULTS_DIR / dataset / "benchmark.json"


def get_benchmarks_csv_path(dataset: str) -> Path:
    return RESULTS_DIR / dataset / "benchmarks.csv"


def get_predictions_dir(dataset: str) -> Path:
    return RESULTS_DIR / dataset / "predictions"
