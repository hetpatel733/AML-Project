import os
import urllib.request
import zipfile
from pathlib import Path

# Base ML Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Dataset Files
ISOT_DIR = DATASETS_DIR / "isot"
LIAR_DIR = DATASETS_DIR / "liar"

ISOT_FAKE_CSV = ISOT_DIR / "Fake.csv"
ISOT_TRUE_CSV = ISOT_DIR / "True.csv"

LIAR_TRAIN_TSV = LIAR_DIR / "train.tsv"
LIAR_VALID_TSV = LIAR_DIR / "valid.tsv"
LIAR_TEST_TSV = LIAR_DIR / "test.tsv"
GLOVE_PATH = Path(os.getenv("GLOVE_PATH", str(DATASETS_DIR / "glove.6B.100d.txt")))
GLOVE_ARCHIVE_PATH = DATASETS_DIR / "glove.6B.zip"


def ensure_glove_embeddings() -> Path:
    """Download and validate the standard GloVe 100d embeddings before training."""
    if GLOVE_PATH.exists() and GLOVE_PATH.stat().st_size > 1024:
        try:
            with open(GLOVE_PATH, "r", encoding="utf-8") as glove_file:
                first_line = glove_file.readline()
            if first_line.strip():
                return GLOVE_PATH
        except Exception:
            pass

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if GLOVE_PATH.exists():
        GLOVE_PATH.unlink()
    if GLOVE_ARCHIVE_PATH.exists():
        GLOVE_ARCHIVE_PATH.unlink()

    archive_urls = [
        "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
        "https://nlp.stanford.edu/data/glove.6B.zip",
    ]

    archive_file = None
    for archive_url in archive_urls:
        print(f"Downloading GloVe archive from {archive_url}...")
        temp_archive = DATASETS_DIR / f"glove.6B.{abs(hash(archive_url))}.zip"
        try:
            urllib.request.urlretrieve(archive_url, str(temp_archive))
            if zipfile.is_zipfile(temp_archive):
                archive_file = temp_archive
                break
        except Exception as exc:
            print(f"Download attempt failed for {archive_url}: {exc}")
        finally:
            if archive_file is None and temp_archive.exists():
                temp_archive.unlink(missing_ok=True)

    if archive_file is None:
        raise RuntimeError("Could not download a valid GloVe archive from the official sources.")

    with zipfile.ZipFile(archive_file) as archive:
        members = [name for name in archive.namelist() if name.endswith("glove.6B.100d.txt")]
        if not members:
            raise FileNotFoundError("The downloaded GloVe archive does not contain glove.6B.100d.txt.")
        archive.extract(members[0], DATASETS_DIR)

    extracted_path = DATASETS_DIR / Path(members[0]).name
    if not extracted_path.exists():
        raise FileNotFoundError(f"GloVe embeddings were not created at {extracted_path}.")

    if extracted_path != GLOVE_PATH:
        extracted_path.replace(GLOVE_PATH)

    archive_file.unlink(missing_ok=True)
    return GLOVE_PATH


# Model Names
MODEL_TFIDF_PAC = "tfidf_pac"
MODEL_TFIDF_RF = "tfidf_random_forest"
MODEL_TFIDF_LR = "tfidf_logistic_regression"
MODEL_GLOVE_CNN_BILSTM = "glove_cnn_bilstm"

# Saved Model & Artifact Paths
def get_model_dir(dataset: str, model_name: str) -> Path:
    return MODELS_DIR / dataset / model_name

def get_model_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "model.joblib"

def get_keras_model_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "model.keras"

def get_vectorizer_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "vectorizer.joblib"

def get_tokenizer_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "tokenizer.joblib"

def get_metadata_path(dataset: str, model_name: str) -> Path:
    return get_model_dir(dataset, model_name) / "metadata.json"

# Evaluation Result Artifacts
def get_benchmark_json_path(dataset: str) -> Path:
    return RESULTS_DIR / dataset / "benchmark.json"


def get_benchmarks_json_path(dataset: str) -> Path:
    """Backward-compatible alias for the authoritative benchmark path."""
    return get_benchmark_json_path(dataset)

def get_benchmarks_csv_path(dataset: str) -> Path:
    return RESULTS_DIR / dataset / "benchmarks.csv"

def get_predictions_dir(dataset: str) -> Path:
    return RESULTS_DIR / dataset / "predictions"

# Keras Configuration Defaults
KERAS_CONFIG = {
    "glove_version": "6B",
    "embedding_dim": 100,
    "vocab_size": 20000,
    "sequence_length": 500,
    "cnn_filters": 128,
    "kernel_size": 5,
    "bilstm_units": 64,
    "dropout": 0.5,
    "dense_units": 32,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}

