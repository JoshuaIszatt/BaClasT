"""Model training, evaluation, and persistence for BaClasT."""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from baclast import __version__
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_REQUIRED_KEYS = {
    "classifier",
    "label_names",
    "k",
    "kmer_vocab",
    "baclasp_version",
    "trained_at",
    "n_genomes",
}


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest with class_weight='balanced'.

    Uses all available CPU cores (n_jobs=-1).
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def evaluate(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict:
    """Evaluate a fitted classifier and print a formatted summary.

    Returns:
        Dict with keys: accuracy, report (str), confusion_matrix (np.ndarray).
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"\n{report}")
    print(f"Confusion matrix:\n{cm}")

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Centroid-based novelty detection
# ---------------------------------------------------------------------------

def compute_centroids(
    X: np.ndarray, y: np.ndarray, label_names: list[str]
) -> dict[str, np.ndarray]:
    """Compute the mean k-mer vector (centroid) for each class.

    Returns:
        Dict mapping species name to its centroid vector.
    """
    centroids = {}
    for i, name in enumerate(label_names):
        mask = y == i
        centroids[name] = X[mask].mean(axis=0)
    return centroids


def compute_distance_threshold(
    X: np.ndarray,
    y: np.ndarray,
    centroids: dict[str, np.ndarray],
    label_names: list[str],
    percentile: float = 99.0,
) -> float:
    """Calibrate an out-of-distribution distance threshold from training data.

    Computes the Euclidean distance from every training genome to its own
    class centroid, then returns the given percentile as the threshold.
    Genomes further than this are flagged as out-of-distribution.
    """
    distances = []
    for i, name in enumerate(label_names):
        mask = y == i
        centroid = centroids[name]
        for vec in X[mask]:
            distances.append(np.linalg.norm(vec - centroid))
    return float(np.percentile(distances, percentile))


def novelty_score(
    query_vector: np.ndarray | list[float],
    centroids: dict[str, np.ndarray],
) -> tuple[str, float]:
    """Compute distance from a query to all centroids.

    Returns:
        (nearest_species, distance) — the closest centroid and its distance.
    """
    query = np.asarray(query_vector)
    nearest = None
    min_dist = float("inf")
    for name, centroid in centroids.items():
        dist = float(np.linalg.norm(query - centroid))
        if dist < min_dist:
            min_dist = dist
            nearest = name
    return nearest, min_dist


def centroid_distances(centroids: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Compute pairwise Euclidean distances between all centroids.

    Returns:
        Dict mapping (species_a, species_b) to distance.
    """
    names = sorted(centroids.keys())
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            dist = float(np.linalg.norm(centroids[a] - centroids[b]))
            result[(a, b)] = dist
    return result


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    clf: RandomForestClassifier,
    label_names: list[str],
    k: int,
    kmer_vocab: list[str],
    path: str | Path,
    centroids: dict[str, np.ndarray] | None = None,
    distance_threshold: float | None = None,
) -> None:
    """Save all model state needed for prediction.

    Payload: classifier, label_names, k, kmer_vocab, baclasp_version,
    trained_at, n_genomes, and optionally centroids + distance_threshold.
    """
    payload = {
        "classifier": clf,
        "label_names": label_names,
        "k": k,
        "kmer_vocab": kmer_vocab,
        "baclasp_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_genomes": clf.n_features_in_ if hasattr(clf, "n_samples_seen_") else 0,
    }
    if centroids is not None:
        payload["centroids"] = centroids
    if distance_threshold is not None:
        payload["distance_threshold"] = distance_threshold
    # n_genomes will be set properly by the caller via the payload;
    # here we store a placeholder that the CLI will override.
    joblib.dump(payload, path)


def load_model(path: str | Path) -> dict:
    """Load model payload. Raises FileNotFoundError if path missing.

    Validates that payload has all required keys before returning.
    Centroids and distance_threshold are optional (older models may lack them).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            "Model file appears corrupted. Retrain with 'train' command."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Model file appears corrupted. Retrain with 'train' command."
        )

    missing = _REQUIRED_KEYS - set(payload.keys())
    if missing:
        raise ValueError(
            f"Model payload is missing required keys: {', '.join(sorted(missing))}"
        )

    return payload
