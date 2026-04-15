__version__ = "0.1.0"

import csv as _csv
from pathlib import Path as _Path

import numpy as _np

_BUNDLED_MODEL = _Path(__file__).resolve().parent / "model.pkl"
_model_cache = None

_FIELDS = [
    "filepath", "filename", "organism_prediction", "confidence",
    "confidence_warning", "nearest_centroid", "distance", "threshold",
    "within_distribution", "baclast_version",
]


def _load_model():
    global _model_cache
    if _model_cache is None:
        from baclast.model import load_model
        _model_cache = load_model(_BUNDLED_MODEL)
    return _model_cache


def predict(file: str) -> dict:
    """Classify a bacterial genome FASTA file.

    Args:
        file: Path to a FASTA file (.fasta, .fa, .fna).

    Returns:
        Dict with keys: filepath, filename, organism_prediction, confidence,
        confidence_warning, nearest_centroid, distance, threshold,
        within_distribution, baclast_version.
    """
    from baclast.features import genome_to_vector
    from baclast.model import novelty_score

    payload = _load_model()
    clf = payload["classifier"]
    label_names = payload["label_names"]
    k = payload["k"]
    kmer_vocab = payload["kmer_vocab"]
    centroids = payload.get("centroids")
    threshold = payload.get("distance_threshold")

    fpath = _Path(file)
    vec = genome_to_vector(fpath, k, kmer_vocab)
    X_q = _np.array([vec])
    pred = clf.predict(X_q)[0]
    proba = clf.predict_proba(X_q)[0]
    species = label_names[pred]
    confidence = round(float(proba[pred]) * 100, 2)

    result = {
        "filepath": str(fpath),
        "filename": fpath.name,
        "organism_prediction": species,
        "confidence": confidence,
        "confidence_warning": "LOW" if confidence < 70.0 else "",
        "baclast_version": __version__,
    }

    if centroids and threshold:
        nearest, dist = novelty_score(vec, centroids)
        result["nearest_centroid"] = nearest
        result["distance"] = round(float(dist), 6)
        result["threshold"] = round(float(threshold), 6)
        result["within_distribution"] = "Yes" if dist <= threshold else "No"
    else:
        result["nearest_centroid"] = ""
        result["distance"] = ""
        result["threshold"] = ""
        result["within_distribution"] = ""

    return result


def to_csv(result: dict, path: str) -> None:
    """Write a prediction result dict to a CSV file.

    Raises:
        FileExistsError: If the file already exists.
    """
    p = _Path(path)
    if p.exists():
        raise FileExistsError(f"Output file already exists: {p}")
    with open(p, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerow(result)
