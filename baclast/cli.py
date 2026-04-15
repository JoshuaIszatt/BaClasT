"""BaClasT CLI — bacterial genome classification tool."""

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

from baclast import __version__
from baclast.features import genome_to_vector
from baclast.model import load_model, novelty_score

_BUNDLED_MODEL = Path(__file__).resolve().parent / "model.pkl"
_FASTA_EXTENSIONS = {".fasta", ".fa", ".fna"}
_MIN_CONFIDENCE = 70.0  # below this, flag as low confidence
_CSV_FIELDS = [
    "filepath",
    "filename",
    "organism_prediction",
    "confidence",
    "confidence_warning",
    "nearest_centroid",
    "distance",
    "threshold",
    "within_distribution",
    "baclast_version",
]


def _find_model(user_path: str | None) -> Path:
    """Resolve model path: user-provided, or bundled default."""
    if user_path:
        p = Path(user_path)
        if not p.exists():
            sys.exit(f"Error: Model file not found: {p}")
        return p
    if _BUNDLED_MODEL.exists():
        return _BUNDLED_MODEL
    sys.exit(f"Error: No model found. Provide --model or install a model to {_BUNDLED_MODEL}")


def _collect_fastas(target: Path) -> list[Path]:
    """Return a list of FASTA files from a file path or directory."""
    if target.is_file():
        if target.suffix not in _FASTA_EXTENSIONS:
            sys.exit(f"Error: {target} does not look like a FASTA file. "
                     f"Expected extensions: {', '.join(sorted(_FASTA_EXTENSIONS))}")
        return [target]
    if target.is_dir():
        fastas = sorted(f for f in target.iterdir() if f.suffix in _FASTA_EXTENSIONS)
        if not fastas:
            sys.exit(f"Error: No FASTA files found in {target}")
        return fastas
    sys.exit(f"Error: Path not found: {target}")


def _classify_one(fpath: Path, clf, label_names, k, kmer_vocab, centroids, threshold) -> dict:
    """Classify a single FASTA and return a result row."""
    vec = genome_to_vector(fpath, k, kmer_vocab)
    X_q = np.array([vec])
    pred = clf.predict(X_q)[0]
    proba = clf.predict_proba(X_q)[0]
    species = label_names[pred]
    confidence = round(proba[pred] * 100, 2)

    if confidence < _MIN_CONFIDENCE:
        warning = "LOW"
    else:
        warning = ""

    row = {
        "filepath": str(fpath),
        "filename": fpath.name,
        "organism_prediction": species,
        "confidence": confidence,
        "confidence_warning": warning,
        "baclast_version": __version__,
    }

    if centroids and threshold:
        nearest, dist = novelty_score(vec, centroids)
        row["nearest_centroid"] = nearest
        row["distance"] = round(dist, 6)
        row["threshold"] = round(threshold, 6)
        row["within_distribution"] = "Yes" if dist <= threshold else "No"
    else:
        row["nearest_centroid"] = ""
        row["distance"] = ""
        row["threshold"] = ""
        row["within_distribution"] = ""

    return row


def main():
    parser = argparse.ArgumentParser(
        prog="baclast",
        description="BaClasT — fast bacterial genome classification using k-mer profiles",
    )
    parser.add_argument(
        "--predict", required=True, metavar="PATH",
        help="Path to a FASTA file or directory of FASTAs",
    )
    parser.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="Write results to a CSV file instead of stdout",
    )
    parser.add_argument(
        "--model", default=None, metavar="FILE",
        help="Path to model .pkl (uses bundled model if omitted)",
    )

    args = parser.parse_args()

    # Check output file doesn't already exist
    if args.output:
        out_path = Path(args.output)
        if out_path.exists():
            sys.exit(f"Error: Output file already exists: {out_path}")

    # Load model
    payload = load_model(_find_model(args.model))
    clf = payload["classifier"]
    label_names = payload["label_names"]
    k = payload["k"]
    kmer_vocab = payload["kmer_vocab"]
    centroids = payload.get("centroids")
    threshold = payload.get("distance_threshold")

    # Collect input FASTAs
    target = Path(args.predict)
    fastas = _collect_fastas(target)

    # Classify
    rows = []
    for i, fpath in enumerate(fastas, 1):
        if len(fastas) > 1:
            print(f"  [{i}/{len(fastas)}] {fpath.name} ... ", end="", flush=True, file=sys.stderr)
        try:
            row = _classify_one(fpath, clf, label_names, k, kmer_vocab, centroids, threshold)
            rows.append(row)
            status = f"{row['organism_prediction']} ({row['confidence']}%)"
            if row["confidence_warning"]:
                status += f" [{row['confidence_warning']} CONFIDENCE]"
            if len(fastas) > 1:
                print(status, file=sys.stderr)
        except (ValueError, Exception) as exc:
            rows.append({
                "filepath": str(fpath),
                "filename": fpath.name,
                "organism_prediction": "SKIPPED",
                "confidence": "",
                "confidence_warning": str(exc),
                "nearest_centroid": "",
                "distance": "",
                "threshold": "",
                "within_distribution": "",
                "baclast_version": __version__,
            })
            if len(fastas) > 1:
                print(f"SKIPPED: {exc}", file=sys.stderr)
            else:
                sys.exit(f"Error: {exc}")

    # Output
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to {out_path} ({len(rows)} genomes)", file=sys.stderr)
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        print(buf.getvalue(), end="")


if __name__ == "__main__":
    main()
