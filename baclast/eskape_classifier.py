"""BaClasT — main CLI entry point for training and prediction."""

import argparse
import sys

from baclast.features import all_kmers, genome_to_vector, load_dataset
from baclast.model import evaluate, load_model, save_model, train_classifier
from baclast.utils import print_banner, setup_logging


def cmd_train(args):
    """Execute the 'train' sub-command."""
    logger = setup_logging(args.verbose)
    print_banner()

    k = args.k
    kmer_vocab = all_kmers(k)

    print(f"Loading genomes from {args.data_dir} (k={k})...")
    try:
        X, y, label_names = load_dataset(args.data_dir, k, kmer_vocab)
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    if len(label_names) < 2:
        sys.exit("Error: Need at least 2 species to train a classifier.")

    n_genomes = len(y)
    print(f"Loaded {n_genomes} genomes across {len(label_names)} species.")

    # 80/20 stratified train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training Random Forest ({args.n_estimators} trees)...")
    clf = train_classifier(X_train, y_train, n_estimators=args.n_estimators)

    print("Evaluating on held-out test set:")
    evaluate(clf, X_test, y_test, label_names)

    # Optional cross-validation
    if args.cv is not None:
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        print(f"\nRunning {args.cv}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Save model
    save_model(clf, label_names, k, kmer_vocab, args.output)

    # Patch in n_genomes (save_model doesn't know the total count)
    import joblib

    payload = joblib.load(args.output)
    payload["n_genomes"] = n_genomes
    joblib.dump(payload, args.output)

    print(f"\nModel saved to {args.output}")


def cmd_predict(args):
    """Execute the 'predict' sub-command."""
    logger = setup_logging(args.verbose)
    print_banner()

    # Load model
    try:
        payload = load_model(args.model)
    except FileNotFoundError:
        sys.exit(f"Error: Model file not found: {args.model}")
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    clf = payload["classifier"]
    label_names = payload["label_names"]
    k = payload["k"]
    kmer_vocab = payload["kmer_vocab"]

    # Extract features from input FASTA
    try:
        vec = genome_to_vector(args.fasta, k, kmer_vocab)
    except FileNotFoundError:
        sys.exit(f"Error: FASTA file not found: {args.fasta}")
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    import numpy as np

    X_query = np.array([vec])
    pred = clf.predict(X_query)[0]
    proba = clf.predict_proba(X_query)[0]

    species = label_names[pred]
    confidence = proba[pred] * 100

    print(f"Predicted species: {species}")
    print(f"Confidence: {confidence:.1f}%")

    if args.verbose:
        print("\nAll species probabilities:")
        # Sort by probability descending
        ranked = sorted(
            zip(label_names, proba), key=lambda x: x[1], reverse=True
        )
        max_name_len = max(len(name) for name in label_names)
        for name, prob in ranked:
            bar_len = int(prob * 40)
            bar = "#" * bar_len
            print(f"  {name:<{max_name_len}}  {prob * 100:5.1f}%  |{bar}")


def main():
    """Main entry point — parse arguments and dispatch to sub-commands."""
    parser = argparse.ArgumentParser(
        prog="baclasp",
        description="BaClasT — Bacterial Classification Tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train sub-command
    train_parser = subparsers.add_parser("train", help="Train a classifier")
    train_parser.add_argument(
        "--data_dir", required=True, help="Directory with species sub-folders"
    )
    train_parser.add_argument(
        "--output", default="model.pkl", help="Output model path (default: model.pkl)"
    )
    train_parser.add_argument(
        "--k", type=int, default=4, help="K-mer length (default: 4)"
    )
    train_parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of trees (default: 200)",
    )
    train_parser.add_argument(
        "--cv", type=int, default=None, help="Number of CV folds (optional)"
    )
    train_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    # predict sub-command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict species for a FASTA file"
    )
    predict_parser.add_argument(
        "--model", required=True, help="Path to trained model .pkl"
    )
    predict_parser.add_argument(
        "--fasta", required=True, help="Path to input FASTA file"
    )
    predict_parser.add_argument(
        "--verbose", action="store_true", help="Show all species probabilities"
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
