"""K-mer feature extraction for bacterial genome classification."""

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from Bio import SeqIO

_DNA_BASES = set("ACGTNRYSWKMBDHV")
# Genome size bounds (bytes of sequence, not file size)
MIN_GENOME_BP = 500_000    # 500 Kb — smallest real bacterial genomes
MAX_GENOME_BP = 15_000_000  # 15 Mb — well above largest ESKAPE + 1 SD


def validate_fasta(fasta_path: Path, records: list) -> str:
    """Validate that parsed FASTA records look like a bacterial genome.

    Returns the combined uppercase sequence string.
    Raises ValueError with a descriptive message on failure.
    """
    if not records:
        raise ValueError(f"FASTA contains no sequences: {fasta_path}")

    combined = "N".join(str(r.seq) for r in records).upper()

    # Check it's DNA, not protein
    non_dna = set(combined) - _DNA_BASES
    if non_dna:
        raise ValueError(
            f"FASTA contains non-DNA characters ({', '.join(sorted(non_dna))}): "
            f"{fasta_path} — is this a protein FASTA?"
        )

    # Check genome size
    seq_len = len(combined.replace("N", ""))
    if seq_len < MIN_GENOME_BP:
        raise ValueError(
            f"Genome too small ({seq_len:,} bp, minimum {MIN_GENOME_BP:,} bp): "
            f"{fasta_path}"
        )
    if seq_len > MAX_GENOME_BP:
        raise ValueError(
            f"Genome too large ({seq_len:,} bp, maximum {MAX_GENOME_BP:,} bp): "
            f"{fasta_path}"
        )

    return combined


def all_kmers(k: int) -> list[str]:
    """Return all possible DNA k-mers of length k in lexicographic order."""
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def kmer_frequencies(sequence: str, k: int, kmer_vocab: list[str]) -> list[float]:
    """Compute normalised k-mer frequency vector.

    Args:
        sequence: Concatenated genome string (uppercase).
        k: K-mer length.
        kmer_vocab: Ordered vocabulary of all k-mers.

    Returns:
        List of floats, same length as kmer_vocab. All-zeros if no valid
        k-mers found.
    """
    counts = Counter()
    total = 0
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if all(c in "ACGT" for c in kmer):
            counts[kmer] += 1
            total += 1

    if total == 0:
        return [0.0] * len(kmer_vocab)

    return [counts[kmer] / total for kmer in kmer_vocab]


def genome_to_vector(
    fasta_path: str | Path, k: int, kmer_vocab: list[str], validate: bool = True
) -> list[float]:
    """Parse a FASTA file, concatenate contigs with 'N' separator, return freq vector.

    Args:
        fasta_path: Path to a FASTA file.
        k: K-mer length.
        kmer_vocab: Ordered vocabulary of all k-mers.
        validate: If True, check DNA content and genome size bounds.

    Raises:
        FileNotFoundError: If fasta_path does not exist.
        ValueError: If FASTA fails validation.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records = list(SeqIO.parse(str(fasta_path), "fasta"))

    if validate:
        combined = validate_fasta(fasta_path, records)
    else:
        if not records:
            raise ValueError(f"FASTA contains no sequences: {fasta_path}")
        combined = "N".join(str(r.seq) for r in records).upper()

    return kmer_frequencies(combined, k, kmer_vocab)


def load_dataset(
    data_dir: str | Path, k: int, kmer_vocab: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk data_dir; each sub-directory is a species label.

    Returns:
        (X, y, label_names) where X is the feature matrix, y is the label
        indices, and label_names maps indices to species names.

    Skips sub-directories with no FASTA files (warns, does not error).
    Accepted extensions: .fasta, .fa, .fna
    """
    import logging

    logger = logging.getLogger("baclast")
    data_dir = Path(data_dir)

    fasta_extensions = {".fasta", ".fa", ".fna"}
    label_names: list[str] = []
    X_rows: list[list[float]] = []
    y_labels: list[int] = []

    for species_dir in sorted(data_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        fasta_files = [
            f for f in species_dir.iterdir() if f.suffix in fasta_extensions
        ]
        if not fasta_files:
            logger.warning("No FASTA files in %s — skipping", species_dir.name)
            continue

        label_idx = len(label_names)
        label_names.append(species_dir.name)

        for fpath in sorted(fasta_files):
            print(f"  Loading {species_dir.name}/{fpath.name}")
            vec = genome_to_vector(fpath, k, kmer_vocab, validate=False)
            X_rows.append(vec)
            y_labels.append(label_idx)

    return np.array(X_rows), np.array(y_labels), label_names
