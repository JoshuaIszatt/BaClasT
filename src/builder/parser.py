"""Parse NCBI datasets genome summary JSONL into a pandas DataFrame."""

import json
from pathlib import Path

import pandas as pd

# Fields extracted from NCBI datasets JSONL
_NUMERIC_COLS = [
    "genome_size",
    "n_contigs",
    "contig_n50",
    "scaffold_n50",
    "gc_percent",
    "genes_total",
    "genes_protein",
]


def parse_genome_jsonl(json_path: str | Path) -> pd.DataFrame:
    """Parse NCBI datasets genome summary JSONL into a DataFrame.

    Each line of the input file is a JSON object produced by:
        datasets summary genome taxon "<name>" --as-json-lines
    """
    records = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append({
                # Identity
                "accession":       d.get("accession"),
                "organism_name":   d.get("organism", {}).get("organism_name"),
                "strain":          d.get("organism", {}).get("infraspecific_names", {}).get("strain"),
                # Assembly info
                "assembly_level":  d.get("assembly_info", {}).get("assembly_level"),
                "assembly_name":   d.get("assembly_info", {}).get("assembly_name"),
                "refseq_category": d.get("assembly_info", {}).get("refseq_category"),
                "release_date":    d.get("assembly_info", {}).get("release_date"),
                "assembly_status": d.get("assembly_info", {}).get("assembly_status"),
                "submitter":       d.get("assembly_info", {}).get("submitter"),
                # Assembly stats
                "genome_size":     d.get("assembly_stats", {}).get("total_sequence_length"),
                "n_contigs":       d.get("assembly_stats", {}).get("number_of_contigs"),
                "contig_n50":      d.get("assembly_stats", {}).get("contig_n50"),
                "scaffold_n50":    d.get("assembly_stats", {}).get("scaffold_n50"),
                "gc_percent":      d.get("assembly_stats", {}).get("gc_percent"),
                # Annotation
                "genes_total":     d.get("annotation_info", {}).get("stats", {}).get("gene_counts", {}).get("total"),
                "genes_protein":   d.get("annotation_info", {}).get("stats", {}).get("gene_counts", {}).get("protein_coding"),
            })

    df = pd.DataFrame(records)

    if not df.empty:
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print summary statistics of a genome dataset DataFrame to stdout."""
    if df.empty:
        print("No genomes found.")
        return
    print(f"Shape: {df.shape}")
    print(f"\nAssembly levels:\n{df['assembly_level'].value_counts()}")
    print(f"\nRefSeq categories:\n{df['refseq_category'].value_counts()}")
    print(f"\nNumeric summary:")
    numeric = [c for c in ["genome_size", "n_contigs", "contig_n50", "gc_percent", "genes_total"] if c in df.columns]
    if numeric:
        print(df[numeric].describe())
