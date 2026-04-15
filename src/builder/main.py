"""BaClasT genome builder CLI — download and curate training genomes."""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.builder.parser import describe_dataset, parse_genome_jsonl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_METADATA_DIR = _PROJECT_ROOT / "genomes" / "metadata"


def _taxon_abbreviation(taxon: str) -> str:
    """'Pseudomonas aeruginosa' -> 'PA', 'Enterobacter spp' -> 'ES'."""
    parts = taxon.strip().split()
    if len(parts) < 2:
        return parts[0][0].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _metadata_path_for_taxon(taxon: str, subsample: int | None, seed: int) -> Path:
    """Build path like genomes/metadata/130426_PA_50_42.json."""
    _METADATA_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%d%m%y")
    abbrev = _taxon_abbreviation(taxon)
    sub = subsample if subsample is not None else "all"
    return _METADATA_DIR / f"{date_str}_{abbrev}_{sub}_{seed}.json"


def _run_datasets_summary(taxon: str, assembly_level: str, output_path: Path) -> None:
    """Run `datasets summary genome taxon` and write JSONL to output_path."""
    cmd = [
        "datasets", "summary", "genome", "taxon", taxon,
        "--as-json-lines",
        "--assembly-level", assembly_level,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Error: datasets command failed:\n{result.stderr}")
    output_path.write_text(result.stdout)
    print(f"Saved genome summaries to {output_path}")


def _download_one(accession: str, output_dir: Path) -> bool:
    """Download a single genome FASTA. Returns True on success."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "ncbi_dataset.zip"

        cmd = [
            "datasets", "download", "genome", "accession", accession,
            "--include", "genome",
            "--filename", str(zip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False

        extract_dir = tmpdir / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        data_dir = extract_dir / "ncbi_dataset" / "data"
        if not data_dir.exists():
            return False

        for acc_dir in data_dir.iterdir():
            if not acc_dir.is_dir():
                continue
            for fna in acc_dir.glob("*.fna"):
                shutil.copy2(fna, output_dir / f"{accession}.fna")
                return True

    return False


def _download_genomes(accessions: list[str], output_dir: Path) -> None:
    """Download genome FASTAs one at a time with progress."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip accessions already downloaded (resume-friendly)
    # Only check against the requested accessions, not all files in the dir
    requested = set(accessions)
    already = {f.stem for f in output_dir.glob("*.fna") if f.stem in requested}
    remaining = [a for a in accessions if a not in already]
    if already:
        print(f"Skipping {len(already)} already downloaded, {len(remaining)} remaining.")

    total = len(accessions)
    done = len(already)
    failed = []

    for acc in remaining:
        done += 1
        print(f"  [{done}/{total}] {acc} ... ", end="", flush=True)
        if _download_one(acc, output_dir):
            print("ok")
        else:
            print("FAILED")
            failed.append(acc)

    total_in_dir = len(list(output_dir.glob("*.fna")))
    if remaining:
        print(f"\nDownloaded {len(remaining) - len(failed)} of {len(remaining)} genomes "
              f"({total_in_dir} total in {output_dir.name}/)")
    else:
        print(f"All {total} genomes already present ({total_in_dir} total in {output_dir.name}/)")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


def _resolve_path(user_path: str) -> Path:
    """Resolve a user-provided path relative to the project root."""
    p = Path(user_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def cmd_build(args):
    """Main build workflow: summarize, describe, confirm, download."""
    # Step 1: Get or load genome summary JSON
    if args.json:
        json_path = _resolve_path(args.json)
        if not json_path.exists():
            sys.exit(f"Error: JSON file not found: {json_path}")
    elif args.taxon:
        json_path = _metadata_path_for_taxon(args.taxon, args.subsample, args.seed)
        if json_path.exists():
            print(f"Metadata already exists: {json_path}")
        else:
            _run_datasets_summary(args.taxon, args.assembly_level, json_path)
    else:
        sys.exit("Error: provide either --taxon or --json")

    # Step 2: Parse, save CSV, and describe
    print(f"\nParsing {json_path}...")
    df = parse_genome_jsonl(json_path)
    csv_path = json_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved parsed metadata to {csv_path}")
    describe_dataset(df)

    # Step 3: Subsample
    if args.subsample and args.subsample < len(df):
        # Shuffle all indices with the seed, then take the first N.
        # This guarantees subsample(50) is a subset of subsample(100)
        # for the same seed — enabling incremental downloads.
        rng = random.Random(args.seed)
        all_indices = list(range(len(df)))
        rng.shuffle(all_indices)
        indices = sorted(all_indices[: args.subsample])
        selected = df.iloc[indices].copy()
        print(f"\nSubsampled {args.subsample} of {len(df)} genomes (seed={args.seed})")
    else:
        selected = df
        if args.subsample:
            print(f"\nRequested subsample ({args.subsample}) >= available ({len(df)}), using all.")

    accessions = selected["accession"].tolist()
    # Derive output dir from organism name in the data
    taxon_name = args.taxon if args.taxon else df["organism_name"].iloc[0]
    species_dir_name = taxon_name.strip().replace(" ", "_")
    output_dir = _PROJECT_ROOT / "genomes" / species_dir_name
    print(f"\nSelected {len(accessions)} accessions for download.")
    print(f"Output directory: {output_dir}")

    # Step 4: Confirm
    response = input("\nProceed with download? [Y/n] ").strip().lower()
    if response in ("n", "no"):
        print("Aborted.")
        sys.exit(0)

    # Step 5: Download
    _download_genomes(accessions, output_dir)

    # Step 6: Save manifest for provenance tracking
    manifest = {
        "taxon": args.taxon or df["organism_name"].iloc[0],
        "source_json": str(json_path),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "assembly_level": args.assembly_level,
        "total_available": len(df),
        "subsample_size": len(accessions),
        "seed": args.seed,
        "accessions": accessions,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest saved to {manifest_path}")


ESKAPE_TAXA = [
    "Enterococcus faecium",
    "Staphylococcus aureus",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Enterobacter cloacae",
]


def cmd_download_eskape(args):
    """Download genomes for all six ESKAPE species by calling cmd_build for each."""
    print(f"Downloading all ESKAPE species (subsample={args.subsample}, "
          f"seed={args.seed}, assembly-level={args.assembly_level})\n")

    for taxon in ESKAPE_TAXA:
        print(f"\n{'='*60}")
        print(f"  {taxon}")
        print(f"{'='*60}")

        # Build a namespace that looks like cmd_build expects
        build_args = argparse.Namespace(
            taxon=taxon,
            json=None,
            subsample=args.subsample,
            seed=args.seed,
            assembly_level=args.assembly_level,
        )
        cmd_build(build_args)

    print(f"\n{'='*60}")
    print("All ESKAPE species downloaded.")


_OTHER_SCOPE_FILE = _PROJECT_ROOT / "genomes" / "other_scope.json"


def cmd_download_other(args):
    """Download diverse non-ESKAPE genomes for the 'Other' negative class.

    Reads taxa from the scope file, randomly selects a subset of species
    (controlled by --subsample and --seed), then downloads one reference
    genome per selected species into genomes/Other/.
    """
    scope_path = _resolve_path(args.scope) if args.scope else _OTHER_SCOPE_FILE
    if not scope_path.exists():
        sys.exit(f"Error: scope file not found: {scope_path}")

    with open(scope_path) as f:
        scope = json.load(f)

    all_taxa = [entry["name"] for entry in scope["taxa"]]
    print(f"Scope file: {scope_path.name}")
    print(f"Total taxa in scope: {len(all_taxa)}")
    print(f"Excluded (ESKAPE): {', '.join(scope['excluded'])}\n")

    # Subsample species from the scope
    if args.subsample and args.subsample < len(all_taxa):
        rng = random.Random(args.seed)
        shuffled = list(all_taxa)
        rng.shuffle(shuffled)
        selected_taxa = sorted(shuffled[: args.subsample])
        print(f"Subsampled {args.subsample} of {len(all_taxa)} taxa (seed={args.seed})")
    else:
        selected_taxa = all_taxa
        print(f"Using all {len(selected_taxa)} taxa")

    print("\nSelected taxa:")
    for t in selected_taxa:
        print(f"  - {t}")

    # For each selected taxon, download 1 genome into genomes/Other/
    # We use cmd_build with subsample=1 so each species gets one representative
    genomes_per_species = args.genomes_per_species
    print(f"\nDownloading {genomes_per_species} genome(s) per species "
          f"into genomes/Other/")

    response = input("\nProceed? [Y/n] ").strip().lower()
    if response in ("n", "no"):
        print("Aborted.")
        sys.exit(0)

    for taxon in selected_taxa:
        print(f"\n--- {taxon} ---")
        build_args = argparse.Namespace(
            taxon=taxon,
            json=None,
            subsample=genomes_per_species,
            seed=args.seed,
            assembly_level=args.assembly_level,
        )
        # Override the output dir: all Other genomes go into genomes/Other/
        # We do this by temporarily running the build steps manually
        json_path = _metadata_path_for_taxon(taxon, genomes_per_species, args.seed)
        if json_path.exists():
            print(f"  Metadata exists: {json_path.name}")
        else:
            _run_datasets_summary(taxon, args.assembly_level, json_path)

        df = parse_genome_jsonl(json_path)
        csv_path = json_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        if df.empty:
            print(f"  No genomes found for {taxon}, skipping.")
            continue

        # Subsample genomes for this species
        if genomes_per_species < len(df):
            rng = random.Random(args.seed)
            all_indices = list(range(len(df)))
            rng.shuffle(all_indices)
            indices = sorted(all_indices[:genomes_per_species])
            selected = df.iloc[indices]
        else:
            selected = df

        accessions = selected["accession"].tolist()
        output_dir = _PROJECT_ROOT / "genomes" / "Other"
        _download_genomes(accessions, output_dir)

    # Save manifest for the Other class
    output_dir = _PROJECT_ROOT / "genomes" / "Other"
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = sorted(f.stem for f in output_dir.glob("*.fna"))
    manifest = {
        "description": "Non-ESKAPE negative training class",
        "scope_file": str(scope_path),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "assembly_level": args.assembly_level,
        "taxa_selected": selected_taxa,
        "genomes_per_species": genomes_per_species,
        "seed": args.seed,
        "total_downloaded": len(downloaded),
        "accessions": downloaded,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nOther class: {len(downloaded)} genomes in {output_dir}")
    print(f"Manifest saved to {manifest_path}")


def cmd_metadata_summary(args):
    """Summarize all metadata files in genomes/metadata/ by species."""
    csv_files = sorted(_METADATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No metadata CSVs found in {_METADATA_DIR}")
        return

    import pandas as pd

    print(f"Found {len(csv_files)} metadata file(s) in {_METADATA_DIR}\n")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        species = df["organism_name"].iloc[0] if not df.empty else "Unknown"
        n_available = len(df)

        # Check how many are actually downloaded
        species_dir_name = species.strip().replace(" ", "_")
        genome_dir = _PROJECT_ROOT / "genomes" / species_dir_name
        n_downloaded = len(list(genome_dir.glob("*.fna"))) if genome_dir.exists() else 0

        print(f"{'='*60}")
        print(f"  {species}")
        print(f"  Source: {csv_path.name}")
        print(f"{'='*60}")
        print(f"  Available:  {n_available:>6}")
        print(f"  Downloaded: {n_downloaded:>6}")

        numeric_cols = ["genome_size", "gc_percent", "n_contigs", "genes_total"]
        present = [c for c in numeric_cols if c in df.columns]
        if present:
            summary = df[present].describe().loc[["mean", "std", "min", "max"]]
            print(f"\n{summary.to_string()}")
        print()

    # Overall summary
    all_dfs = [pd.read_csv(f) for f in csv_files]
    total_available = sum(len(d) for d in all_dfs)
    total_downloaded = 0
    species_list = []
    for d in all_dfs:
        if d.empty:
            continue
        sp = d["organism_name"].iloc[0]
        species_list.append(sp)
        sp_dir = _PROJECT_ROOT / "genomes" / sp.strip().replace(" ", "_")
        if sp_dir.exists():
            total_downloaded += len(list(sp_dir.glob("*.fna")))

    print(f"{'='*60}")
    print(f"  TOTAL")
    print(f"{'='*60}")
    print(f"  Species:    {len(species_list):>6}")
    print(f"  Available:  {total_available:>6}")
    print(f"  Downloaded: {total_downloaded:>6}")


def main():
    parser = argparse.ArgumentParser(
        prog="baclast",
        description="BaClasT — build and curate genome training sets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build sub-command — single species
    build_parser = subparsers.add_parser(
        "build", help="Download and curate genomes for a species"
    )
    source = build_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--taxon", help="NCBI taxon name (e.g. 'Pseudomonas aeruginosa')")
    source.add_argument("--json", help="Path to existing NCBI datasets JSONL file")
    build_parser.add_argument(
        "--subsample", type=int, default=None,
        help="Number of genomes to randomly subsample",
    )
    build_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible subsampling (default: 42)",
    )
    build_parser.add_argument(
        "--assembly-level", default="complete",
        help="Assembly level filter for datasets CLI (default: complete)",
    )

    # download-eskape sub-command — all six species
    eskape_parser = subparsers.add_parser(
        "download-eskape", help="Download genomes for all ESKAPE species"
    )
    eskape_parser.add_argument(
        "--subsample", type=int, default=None,
        help="Number of genomes per species to randomly subsample",
    )
    eskape_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible subsampling (default: 42)",
    )
    eskape_parser.add_argument(
        "--assembly-level", default="complete",
        help="Assembly level filter for datasets CLI (default: complete)",
    )

    # download-other sub-command — negative class
    other_parser = subparsers.add_parser(
        "download-other", help="Download diverse non-ESKAPE genomes for the Other class"
    )
    other_parser.add_argument(
        "--scope", default=None,
        help="Path to scope JSON file (default: genomes/other_scope.json)",
    )
    other_parser.add_argument(
        "--subsample", type=int, default=None,
        help="Number of species to randomly select from the scope",
    )
    other_parser.add_argument(
        "--genomes-per-species", type=int, default=1,
        help="Number of genomes to download per species (default: 1)",
    )
    other_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible selection (default: 42)",
    )
    other_parser.add_argument(
        "--assembly-level", default="complete",
        help="Assembly level filter (default: complete)",
    )

    # metadata-summary sub-command
    subparsers.add_parser(
        "metadata-summary", help="Summarize all metadata files by species"
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "download-eskape":
        cmd_download_eskape(args)
    elif args.command == "download-other":
        cmd_download_other(args)
    elif args.command == "metadata-summary":
        cmd_metadata_summary(args)


if __name__ == "__main__":
    main()
