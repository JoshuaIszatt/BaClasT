"""Shared helpers for BaClasT — file loading, logging."""

import logging
from pathlib import Path

FASTA_EXTENSIONS = {".fasta", ".fa", ".fna"}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Return a configured logger. INFO by default, DEBUG if verbose."""
    logger = logging.getLogger("baclast")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def print_banner():
    """Print BaClasT ASCII banner and version to stdout."""
    banner = r"""
 ____        ____ _           _____
| __ )  __ / ___| | __ _ __|_   _|
|  _ \ / _` | |   | |/ _` / __|| |
| |_) | (_| | |___| | (_| \__ \| |
|____/ \__,_|\____|_|\__,_|___/|_|
  Bacterial Classification Tool v0.1.0
"""
    print(banner)


def find_fasta_files(directory: Path) -> list[Path]:
    """Return all .fasta/.fa/.fna files in a directory (non-recursive)."""
    return sorted(
        f for f in directory.iterdir() if f.suffix in FASTA_EXTENSIONS
    )
