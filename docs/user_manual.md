# BaClasT User Manual

## Install

```bash
uv add baclast
```

## CLI

```bash
baclast --predict genome.fna
baclast --predict genomes/ -o results.csv
```

## Python

```python
import src.classifier as baclast

baclast.predict(file="genome.fna")
baclast.to_csv(baclast.predict(file="genome.fna"), "results.csv")
```

Accepts `.fasta`, `.fa`, `.fna` files. Pass a file for single prediction, a directory for batch. Results are CSV to stdout unless `-o` is given.

### Input
BaClasT validates input FASTAs before classification:
- Rejects non-DNA sequences (e.g. protein FASTAs)
- Rejects genomes < 500 Kb (fragments, not complete genomes)
- Rejects genomes > 15 Mb (larger than any known ESKAPE genome)
- In batch mode, invalid files are marked as `SKIPPED` in the output with the error in `confidence_warning`

### Output
| Column | Description |
|---|---|
| `filepath` | Full path to the input FASTA |
| `filename` | Filename only |
| `organism_prediction` | Predicted species (one of 6 ESKAPE species, "Other", or "SKIPPED") |
| `confidence` | Random Forest confidence for the top prediction (0-100%) |
| `confidence_warning` | `LOW` if confidence < 70%, empty otherwise |
| `nearest_centroid` | Closest species in k-mer centroid space |
| `distance` | Euclidean distance to nearest centroid |
| `threshold` | Calibrated out-of-distribution threshold (99th percentile of training distances) |
| `within_distribution` | "Yes" if distance <= threshold, "No" otherwise |
| `baclast_version` | Tool version used for this prediction |

### Interpretation

| Prediction | within_distribution | confidence_warning | Interpretation |
|---|---|---|---|
| ESKAPE species | Yes | | Confident identification |
| ESKAPE species | Yes | LOW | Ambiguous — check second-closest species |
| ESKAPE species | No | | Unusual genome — treat with caution |
| Other | Yes | | Not an ESKAPE pathogen |
| Other | No | | Unknown organism — not in training data |
| SKIPPED | | (error message) | File failed validation |
