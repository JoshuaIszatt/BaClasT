# BaClasT -- Bacterial Classification Tool

Fast classification of assembled bacterial genomes into ESKAPE pathogen species using k-mer frequency profiling.

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

## What it classifies

ESKAPE pathogens (*E. faecium*, *S. aureus*, *K. pneumoniae*, *A. baumannii*, *P. aeruginosa*, *E. cloacae*) plus an "Other" class for non-ESKAPE bacteria. Includes centroid-based out-of-distribution detection.

## How it works

Computes 4-mer frequency profiles (256 features) from genome assemblies and classifies with a Random Forest. A bundled pre-trained model is included -- no training data or setup required.

## Requirements

Python >= 3.12, biopython, scikit-learn, joblib, numpy.

## License

MIT
