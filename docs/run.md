# run.ipynb — Cell-by-cell guide

## Cell 1 — Load data

Imports the feature extraction module and loads all genome FASTAs from `genomes/`. Each subdirectory becomes a class label. The 4-mer vocabulary (256 tetramers) is generated once and reused throughout. Outputs the total genome count and class names.

## Cell 2 — Train/test split

Splits the dataset 80/20 using stratified sampling (`random_state=42`). Stratification ensures each species is proportionally represented in both sets. Prints the train and test counts.

## Cell 3 — Train

Trains a Random Forest classifier (200 trees, balanced class weights, all CPU cores). This is the core classification model.

## Cell 4 — Evaluate

Runs the trained model on the held-out test set. Prints:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix

The returned `results` dict contains these programmatically for further analysis.

## Cell 5 — Cross-validation

Runs 5-fold stratified cross-validation on the full dataset (not just the train split). Reports mean +/- standard deviation accuracy, giving a more robust generalisation estimate than the single train/test split.

## Cell 6 — Compute centroids and novelty threshold

Computes the mean k-mer vector (centroid) for each species from the training data. Then calibrates the out-of-distribution (OOD) threshold at the 99th percentile of within-class distances — any query genome further than this from all centroids is flagged as OOD.

## Cell 7 — Visualise centroids (PCA scatter)

Projects all genomes and their class centroids into 2D using PCA. Shows how species cluster in k-mer space and which species are closest to each other from the classifier's perspective. Centroids are marked with stars. Axis labels show explained variance.

## Cell 8 — Visualise pairwise distances (heatmap)

Displays a heatmap of Euclidean distances between all species centroids in the full 256-dimensional k-mer space. Annotated with distance values. Shows which species pairs the classifier will find hardest to distinguish.

## Cell 9 — Save model

Serialises the trained model, vocabulary, centroids, and threshold into a single `.pkl` file. This file contains everything needed for prediction — no retraining required. The genome count is patched in for provenance.

## Cell 10 — Predict a single genome

Loads a saved model and classifies an unknown genome FASTA. Reports:
- **RF prediction** — species name and confidence percentage
- **Novelty score** — distance to nearest centroid and whether it falls within the OOD threshold
- **Full probability breakdown** — ranked bar chart of all species probabilities
