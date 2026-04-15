# BaClasT: Bacterial Classification Tool using k-mer frequency profiling and Random Forest classification

## Manuscript notes and design decisions

> This document captures the rationale behind every major design decision in BaClasT, structured to support writing a bioinformatics publication. Each section maps to a standard manuscript section and includes the reasoning discussed during development.

---

## Abstract (draft)

BaClasT is a lightweight, dependency-minimal tool for classifying assembled bacterial genomes into ESKAPE pathogen species using tetramer (4-mer) frequency profiling and a Random Forest classifier. The tool achieves species-level discrimination from a 256-dimensional feature vector computed directly from genome assemblies without alignment to a reference. BaClasT includes a centroid-based novelty detection layer to identify out-of-distribution genomes that do not belong to any ESKAPE species, addressing a critical limitation of closed-set classifiers. All genome selection, subsampling, and model training parameters are tracked for full reproducibility.

---

## Introduction

### Problem statement

Rapid taxonomic identification of bacterial pathogens from genome assemblies is a routine need in clinical microbiology and bioinformatics. The ESKAPE pathogens (*Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, and *Enterobacter cloacae*) are responsible for the majority of hospital-acquired infections and represent the greatest challenge for antibiotic resistance. Tools that can rapidly and accurately classify assembled genomes into these species — without requiring large reference databases, alignment, or specialised bioinformatics infrastructure — are valuable for clinical and research settings.

### Existing approaches and motivation

Established tools such as Kraken2, GTDB-Tk, and BLAST-based methods are highly accurate but require large pre-built databases (often tens of gigabytes), conda environments, or significant computational resources. BaClasT was designed with a different philosophy: a pure Python tool with no bioinformatics stack dependencies that runs on a standard laptop or HPC login node. The approach trades the generality of database-driven methods for speed and simplicity in a focused classification task.

### Why k-mer frequencies

K-mer frequency profiling is well-established for taxonomic classification (Wood & Salzberg, 2014; Ounit et al., 2015). Different bacterial species have evolved distinct nucleotide composition biases due to differences in codon usage, GC content, and repetitive element distribution. These biases are captured by k-mer frequency vectors even without alignment. At the whole-genome scale (typically 2-7 Mb for ESKAPE pathogens), k-mer frequency signals are highly reproducible within species and discriminating between species.

---

## Methods

### 2.1 Genome acquisition and curation

#### Source

All genome assemblies were obtained from NCBI RefSeq using the NCBI datasets CLI tool. Only **complete genome** assemblies were included (assembly_level = "Complete Genome") to ensure consistent, high-quality input sequences.

#### Genome selection strategy

**Decision: Random subsampling over exhaustive download or phylogenetic balancing.**

For each ESKAPE species, NCBI contains hundreds to thousands of complete genome assemblies (e.g., 3,043 for *P. aeruginosa* at time of download). Three selection strategies were considered:

1. **Download all available genomes.** Rejected because: (a) massive class imbalance between species with thousands of assemblies and those with fewer; (b) overrepresentation of dominant clinical clones that are oversampled in public databases (e.g., *P. aeruginosa* PAO1/PA14 lineages), biasing the k-mer profile toward specific lineages rather than the species broadly; (c) diminishing returns — 256 features do not require thousands of training examples.

2. **Phylogenetic tree construction with clade-balanced sampling.** Rejected as disproportionate to the task. Phylogenetic balancing is appropriate for strain-level or sub-species discrimination but unnecessary for species-level classification, where between-species k-mer distances are large relative to within-species variation.

3. **Deterministic random subsampling (chosen).** A reproducible random subsample of N genomes per species, controlled by a seed parameter. This approach provides sufficient diversity for species-level classification while keeping the training set manageable and balanced.

#### Subsampling algorithm

The subsampling algorithm was designed with two properties:

- **Reproducibility:** The same seed and sample size always produces the same genome set.
- **Incremental expansion:** A subsample of size N is guaranteed to be a strict subset of a subsample of size M (M > N) when using the same seed. This enables expanding the training set without re-downloading previously acquired genomes.

Implementation: all available genome indices are shuffled using the provided random seed, then the first N are selected. This differs from `random.sample()`, which does not guarantee the subset property.

#### Sample sizes

[TO BE FILLED: final numbers per species after download]

Approximately 50-100 genomes per species were targeted as sufficient for 256-dimensional feature space. The `class_weight='balanced'` parameter in the Random Forest handles any remaining imbalance.

#### Provenance tracking

Every genome download is fully tracked for reproducibility:

| Artefact | Contents |
|---|---|
| `genomes/metadata/DDMMYY_XX_N_S.json` | Raw NCBI datasets API response (JSONL) |
| `genomes/metadata/DDMMYY_XX_N_S.csv` | Parsed metadata table (accession, strain, genome size, GC%, contig count, gene count) |
| `genomes/<Species>/manifest.json` | Download date, seed, subsample size, assembly level filter, full accession list |

Filename convention: date (DDMMYY), two-letter species abbreviation (first letter of genus + species), subsample count, and random seed. Example: `130426_PA_50_42.json` = *Pseudomonas aeruginosa*, 50 genomes, seed 42, queried 13 April 2026.

### 2.2 Feature extraction

#### K-mer vocabulary

All 4^4 = 256 possible tetramers (4-mers) over the DNA alphabet {A, C, G, T} are used as features, generated in lexicographic order using `itertools.product("ACGT", repeat=4)`. This vocabulary is fixed at training time and stored in the model file to ensure identical feature ordering at prediction time.

**Decision: k=4 (tetramers) over k=3 or k=5.**

- k=3 yields only 64 features — insufficient to capture species-level composition differences.
- k=5 yields 1,024 features — more discriminating but computationally heavier and more prone to overfitting with smaller training sets.
- k=4 (256 features) provides the best trade-off for ESKAPE classification with Random Forest.

The architecture supports arbitrary k values; a comparison utility is planned to empirically validate this choice across datasets.

#### Sequence preprocessing

1. All FASTA records (contigs) in a genome assembly are concatenated with `N` separators. The `N` character is not in the DNA alphabet {A, C, G, T}, so no spurious k-mers span contig boundaries.
2. All sequences are uppercased before counting (Biopython may return lowercase for soft-masked regions).
3. K-mers containing any non-ACGT character (N, R, Y, W, S, etc.) are skipped. Only k-mers composed entirely of canonical bases are counted.

#### Normalisation

Raw k-mer counts are L1-normalised to frequencies (count / total valid k-mers). This is critical because genome sizes vary substantially across ESKAPE species (2-7 Mb); raw counts would cause larger genomes to dominate the feature space.

### 2.3 Classification model

#### Random Forest

A Random Forest classifier (`sklearn.ensemble.RandomForestClassifier`) was chosen for the following reasons:

- Handles 256-dimensional feature spaces well without feature selection
- Robust to overfitting with balanced class weights
- Requires no feature scaling (tree-based)
- Provides per-class probability estimates for confidence scoring
- Supports parallel training across all CPU cores (`n_jobs=-1`)

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Sufficient for convergence on 256 features; diminishing returns beyond this |
| `class_weight` | `'balanced'` | Automatically adjusts weights inversely proportional to class frequencies |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Use all available cores |

#### Train/test split

An 80/20 stratified split (`sklearn.model_selection.train_test_split`, `stratify=y`, `random_state=42`) is used for evaluation. Stratification ensures each species is proportionally represented in both sets.

#### Cross-validation

Optional stratified k-fold cross-validation (`StratifiedKFold`, `shuffle=True`, `random_state=42`) reports mean +/- standard deviation accuracy across folds, providing a more robust estimate of generalisation performance than the single train/test split.

### 2.4 Negative class ("Other")

#### Motivation

A classifier trained exclusively on ESKAPE species will always assign one of six labels to any input genome. If presented with a non-ESKAPE genome (e.g., *Escherichia coli*), it will assign whichever ESKAPE species has the most similar k-mer profile — potentially with high confidence if the query is phylogenetically close to an ESKAPE species (e.g., *E. coli* classified as *Klebsiella* due to shared Enterobacteriaceae k-mer composition).

To address this, a seventh "Other" class was added to the training set, containing diverse non-ESKAPE bacterial genomes. This converts the classifier from a closed-set to a semi-open-set system.

#### Scope and curation

The Other class was curated from 40 clinically relevant bacterial species spanning 8+ phyla:

- **Firmicutes:** *S. epidermidis*, *E. faecalis*, *S. pneumoniae*, *S. pyogenes*, *L. monocytogenes*, *B. subtilis*, *B. cereus*, *C. difficile*, *C. perfringens*
- **Gammaproteobacteria:** *E. coli*, *S. enterica*, *S. marcescens*, *P. mirabilis*, *C. freundii*, *M. morganii*, *H. influenzae*, *L. pneumophila*, *V. cholerae*, *S. maltophilia*, *Y. pestis*, *P. putida*, *P. fluorescens*, *K. oxytoca*, *A. lwoffii*
- **Betaproteobacteria:** *N. meningitidis*, *N. gonorrhoeae*, *B. cenocepacia*, *B. pertussis*
- **Alphaproteobacteria:** *R. prowazekii*
- **Epsilonproteobacteria:** *C. jejuni*, *H. pylori*
- **Actinobacteria:** *M. tuberculosis*, *C. diphtheriae*, *C. acnes*
- **Bacteroidetes:** *B. fragilis*
- **Spirochaetes:** *T. pallidum*, *B. burgdorferi*
- **Chlamydiae:** *C. trachomatis*
- **Fusobacteria:** *F. nucleatum*
- **Tenericutes:** *M. pneumoniae*

**Key design decisions for the Other class:**

1. **Close ESKAPE relatives are deliberately included** (*P. putida*, *P. fluorescens*, *K. oxytoca*, *A. lwoffii*, *S. epidermidis*, *E. faecalis*). These are the hardest negatives — species whose k-mer profiles are most similar to ESKAPE targets. Including them teaches the classifier where the species boundary lies, rather than relying on phylogenetic distance alone.

2. **Clinically relevant species were prioritised** over random bacterial diversity. This is defensible for a clinical tool: the negative class should represent what a clinical lab is likely to encounter, not arbitrary environmental organisms.

3. **Extremes of GC content are included** (*M. tuberculosis* at ~65% GC, *M. pneumoniae* at ~40% GC) to span the full range of bacterial nucleotide composition.

4. **Selection is reproducible** using the same seed/subsample mechanism as ESKAPE downloads, with all parameters tracked in a manifest.

#### Class size balancing

The Other class size should approximately match the total ESKAPE genome count to avoid training imbalance. With `class_weight='balanced'`, the Random Forest adjusts for any remaining disproportion, but roughly equal representation is preferred for centroid and distance threshold calibration.

| ESKAPE per species | Total ESKAPE | Recommended Other | Genomes per Other species |
|---|---|---|---|
| 50 | 300 | ~320 | 8 |
| 100 | 600 | ~600 | 15 |

### 2.5 Novelty detection (centroid-based out-of-distribution scoring)

#### Motivation

Even with the Other class, the Random Forest may misclassify a novel organism not represented in either the ESKAPE or Other training sets. A second layer of out-of-distribution (OOD) detection was added using centroid-based distance scoring, operating independently of the RF classification.

#### Method

1. **Centroid computation:** During training, the mean k-mer frequency vector (centroid) is computed for each class (including Other). This represents the "typical" k-mer profile for that species in 256-dimensional space.

2. **Distance scoring:** At prediction time, the Euclidean distance from the query genome's k-mer vector to every class centroid is computed. The nearest centroid and its distance are reported alongside the RF prediction.

3. **Threshold calibration:** The OOD threshold is set at the 99th percentile of within-class distances observed during training. For every training genome, the distance to its own class centroid is computed; the 99th percentile of these distances defines the boundary beyond which a query is flagged as out-of-distribution.

#### Two-layer detection

The RF prediction and centroid distance operate as complementary layers:

| RF prediction | Centroid distance | Interpretation |
|---|---|---|
| ESKAPE species, high confidence | Below threshold | **Trust the prediction** |
| ESKAPE species, high confidence | Above threshold | **Suspicious** — may be a close relative not in training set |
| "Other", any confidence | Below Other centroid | **Non-ESKAPE**, correctly rejected |
| Any class, low confidence | Above threshold | **Out-of-distribution** — genome not well-represented by any training class |

#### Centroid visualisation

PCA projection of the class centroids to 2D provides an interpretable view of species relationships in the k-mer feature space used by the classifier. This is distinct from phylogenetic distance — it shows how the *classifier* perceives species similarity based on nucleotide composition. A pairwise distance heatmap provides quantitative inter-species distances.

### 2.6 Model persistence

The trained model is serialised as a Python dictionary using joblib, containing all information needed for prediction:

| Key | Type | Purpose |
|---|---|---|
| `classifier` | RandomForestClassifier | Fitted sklearn model |
| `label_names` | list[str] | Species names, ordered by label index |
| `k` | int | K-mer length used during training |
| `kmer_vocab` | list[str] | Ordered k-mer vocabulary (length 4^k) |
| `centroids` | dict[str, ndarray] | Per-class mean k-mer vectors |
| `distance_threshold` | float | 99th percentile OOD threshold |
| `baclasp_version` | str | Software version at training time |
| `trained_at` | str | ISO 8601 training timestamp |
| `n_genomes` | int | Total training genomes |

Storing the vocabulary in the model file ensures prediction always uses the same feature ordering as training, even if the software version changes.

---

## Software architecture

### Design philosophy

BaClasT is split into two packages:

- **Builder** (`src/builder/`): CLI tool for genome acquisition and curation. Wraps the NCBI datasets CLI to download, subsample, and track genome sets. Designed for terminal use.
- **Classifier** (`src/classifier/`): Pure library code for feature extraction, model training, evaluation, novelty detection, and visualisation. Designed for interactive use in IPython/REPL or Jupyter notebooks.

This separation allows researchers to inspect, modify, and re-run any step of the analysis interactively, while providing reproducible CLI workflows for genome acquisition.

### Technology choices

| Component | Choice | Rationale |
|---|---|---|
| Genome parsing | Biopython (Bio.SeqIO) | Handles multi-contig FASTAs, compressed files, edge cases |
| K-mer counting | collections.Counter | No native dependencies, fast enough for assembled genomes |
| Classification | scikit-learn RandomForest | Probability estimates, robust to class imbalance, no GPU needed |
| Model persistence | joblib | scikit-learn recommended, efficient for numpy arrays |
| Metadata handling | pandas | Tabular summaries of NCBI genome metadata |
| Visualisation | matplotlib + sklearn PCA | Standard scientific Python stack |
| Package management | uv | Fast, lockfile-based reproducibility |

**Deliberately excluded:** pandas in the classifier (not needed for k-mer operations), tensorflow/pytorch (overkill for 256-feature RF), conda-only packages, any package requiring a C compiler without a pre-built wheel.

### Reproducibility infrastructure

All parameters affecting genome selection and model training are captured:

1. **Genome selection:** seed, subsample size, assembly level, date, accession lists (manifest.json)
2. **Feature extraction:** k-mer length, vocabulary ordering (stored in model file)
3. **Model training:** n_estimators, random_state, train/test split ratio and seed
4. **Software versions:** pinned in uv.lock, BaClasT version in model payload

---

## Results

[TO BE FILLED after training and evaluation]

### Expected sections:

- Classification accuracy on held-out test set (overall and per-species)
- Confusion matrix showing species-level misclassification patterns
- Cross-validation accuracy (mean +/- std)
- Centroid PCA plot showing species clustering in k-mer space
- Pairwise centroid distance heatmap
- Novelty detection performance: separation between in-distribution and OOD genomes
- Comparison of k=3, k=4, k=5 (if k-mer comparison utility is implemented)

---

## Discussion

### Key points to address

1. **K-mer profiling is sufficient for ESKAPE species-level classification.** The 256-dimensional tetramer frequency vector captures enough nucleotide composition signal to discriminate six species with high accuracy. This is consistent with prior work showing that k-mer profiles are species-discriminating for bacteria (e.g., Pride et al., 2003; Chor et al., 2009).

2. **The Other class and centroid distance provide complementary OOD detection.** The RF Other class handles known non-ESKAPE species seen during training; the centroid distance catches novel organisms not in any training class. Together they provide robust rejection of non-ESKAPE inputs.

3. **Close relatives are the critical test.** The most informative elements of the Other class are species closely related to ESKAPE targets (*P. putida* vs *P. aeruginosa*, *K. oxytoca* vs *K. pneumoniae*, etc.). These test whether the classifier has learned genuine species-level boundaries rather than family-level composition differences.

4. **Limitations:**
   - The tool classifies to species level only; it does not perform strain typing or resistance gene detection.
   - Performance depends on the quality and completeness of input assemblies; highly fragmented or contaminated assemblies may produce unreliable k-mer profiles.
   - The Other class, while diverse, does not cover all bacterial species; a truly novel organism may still receive a confident ESKAPE prediction if its k-mer profile happens to fall within a species cluster.
   - The 99th percentile distance threshold is a heuristic; optimal thresholding may require calibration on a validation set of known OOD genomes.

5. **The tool is designed for simplicity and accessibility.** Unlike database-driven tools, BaClasT requires no pre-built databases, no conda environments, and runs in seconds on a laptop. This makes it suitable for quick triage in clinical settings or as a teaching tool for bioinformatics courses.

---

## References

[TO BE COMPLETED]

- Chor, B., Horn, D., Goldman, N., Levy, Y., & Massingham, T. (2009). Genomic DNA k-mer spectra: models and modalities. *Genome Biology*, 10(10), R108.
- Cock, P. J. A., et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*, 25(11), 1422-1423.
- Ounit, R., Wanamaker, S., Close, T. J., & Lonardi, S. (2015). CLARK: fast and accurate classification of metagenomic and genomic sequences using discriminative k-mers. *BMC Genomics*, 16(1), 236.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Pride, D. T., Meinersmann, R. J., Wassenaar, T. M., & Blaser, M. J. (2003). Evolutionary implications of microbial genome tetranucleotide frequency biases. *Genome Research*, 13(2), 145-158.
- Rice, L. B. (2008). Federal funding for the study of antimicrobial resistance in nosocomial pathogens: no ESKAPE. *Journal of Infectious Diseases*, 197(8), 1079-1081.
- Wood, D. E., & Salzberg, S. L. (2014). Kraken: ultrafast metagenomic sequence classification using exact alignments. *Genome Biology*, 15(3), R46.
