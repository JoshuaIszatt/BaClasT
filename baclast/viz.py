"""Visualization tools for BaClasT — centroid plots and species relationships."""

import numpy as np
from sklearn.decomposition import PCA


def plot_centroids(
    centroids: dict[str, np.ndarray],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    label_names: list[str] | None = None,
    title: str = "Species centroids in k-mer space (PCA)",
    save_path: str | None = None,
):
    """Plot species centroids (and optionally individual genomes) in 2D PCA space.

    Args:
        centroids: Dict mapping species name to centroid vector.
        X: Optional feature matrix to plot individual genomes.
        y: Optional label array (required if X is provided).
        label_names: Optional label names (required if X is provided).
        title: Plot title.
        save_path: If provided, save the figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    names = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in names])

    # Fit PCA on centroids + individual genomes if available
    if X is not None:
        pca = PCA(n_components=2)
        all_data = np.vstack([X, centroid_matrix])
        all_transformed = pca.fit_transform(all_data)
        X_2d = all_transformed[: len(X)]
        centroids_2d = all_transformed[len(X) :]
    else:
        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(centroid_matrix)
        X_2d = None

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(names)))

    # Plot individual genomes if provided
    if X_2d is not None and y is not None and label_names is not None:
        for i, name in enumerate(label_names):
            mask = y == i
            if name in names:
                idx = names.index(name)
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=[colors[idx]], alpha=0.3, s=20, label=None,
                )

    # Plot centroids
    for i, name in enumerate(names):
        ax.scatter(
            centroids_2d[i, 0], centroids_2d[i, 1],
            c=[colors[i]], s=200, marker="*", edgecolors="black",
            linewidths=1, label=name, zorder=5,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_centroid_distances(
    centroids: dict[str, np.ndarray],
    title: str = "Pairwise centroid distances (k-mer space)",
    save_path: str | None = None,
):
    """Plot a heatmap of pairwise Euclidean distances between species centroids.

    Args:
        centroids: Dict mapping species name to centroid vector.
        title: Plot title.
        save_path: If provided, save the figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    names = sorted(centroids.keys())
    n = len(names)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(centroids[names[i]] - centroids[names[j]])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_matrix, cmap="YlOrRd")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Shorten labels for readability: "Pseudomonas_aeruginosa" -> "P. aeruginosa"
    short = [f"{s.split('_')[0][0]}. {s.split('_')[-1]}" if "_" in s else s for s in names]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)

    # Annotate cells with distance values
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{dist_matrix[i, j]:.4f}",
                ha="center", va="center", fontsize=7,
                color="white" if dist_matrix[i, j] > dist_matrix.max() * 0.6 else "black",
            )

    plt.colorbar(im, ax=ax, label="Euclidean distance")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
