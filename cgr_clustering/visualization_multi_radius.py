import matplotlib.pyplot as plt
import numpy as np

from cgr_clustering.utils import distance_matrix


def _upper_triangle_values(M, k=1):
    """Return upper-triangle values of a square matrix (k=1 excludes diagonal)."""
    M = np.asarray(M, dtype=float)
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "Matrix must be square"
    iu = np.triu_indices(M.shape[0], k=k)
    v = M[iu]
    v = v[np.isfinite(v)]
    return v


def _as_similarity(values, input_is_distance=True):
    """Convert distances in [0, 1] to similarities when requested."""
    values = np.asarray(values, dtype=float)
    if input_is_distance:
        return 1.0 - values
    return values


def _smoothed_hist_density(x, bins=200, smooth_sigma_bins=2.0, xlim=(0, 1)):
    """Fast KDE-like curve via histogram density + Gaussian smoothing (no scipy)."""
    x = np.asarray(x, dtype=float)
    lo, hi = xlim
    x = x[(x >= lo) & (x <= hi)]
    if x.size == 0:
        grid = np.linspace(lo, hi, 200)
        return grid, np.zeros_like(grid)

    counts, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if smooth_sigma_bins and smooth_sigma_bins > 0:
        radius = int(max(3, np.ceil(4 * smooth_sigma_bins)))
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / smooth_sigma_bins) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(counts, kernel, mode="same")
    else:
        smooth = counts

    return centers, smooth


def plot_tanimoto_similarity_distributions_multi(
    matrices,
    labels=None,
    input_is_distance=True,
    bins=220,
    smooth_sigma_bins=2.0,
    xlim=(0, 1),
    fill_alpha=0.20,
    fill_alphas=None,
    line_styles=None,
):
    """
    Plot KDE-like distributions of Tanimoto similarities from multiple matrices.

    - matrices: list/tuple/dict of square matrices
    - input_is_distance=True converts values to similarity = 1 - distance
    - Uses upper triangle only (excludes diagonal) to avoid duplicates
    """
    if isinstance(matrices, dict):
        matrix_labels = [str(k) for k in matrices.keys()]
        matrix_values = list(matrices.values())
    else:
        matrix_values = list(matrices)
        matrix_labels = [f"Matrix {i + 1}" for i in range(len(matrix_values))]

    if not matrix_values:
        raise ValueError("At least one matrix is required.")

    if labels is None:
        labels = matrix_labels
    else:
        labels = list(labels)
        if len(labels) != len(matrix_values):
            raise ValueError(
                f"labels length ({len(labels)}) must match number of matrices ({len(matrix_values)})"
            )

    if fill_alphas is None:
        fill_alphas = [fill_alpha] * len(matrix_values)
    else:
        fill_alphas = list(fill_alphas)
        if len(fill_alphas) != len(matrix_values):
            raise ValueError(
                f"fill_alphas length ({len(fill_alphas)}) must match number of matrices ({len(matrix_values)})"
            )

    if line_styles is None:
        line_styles = ["-"] * len(matrix_values)
    else:
        line_styles = list(line_styles)
        if len(line_styles) != len(matrix_values):
            raise ValueError(
                f"line_styles length ({len(line_styles)}) must match number of matrices ({len(matrix_values)})"
            )

    plt.figure(figsize=(10, 6))
    for label, matrix, alpha, linestyle in zip(labels, matrix_values, fill_alphas, line_styles):
        values = _upper_triangle_values(matrix, k=1)
        sims = _as_similarity(values, input_is_distance=input_is_distance)
        x, y = _smoothed_hist_density(
            sims,
            bins=bins,
            smooth_sigma_bins=smooth_sigma_bins,
            xlim=xlim,
        )
        line, = plt.plot(x, y, label=label, linestyle=linestyle)
        plt.fill_between(x, 0, y, alpha=alpha, color=line.get_color())

    if input_is_distance:
        plt.xlabel("Distance")
        title = "Distribution of Tanimoto Distances"
    else:
        plt.xlabel("Similarity")
        title = "Distribution of Similarities"
    plt.title(title)
    plt.ylabel("Density")
    plt.xlim(*xlim)
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.show()


def plot_tanimoto_similarity_distributions_by_radius(
    sb_cgrs_dict,
    ab_dist_matrix,
    poss_radii=(2, 3, 4),
    length=4096,
    bins=80,
    smooth_sigma_bins=3.0,
    sb_fill_alpha=0.60,
    ab_fill_alpha=0.20,
    sb_dotted_lines=False,
    sb_line_style=":",
    title_1 = 'SB-CGR',
    title_2= 'ABS',
):
    """
    Build SB-CGR distance matrices for each radius, then plot all SB-CGR radii
    against Atom-Bond in one distribution figure.

    Styling:
    - SB-CGR curves use sb_fill_alpha (default 0.60)
    - Atom-Bond curve uses ab_fill_alpha (default 0.20)
    - Set sb_dotted_lines=True to make SB-CGR border lines dotted

    Returns:
        dict: {radius: SB-CGR distance matrix}
    """
    poss_radii = list(poss_radii)
    diff_sb_cgr_dists = {}
    for radius in poss_radii:
        diff_sb_cgr_dists[radius] = distance_matrix(sb_cgrs_dict, radius=radius, length=length)

    similarity_matrices = [1 - diff_sb_cgr_dists[r] for r in poss_radii] + [1 - ab_dist_matrix]
    labels = [f"{title_1} (r={r})" for r in poss_radii] + [f"{title_2}"]
    sb_styles = [sb_line_style if sb_dotted_lines else "-"] * len(poss_radii)
    line_styles = sb_styles + ["-"]
    fill_alphas = [sb_fill_alpha] * len(poss_radii) + [ab_fill_alpha]

    plot_tanimoto_similarity_distributions_multi(
        similarity_matrices,
        labels=labels,
        input_is_distance=False,
        bins=bins,
        smooth_sigma_bins=smooth_sigma_bins,
        fill_alphas=fill_alphas,
        line_styles=line_styles,
    )

    return diff_sb_cgr_dists
