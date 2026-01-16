import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment


def _extract_labels(first_dict, second_dict):
    """
    Internal helper to extract label maps and common items from two cluster dicts.
    """
    labels1 = {node: cid for cid, data in first_dict.items() for node in data['route_ids']}
    labels2 = {}
    for cid, nodes in second_dict.items():
        # second_dict may map to a list or to a dict with 'route_ids'
        if isinstance(nodes, dict) and 'route_ids' in nodes:
            node_list = nodes['route_ids']
        else:
            node_list = nodes
        for node in node_list:
            labels2[node] = cid
    common = sorted(set(labels1) & set(labels2))
    return labels1, labels2, common


def calculate_similarity_metrics(first_dict, second_dict, print_stats=False):
    """
    Compute ARI, NMI, entropy, MI, and VI between two clusterings.
    Returns y_true and y_pred lists aligned on the common item set.
    """
    labels1, labels2, common = _extract_labels(first_dict, second_dict)
    if not common:
        print("No common items found between the two clusterings.")
        return [], []

    y_true = [labels1[n] for n in common]
    y_pred = [labels2[n] for n in common]

    # External metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Info-theoretic metrics
    counts1 = np.unique(y_true, return_counts=True)[1]
    counts2 = np.unique(y_pred, return_counts=True)[1]
    H1 = entropy(counts1)
    H2 = entropy(counts2)
    MI = mutual_info_score(y_true, y_pred)
    VI = H1 + H2 - 2 * MI
    if print_stats:
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"Entropy first: {H1:.4f}")
        print(f"Entropy second: {H2:.4f}")
        print(f"Mutual Information: {MI:.4f}")
        print(f"Variation of Information (VI): {VI:.4f}\n")

    return y_true, y_pred


def match_clusters(first_dict, second_dict):
    """
    Match clusters by maximizing average Jaccard similarity using the Hungarian algorithm.
    Returns list of matched tuples (cid1, cid2, score) and the average score.
    """
    labels1, labels2, common = _extract_labels(first_dict, second_dict)

    # Build set-based clusters restricted to common items
    clusters1 = {cid: set(data['route_ids']) & set(common)
                 for cid, data in first_dict.items()}
    clusters2 = {cid: set(nodes) & set(common)
                 for cid, nodes in second_dict.items()}

    keys1, keys2 = list(clusters1), list(clusters2)
    J = np.zeros((len(keys1), len(keys2)))
    for i, c1 in enumerate(keys1):
        for j, c2 in enumerate(keys2):
            s1, s2 = clusters1[c1], clusters2[c2]
            if s1 or s2:
                J[i, j] = len(s1 & s2) / len(s1 | s2)

    # Hungarian: maximize J by minimizing (1-J)
    row_ind, col_ind = linear_sum_assignment(1 - J)
    matched = [(keys1[i], keys2[j], J[i, j]) for i, j in zip(row_ind, col_ind)]
    avg_jaccard = float(np.mean([score for *_, score in matched]))
    return matched, avg_jaccard


def compute_contingency_matrix(y_true, y_pred):
    """
    Create a pandas contingency table from two label lists.
    """
    return pd.crosstab(
        pd.Series(y_true, name='first_partition'),
        pd.Series(y_pred, name='second_partition')
    )


def contingency_nodes(first_dict, second_dict, row_key, col_key):
    """Return sorted list of nodes shared by two specific clusters."""
    s1 = set(first_dict[row_key]['route_ids'])
    s2 = set(second_dict[col_key])
    return sorted(s1 & s2)


def diagonalize_contingency(cm):
    """
    Reorder rows (first_partition) and columns (second_partition)
    so that the Hungarian matches are on the diagonal, sorted by
    the cell value |A∩B| (biggest first). Zeros are pushed to the end.

    Returns
    -------
    cm_sorted : pd.DataFrame
        Reordered contingency matrix.
    matches   : list of (row_label, col_label, jaccard, overlap)
        Matches in the order used for the diagonal.
    """
    # --- build Jaccard matrix from cm ---
    inter = cm.to_numpy(dtype=float)         # |A∩B|
    row_sum = inter.sum(axis=1, keepdims=True)
    col_sum = inter.sum(axis=0, keepdims=True)
    union = row_sum + col_sum - inter        # |A∪B|

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 0.0)

    # --- Hungarian: maximize Jaccard -> minimize (1 - J) ---
    row_ind, col_ind = linear_sum_assignment(1 - jaccard)

    row_labels = np.array(cm.index)
    col_labels = np.array(cm.columns)

    # collect (row, col, jaccard, overlap)
    matches = [
        (row_labels[i], col_labels[j], float(jaccard[i, j]), float(inter[i, j]))
        for i, j in zip(row_ind, col_ind)
    ]

    # sort: non-zero overlaps first, then by overlap desc, then by Jaccard desc
    matches.sort(key=lambda t: (t[3] == 0, -t[3], -t[2]))

    # new order for rows and columns from these matches
    new_rows = [r for r, _, _, _ in matches]
    new_cols = [c for _, c, _, _ in matches]

    matches = [t[:3] for t in matches]

    # (in case of rectangular / unused clusters, append remaining ones)
    new_rows += [r for r in cm.index   if r not in new_rows]
    new_cols += [c for c in cm.columns if c not in new_cols]

    cm_sorted = cm.loc[new_rows, new_cols]
    return cm_sorted, matches