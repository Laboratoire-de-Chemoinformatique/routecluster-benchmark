from collections import Counter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, SVG
from synplan.chem.reaction_routes.visualisation import cgr_display

from cgr_clustering.comparing import contingency_nodes

def plot_distribution(group_sizes, clusters=None, method='LSTM'):
    if len(group_sizes) > 10:

        # Count how often each value appears
        freq = Counter(group_sizes)
        values = sorted(freq.keys())
        counts = [freq[v] for v in values]

        # Plot
        plt.figure(figsize=(9, 6))
        plt.bar(values, counts, width=0.8)
        plt.xlabel('Size of Clusters (Number of routes)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {len(group_sizes)} Clusters by {method} method')
        plt.xticks(values)
        # plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.grid(False)

        # Add count labels above each bar
        for x, y in zip(values, counts):
            plt.text(x, y + max(counts)*0.01, str(y), ha='center', va='bottom')

        plt.tight_layout()

    else:
        sns.set_style("whitegrid")
        labels = clusters.keys()
        total_num = sum(group_sizes)
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            group_sizes,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=1.1,
            colors=sns.color_palette("pastel")
        )
        plt.setp(autotexts, size=13, weight="bold")
        ax.legend(wedges, labels, title=f"Clusters", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=13, title_fontsize=13)
        ax.set_title(f'Cluster size distribution - {method} for {total_num} routes')

    plt.show()

def display_contingency_matrix(matched, cm, method='TED'):
    """
    Plot a heatmap of the contingency matrix, highlighting matched pairs.

    Parameters
    ----------
    matched : list of (row_label, col_label, score)
        Output from your matching, where labels correspond to
        cm.index (rows) and cm.columns (cols).
    cm : pandas.DataFrame
        Contingency matrix (already reordered, e.g. cm_sorted).
    """
    # Labels and matrix
    row_labels = list(cm.index)
    col_labels = list(cm.columns)
    np_cm = cm.to_numpy()

    # Label -> index maps
    row_pos = {lab: i for i, lab in enumerate(row_labels)}
    col_pos = {lab: j for j, lab in enumerate(col_labels)}

    # Convert matches (labels) to matrix indices
    highlight = []
    for r_lab, c_lab, score in matched:
        if r_lab in row_pos and c_lab in col_pos:
            i = row_pos[r_lab]
            j = col_pos[c_lab]
            highlight.append((i, j, score))

    fig, ax = plt.subplots(figsize=(6 if method in ['TED', 'Atom-Bond'] else 12,
                                    5 if method in ['TED', 'Atom-Bond'] else 3))

    im = ax.imshow(np_cm, aspect='auto', cmap='Greys',
                   vmin=0, vmax=np_cm.max()+1)

    # Annotate all non-zero cells in grey
    for i in range(np_cm.shape[0]):
        for j in range(np_cm.shape[1]):
            if np_cm[i, j] > 0:
                ax.text(j, i, int(np_cm[i, j]),
                        ha='center', va='center', color='grey')

    # Add red highlights for matched pairs
    for i, j, score in highlight:
        if np_cm[i, j] == 0:
            continue  # nothing to highlight
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 linewidth=2, edgecolor='red',
                                 facecolor='none',
                                 alpha=0.5 + 0.5 * score)
        ax.add_patch(rect)
        ax.text(j, i, int(np_cm[i, j]),
                ha='center', va='center',
                color='red', fontweight='bold')

    # Formatting
    ax.set_title(f"Contingency Matrix ({method} vs SB–CGR)")
    ax.set_xlabel(f"{method} Clusters")
    ax.set_ylabel("SB–CGR Clusters")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Number of Routes', rotation=270, labelpad=15)
    # Major ticks = cell centers (for labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    # Minor ticks = cell boundaries (for grid)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)

    # Grid on boundaries, not centers
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.8)
    ax.grid(which='major', visible=False)  # important if a style turned it on
    ax.tick_params(which='minor', bottom=False, left=False)

    # Make sure the image extent matches the boundary grid
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)  # keep row 0 at the top
    plt.tight_layout()
    plt.show()

def plot_venn_diagram(synplanner_sb_cgrs, askcos_sb_cgrs, aizynth_sb_cgrs):

    from matplotlib_venn import venn3
    # Plot Venn diagram
    plt.figure(figsize=(8, 8))

    label_tools = ('SynPlanner', 'ASKCOS', 'AiZynthFinder')
    label_colors = ['g', 'r', 'b']
    venn = venn3([synplanner_sb_cgrs, askcos_sb_cgrs, aizynth_sb_cgrs], label_tools, set_colors =label_colors,)
    # plt.title('Overlap of SB-CGRs Across Planners')

    # Increase font size for the subset labels (numbers)
    for text in venn.subset_labels:
        if text: # Check if the text object exists
            text.set_fontsize(14) # Adjust the font size as needed

    # Increase font size for the set labels (SynPlanner, ASKCOS, AiZynthFinder)
    for t, c in zip(venn.set_labels, label_colors):
        if t:
            t.set_fontsize(16)
            t.set_color(c)
    plt.show()

def plot_sb_histogram(synplanner_perc, askcos_perc, aizynth_perc):
    # Data

    # All bond counts
    x_labels = sorted(set(synplanner_perc) | set(askcos_perc) | set(aizynth_perc))
    x = range(len(x_labels))
    width = 0.25

    # Y values aligned to x_labels; track presence
    data = {
        'SynPlanner': (synplanner_perc, [synplanner_perc.get(k, None) for k in x_labels], 'green', -width),
        'ASKCOS':     (askcos_perc,    [askcos_perc.get(k, None) for k in x_labels],    'red',   0),
        'AiZynthFinder': (aizynth_perc, [aizynth_perc.get(k, None) for k in x_labels], 'blue',  width)
    }

    plt.figure(figsize=(9, 5))

    for name, (dct, y_vals, color, offset) in data.items():
        # Plot bars
        plt.bar([i + offset for i in x], [v or 0 for v in y_vals], width=width, color=color, label=name)
        xs = [i + offset for i in x]
        # Markers
        for x_pos, v in zip(xs, y_vals):
            if v is None:
                # Gray X for missing
                plt.scatter([x_pos], [1], marker='x', c='gray', s=80, linewidths=2, zorder=5)
            elif v < 4:
                # Open circle for small bars
                plt.scatter([x_pos], [v + 1], marker='o',
                            facecolors='none', edgecolors=color,
                            s=80, linewidths=2, zorder=5)

    plt.xticks(x, x_labels)
    plt.xlabel('Number of strategic bonds')
    plt.ylabel('Percentage of synthetic routes')
    plt.title('Distribution of Strategic Bonds by Planning Method')
    plt.legend()
    plt.tight_layout()
    plt.show()


def _upper_triangle_values(M, k=1):
    """Return upper-triangle values of a square matrix (k=1 excludes diagonal)."""
    M = np.asarray(M, dtype=float)
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "Matrix must be square"
    iu = np.triu_indices(M.shape[0], k=k)
    v = M[iu]
    v = v[np.isfinite(v)]
    return v


def _as_similarity(values, input_is_distance=True):
    """
    If input is a distance in [0,1], convert to similarity = 1 - distance.
    If already similarity, return unchanged.
    """
    if input_is_distance:
        return 1.0 - values
    return values


def _smoothed_hist_density(x, bins=200, smooth_sigma_bins=2.0, xlim=(0, 1)):
    """
    Fast KDE-like curve via histogram density + Gaussian smoothing (no scipy).
    Returns (grid_centers, density).
    """
    x = np.asarray(x, dtype=float)
    lo, hi = xlim
    x = x[(x >= lo) & (x <= hi)]
    if x.size == 0:
        grid = np.linspace(lo, hi, 200)
        return grid, np.zeros_like(grid)

    counts, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Gaussian smoothing kernel in "bin space"
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        radius = int(max(3, np.ceil(4 * smooth_sigma_bins)))
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / smooth_sigma_bins) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(counts, kernel, mode="same")
    else:
        smooth = counts

    return centers, smooth


def plot_tanimoto_similarity_distributions(
    M1,
    M2,
    labels=("Matrix 1", "Matrix 2"),
    input_is_distance=True,     # set False if M1/M2 already store similarity
    bins=220,
    smooth_sigma_bins=2.0,
    xlim=(0, 1),
):
    """
    Plot KDE-like distributions of Tanimoto similarities from two matrices using pyplot.

    - M1, M2: square matrices of distances or similarities
    - input_is_distance=True assumes entries are distances in [0,1] and converts to similarity = 1 - distance
    - Uses upper triangle only (excludes diagonal) to avoid duplicates
    """
    v1 = _upper_triangle_values(M1, k=1)
    v2 = _upper_triangle_values(M2, k=1)

    s1 = _as_similarity(v1, input_is_distance=input_is_distance)
    s2 = _as_similarity(v2, input_is_distance=input_is_distance)

    x1, y1 = _smoothed_hist_density(s1, bins=bins, smooth_sigma_bins=smooth_sigma_bins, xlim=xlim)
    x2, y2 = _smoothed_hist_density(s2, bins=bins, smooth_sigma_bins=smooth_sigma_bins, xlim=xlim)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(x1, y1, label=labels[0])
    plt.fill_between(x1, 0, y1, alpha=0.25, color=line1.get_color())

    line2, = plt.plot(x2, y2, label=labels[1])
    plt.fill_between(x2, 0, y2, alpha=0.25, color=line2.get_color())

    
    if input_is_distance:
        plt.xlabel("Distance")
        title="Distribution of Tanimoto Distances"
        plt.title(title)
    else:
        plt.xlabel("Similarity")
        title = "Distribution of Similarities"
        plt.title(title)
    plt.ylabel("Density")
    plt.xlim(*xlim)
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.show()


def plot_distance_matrix_correlation(D1, D2, method_1="Method 1", method_2="Method 2"):
    """
    Scatter plot of upper-triangle distances + Pearson/Spearman correlations.
    D1, D2: square distance matrices with same shape.
    """
    D1 = np.asarray(D1, dtype=float)
    D2 = np.asarray(D2, dtype=float)
    assert D1.shape == D2.shape, f"Shape mismatch: {D1.shape} vs {D2.shape}"

    x = _upper_triangle_values(D1)
    y = _upper_triangle_values(D2)

    # Ensure matched filtering for finite pairs
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # Correlations
    pearson = float(np.corrcoef(x, y)[0, 1])

    # Spearman without scipy: rank-transform then Pearson
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    spearman = float(np.corrcoef(rx, ry)[0, 1])

    title = f"Distance matrices correlation: {method_1} vs {method_2}"
    # Plot
    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xlabel(f"Distances ({method_1})")
    plt.ylabel(f"Distances ({method_2})")
    plt.title(f"{title}\nPearson r={pearson:.3f} | Spearman ρ={spearman:.3f}")

    # Optional y=x reference line (scaled to data range)
    mn = np.nanmin([x.min(), y.min()])
    mx = np.nanmax([x.max(), y.max()])
    plt.plot([mn, mx], [mn, mx], linewidth=1)

    plt.tight_layout()
    plt.show()

    return {"pearson_r": pearson, "spearman_rho": spearman, "n_pairs": int(len(x))}


def plotly_dm_correlation(
    D1,
    D2,
    name1="Matrix 1",
    name2="Matrix 2",
    labels=None,          # optional: list/array of node ids (len = n)
    use_upper_triangle=True
):
    """
    Interactive Plotly scatter of distances (D1 vs D2).
    Click a dot to print the (row=i, col=j) indices from the distance matrix.
    Also supports box/lasso selection (shows first 20 selected pairs).

    Works best in Jupyter (FigureWidget + ipywidgets).
    """

    D1 = np.asarray(D1, dtype=float)
    D2 = np.asarray(D2, dtype=float)

    assert D1.ndim == 2 and D1.shape[0] == D1.shape[1], "D1 must be square"
    assert D2.shape == D1.shape, "D2 must have the same shape as D1"

    n = D1.shape[0]

    # Use upper triangle to avoid duplicate pairs (i,j) and (j,i)
    if use_upper_triangle:
        I, J = np.triu_indices(n, k=1)
        x = D1[I, J]
        y = D2[I, J]
    else:
        I, J = np.indices((n, n))
        I, J = I.ravel(), J.ravel()
        x, y = D1.ravel(), D2.ravel()

    # Drop non-finite pairs
    m = np.isfinite(x) & np.isfinite(y)
    I, J, x, y = I[m], J[m], x[m], y[m]

    # customdata stores the matrix indices (and optional labels)
    if labels is not None:
        labels = np.asarray(labels)
        assert len(labels) == n, "labels must have length n"
        Li = labels[I]
        Lj = labels[J]
        customdata = np.array(list(zip(I, J, Li, Lj)), dtype=object)
        hovertemplate = (
            f"{name1}: %{{x:.6f}}<br>"
            f"{name2}: %{{y:.6f}}<br>"
            "i=%{customdata[0]}, j=%{customdata[1]}<br>"
            "label_i=%{customdata[2]}<br>"
            "label_j=%{customdata[3]}<extra></extra>"
        )
    else:
        customdata = np.stack([I, J], axis=1)
        hovertemplate = (
            f"{name1}: %{{x:.6f}}<br>"
            f"{name2}: %{{y:.6f}}<br>"
            "i=%{customdata[0]}, j=%{customdata[1]}<extra></extra>"
        )

    fig = go.FigureWidget(
        data=[
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=5, opacity=0.35),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        ],
        layout=go.Layout(
            title=f"Distance matrices correlation: {name1} vs {name2}",
            xaxis_title=name1,
            yaxis_title=name2,
            dragmode="select",  # enables box/lasso selection
        ),
    )
    # Diagonal y = x line (reference)
    minv = float(np.nanmin(np.concatenate([x, y])))
    maxv = float(np.nanmax(np.concatenate([x, y])))

    m = np.isfinite(x) & np.isfinite(y)
    I, J, x, y = I[m], J[m], x[m], y[m]

    below = int(np.sum(y < x))
    above = int(np.sum(y > x))
    equal = int(np.sum(np.isclose(y, x)))  # optional
    total = len(x)

    print(f"Total dots: {total}")
    print(f"Below diagonal (y<x): {below} ({below/total:.2%})")
    print(f"Above diagonal (y>x): {above} ({above/total:.2%})")
    print(f"On diagonal (y≈x):    {equal} ({equal/total:.2%})")

    fig.add_shape(
        type="line",
        x0=minv, y0=minv,
        x1=maxv, y1=maxv,
        xref="x", yref="y",
        line=dict(width=2, dash="dash"),
        layer="below",  # keep it behind the points
    )

    # Output widgets
    out = widgets.Output(layout={"border": "1px solid #ddd", "padding": "6px"})
    sel = widgets.HTML()

    def on_click(trace, points, state):
        if not points.point_inds:
            return
        k = points.point_inds[0]

        cd = customdata[k]
        i, j = int(cd[0]), int(cd[1])

        with out:
            out.clear_output()
            if labels is not None:
                print(
                    f"Clicked: (i={i}, j={j}) | "
                    f"label_i={cd[2]} label_j={cd[3]} | "
                    f"{name1}={x[k]:.6f} {name2}={y[k]:.6f}"
                )
            else:
                print(
                    f"Clicked: (i={i}, j={j}) | {name1}={x[k]:.6f} {name2}={y[k]:.6f}"
                )

    def on_select(trace, points, selector):
        inds = points.point_inds
        if not inds:
            sel.value = ""
            return
        preview = inds[:20]
        pairs = [(int(customdata[k][0]), int(customdata[k][1])) for k in preview]
        more = "" if len(inds) <= 20 else f" … (+{len(inds)-20} more)"
        sel.value = f"<b>Selected {len(inds)} points</b>: {pairs}{more}"

    fig.data[0].on_click(on_click)
    fig.data[0].on_selection(on_select)
    fig.update_layout(width=800, height=800)

    ui = widgets.VBox([fig, sel, out])
    display(ui)

    return fig, ui

def show_routes_in_cell(sbp_clusters, clusters_other, sb_cgr_ind, other_ind, images, first_n=10):
    sb_cgr = sbp_clusters[sb_cgr_ind]['sb_cgr']
    sb_cgr.clean2d()
    display(SVG(cgr_display(sb_cgr)))
    routes_list = contingency_nodes(sbp_clusters, clusters_other, sb_cgr_ind, other_ind) 
    print('Number of routes in cell', len(routes_list))
    print('Routes ids in AiZynthFinder list', routes_list)
    for route in routes_list[:first_n]:
        print('ID:',route)
        # display(route_collections[route].make_images()[0])
        display(images[route])
