from IPython.display import display, SVG
from multitarget.utils import detect_suffixes_cluster
from synplan.chem.reaction_routes.visualisation import cgr_display
from synplan.utils.visualisation import get_route_svg
import re
import matplotlib.pyplot as plt
import numpy as np

def _safe_list(x):
    return x if isinstance(x, list) else []

def show_n_routes_per_target_from_grouped_row(
    final_df_grouped_sb,
    row_idx: int,
    data: dict,
    n_routes: int = 5,
    route_list_prefix: str = "route_id",   # expects route_id_A_list columns
    target_prefix: str = "target",
    cluster_prefix: str = "cluster_id",    # e.g., cluster_id_A
    routecgr_dict_key: str = "all_route_cgrs",
    show_routecgr: bool = True,
):
    """
    For one row in final_df_grouped_sb:
      For each suffix (A/B/C/...):
        Target X:
          SB-CGR (cluster_id_X)
          Route 1 pathway (+ optional RouteCGR)
          Route 2 pathway (+ optional RouteCGR)
          ...
    """

    row = final_df_grouped_sb.loc[row_idx]

    # detect suffixes (use your helper)
    suffixes = detect_suffixes_cluster(final_df_grouped_sb)
    suffixes = [s for s in suffixes if f"{route_list_prefix}_{s}_list" in final_df_grouped_sb.columns]
    if not suffixes:
        raise KeyError("No columns like route_id_A_list / route_id_B_list found.")

    # show score first if present
    for sc in ["route_cgr_score", "score", "sbscore"]:
        if sc in final_df_grouped_sb.columns:
            print(f"{sc}: {row.get(sc)}")
            break
    print("=" * 120)

    for suf in suffixes:
        t = row.get(f"{target_prefix}_{suf}", None)
        cl = row.get(f"{cluster_prefix}_{suf}", None)
        rlist = _safe_list(row.get(f"{route_list_prefix}_{suf}_list", []))

        # normalize
        cl = None if cl is None or (isinstance(cl, float) and str(cl) == "nan") else str(cl)
        rlist = [int(x) for x in rlist if x is not None]

        print(f"Target {suf}: {t}")
        print(f"Cluster {suf}: {cl}")
        print("-" * 120)

        # ---- SB-CGR (once) ----
        if t is not None and cl is not None:
            try:
                sb = data[t]["clusters"][cl]["sb_cgr"]
                sb.clean2d()
                try:
                    display(SVG(cgr_display(sb)))
                except Exception:
                    display(sb)
            except Exception as e:
                print(f"(SB-CGR display failed: {e})")
        else:
            print("(Missing target or cluster_id; cannot show SB-CGR.)")

        # ---- Routes (first N) ----
        if not rlist:
            print("(No routes in route list.)")
            print("=" * 120)
            continue

        n_show = min(n_routes, len(rlist))
        for i, rid in enumerate(rlist[:n_show], start=1):
            print(f"Route {i}: route_id={rid}")

            tree = data[t]["tree"]
            # Pathway
            try:
                display(SVG(get_route_svg(tree, int(rid))))
            except Exception as e:
                print(f"  (Pathway display failed: {e})")

            # Optional RouteCGR
            if show_routecgr:
                try:
                    rcgr = data[t].get(routecgr_dict_key, {}).get(int(rid), None)
                    if rcgr is None:
                        print("  (RouteCGR missing)")
                    else:
                        rcgr.clean2d()
                        try:
                            display(SVG(cgr_display(rcgr)))
                        except Exception:
                            display(rcgr)
                except Exception as e:
                    print(f"  (RouteCGR display failed: {e})")

            print("-" * 120)

        print("=" * 120)


def _suffixes_from_df(df, prefix):
    # returns suffixes for columns like f"{prefix}_{suffix}"
    sfx = []
    pat = re.compile(rf"^{re.escape(prefix)}_(.+)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            sfx.append(m.group(1))
    return sorted(set(sfx))


def show_tuple_row_sb(
    tuples_df_unique,
    row_idx: int,
    data: dict,
    sb_cluster_col_prefix: str = "cluster_id",
    route_id_col_prefix: str = "route_id",
    target_col_prefix: str = "target",
    routecgr_dict_key: str = "all_route_cgrs",
    show_routecgr: bool = True,
):
    """
    Given tuples_df_unique + a row index:
      - prints target/cluster_id/route_id for each suffix (A/B/C/...)
      - displays SB-CGR for the selected cluster_id
      - displays pathway SVG for the selected route_id
      - optionally displays RouteCGR (if present)

    Assumptions:
      data[target]['clusters'][cluster_id]['sb_cgr'] exists
      data[target]['tree'] exists
      data[target][routecgr_dict_key][route_id] exists (dict: route_id -> RouteCGR)
      You have:
        - get_route_svg(tree, route_id)
        - cgr_display(cgr) OR display(cgr) works
    """

    row = tuples_df_unique.loc[row_idx]

    # detect suffixes by route_id_* columns (most reliable)
    suffixes = _suffixes_from_df(tuples_df_unique, route_id_col_prefix)
    if not suffixes:
        raise KeyError(f"No columns like '{route_id_col_prefix}_X' found in tuples_df_unique.")

    print(f"Row index: {row_idx}")

    # print whichever score column exists
    if "route_cgr_score" in tuples_df_unique.columns:
        print("Route score:", row.get("route_cgr_score"))
    elif "score" in tuples_df_unique.columns:
        print("Score:", row.get("score"))

    if "min_tanimoto" in tuples_df_unique.columns:
        print("min_tanimoto:", row.get("min_tanimoto"))
    if "mean_tanimoto" in tuples_df_unique.columns:
        print("mean_tanimoto:", row.get("mean_tanimoto"))

    print("-" * 80)

    for suf in suffixes:
        tcol = f"{target_col_prefix}_{suf}"
        ccol = f"{sb_cluster_col_prefix}_{suf}"
        rcol = f"{route_id_col_prefix}_{suf}"

        # require at least target + route id to proceed
        if tcol not in tuples_df_unique.columns or rcol not in tuples_df_unique.columns:
            continue

        target = row.get(tcol, None)
        cluster_id = row.get(ccol, None) if ccol in tuples_df_unique.columns else None
        route_id = row.get(rcol, None)

        # normalize
        cluster_id = None if cluster_id is None or (isinstance(cluster_id, float) and str(cluster_id) == "nan") else str(cluster_id)
        route_id = None if route_id is None or (isinstance(route_id, float) and str(route_id) == "nan") else int(route_id)

        print(f"[{suf}] target={target} | cluster_id={cluster_id} | route_id={route_id}")

        # ---- SB-CGR (cluster_id) ----
        if target is not None and cluster_id is not None:
            try:
                sb = data[target]["clusters"][cluster_id]["sb_cgr"]
                sb.clean2d()
                try:
                    display(SVG(cgr_display(sb)))
                except Exception:
                    display(sb)
            except Exception as e:
                print(f"  (SB-CGR display failed for {target} cluster {cluster_id}: {e})")

        # ---- Pathway SVG for route_id ----
        if target is not None and route_id is not None:
            try:
                tree = data[target]["tree"]
                display(SVG(get_route_svg(tree, route_id)))
            except Exception as e:
                print(f"  (Pathway display failed for {target} route {route_id}: {e})")

        # ---- RouteCGR object (optional) ----
        if show_routecgr and target is not None and route_id is not None:
            try:
                all_route_cgrs = data[target].get(routecgr_dict_key, {})
                rcgr = all_route_cgrs.get(route_id, None)
                if rcgr is None:
                    print(f"  (RouteCGR not found in data[{target!r}][{routecgr_dict_key!r}] for route_id={route_id})")
                else:
                    rcgr.clean2d()
                    try:
                        display(SVG(cgr_display(rcgr)))
                    except Exception:
                        display(rcgr)
            except Exception as e:
                print(f"  (RouteCGR display failed for {target} route {route_id}: {e})")

        print("-" * 80)


def plot_histogram(histogram_data):

    bin_width = 0.1  # change to 0.02, 0.05, etc. if you want fewer/more bars
    bins = np.arange(histogram_data.mean(), histogram_data.max() + bin_width, bin_width)

    plt.figure(figsize=(8, 4))
    # plt.hist(histogram_data, bins=bins, edgecolor="black")  # frequency counts by default
    plt.hist(histogram_data, bins="auto", edgecolor="black")
    plt.xlabel("Tanimoto Sum")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Aggregated Cross-Target Similarity (RouteCGR; Tanimoto Sum)")
    plt.xlim(4.75, 6.25)
    plt.tight_layout()
    plt.show()