import numpy as np
import re
import pandas as pd
import string
from itertools import combinations
from typing import Optional, List
from synplan.chem.utils_copy import cgr_to_packed_fp, tanimoto_packed
from multitarget.utils import *
from multitarget.utils import tanimoto_upper_bound_from_ones


def _clusters_for_target(data: dict, target: str) -> dict:
    """Helper accessor. Adjust here if your data structure differs."""
    return data[target]["clusters"]


def build_sb_cluster_fp_table(
    data: dict,
    target: str,
    max_radius: int = 1,
    length: int = 4096,
) -> pd.DataFrame:
    """
    Build a per-target table of SB‑CGR fingerprints for all clusters.

    Returns columns:
      target, cluster_id, packed, ones, group_size, strat_bonds, route_ids
    """
    rows = []
    clusters = _clusters_for_target(data, target)
    for cluster_id, cl in clusters.items():
        packed, ones = cgr_to_packed_fp(cl["sb_cgr"], max_radius=max_radius, length=length)
        rows.append({
            "target": target,
            "cluster_id": str(cluster_id),
            "packed": packed,
            "ones": int(ones),
            "group_size": cl.get("group_size"),
            "strat_bonds": cl.get("strat_bonds"),
            "route_ids": cl.get("route_ids", []),
        })
    return pd.DataFrame(rows)


def build_multitarget_hits_from_sb_clusters(
    sb_fp_all: pd.DataFrame,
    min_sim: float = 0.8,
    targets: Optional[List[str]] = None,
    beam_width: int = 5000,
    top_k: Optional[int] = 2000,
    target_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Multi-target matches directly among SB-CGR clusters (no communities/medoids).

    sb_fp_all required columns:
      target, cluster_id, packed, ones

    Returns columns:
      target_A, cluster_id_A, target_B, cluster_id_B, ...,
      score (sum of all pairwise TI),
      min_tanimoto, mean_tanimoto
    """
    pairs_total = 0
    pairs_pass  = 0

    required = {"target", "cluster_id", "packed", "ones"}
    missing = required - set(sb_fp_all.columns)
    if missing:
        raise KeyError(f"sb_fp_all missing required columns: {sorted(missing)}")

    df = sb_fp_all.copy()

    # choose targets
    if targets is None:
        targets = sorted(df["target"].unique().tolist())
    else:
        targets = list(targets)

    if target_order is None:
        target_order = targets[:]
    else:
        target_order = [t for t in target_order if t in targets] + [t for t in targets if t not in target_order]

    # filter
    df = df[df["target"].isin(targets)].reset_index(drop=True)

    # node ids
    df["_node_id"] = np.arange(len(df), dtype=int)

    # per-target views
    by_target = {
        t: df.loc[df["target"] == t, ["_node_id", "packed", "ones", "cluster_id"]].copy()
        for t in targets
    }

    # adjacency + sim lookup across targets
    adj = {int(nid): {} for nid in df["_node_id"].values}   # node -> other_target -> set(node_ids)
    sim_lookup = {}  # (u,v) -> sim (store both directions)

    for tA, tB in combinations(targets, 2):
        A = by_target[tA].reset_index(drop=True)
        B = by_target[tB].reset_index(drop=True)

        for i in range(len(A)):
            u = int(A.loc[i, "_node_id"])
            pa, oa = A.loc[i, "packed"], int(A.loc[i, "ones"])

            for j in range(len(B)):
                v = int(B.loc[j, "_node_id"])
                pb, ob = B.loc[j, "packed"], int(B.loc[j, "ones"])
                pairs_total += 1
                # cheap prune
                if tanimoto_upper_bound_from_ones(oa, ob) < min_sim:
                    continue

                s = tanimoto_packed(pa, oa, pb, ob)
                if s >= min_sim:
                    adj[u].setdefault(tB, set()).add(v)
                    adj[v].setdefault(tA, set()).add(u)
                    sim_lookup[(u, v)] = s
                    sim_lookup[(v, u)] = s
                    pairs_pass += 1

    # expand smaller targets first
    expand_order = sorted(targets, key=lambda t: len(by_target[t]))

    # seed beam with first target nodes
    first_t = expand_order[0]
    partials = [({first_t: int(r["_node_id"])}, 0.0, [])
                for _, r in by_target[first_t].iterrows()]

    # beam expansion
    for t_new in expand_order[1:]:
        new_partials = []
        all_nodes_new = set(by_target[t_new]["_node_id"].astype(int).tolist())

        for sel, score, sims in partials:
            # candidates must connect to ALL previously chosen nodes
            cands = all_nodes_new.copy()
            for t_prev, u in sel.items():
                cands &= adj[u].get(t_new, set())
                if not cands:
                    break
            if not cands:
                continue

            for v in cands:
                add_sims = []
                add_score = 0.0
                ok = True

                for u in sel.values():
                    s_uv = sim_lookup.get((u, v))
                    if s_uv is None:
                        ok = False
                        break
                    add_sims.append(s_uv)
                    add_score += s_uv

                if not ok:
                    continue

                sel2 = dict(sel)
                sel2[t_new] = int(v)
                new_partials.append((sel2, score + add_score, sims + add_sims))

        if not new_partials:
            return pd.DataFrame()

        new_partials.sort(key=lambda x: x[1], reverse=True)
        partials = new_partials[:beam_width]

    # format output
    letters = list(string.ascii_uppercase)
    if len(target_order) > len(letters):
        raise ValueError("Too many targets for A/B/C suffixes (>26).")

    node_row = df.set_index("_node_id")[["target", "cluster_id"]].to_dict("index")

    rows = []
    for sel, score, sims in partials:
        out = {"score": float(score)}
        for k, t in enumerate(target_order):
            suf = letters[k]
            nid = sel.get(t)
            r = node_row[nid]
            out[f"target_{suf}"] = r["target"]
            out[f"cluster_id_{suf}"] = str(r["cluster_id"])

        out["min_tanimoto"] = float(min(sims)) if sims else np.nan
        out["mean_tanimoto"] = float(np.mean(sims)) if sims else np.nan
        rows.append(out)

    multi_hits = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if top_k is not None:
        multi_hits = multi_hits.head(int(top_k)).reset_index(drop=True)

    print("pairs_total:", pairs_total)
    print("pairs_pass :", pairs_pass)
    print("pairs_total + pairs_pass:", pairs_total + pairs_pass)

    return multi_hits


def _detect_suffixes_from_targets(df: pd.DataFrame):
    """Find suffixes X where target_X exists."""
    sfx = []
    for c in df.columns:
        m = re.match(r"^target_(.+)$", c)
        if m:
            sfx.append(m.group(1))
    return sorted(sfx)


def build_cluster_routes_table_from_multi_hits(
    multi_hits_sb: pd.DataFrame,
    sb_fp_all: pd.DataFrame,
    routes_col_sbfp: str = "route_ids",
) -> pd.DataFrame:
    """
    Build merged route_id lists for ONLY the SB-clusters referenced in multi_hits_sb.

    multi_hits_sb: output of build_multitarget_hits (SB level)
      expects columns like:
        target_A, (cluster_id_A or medoid_cluster_A), target_B, ...

    sb_fp_all: per-cluster SB fp table (your build_sb_cluster_fp_table concat)
      expects columns:
        target, cluster_id, route_ids

    Returns:
      target, cluster_id, merged_routes(list[int]), n_routes
    """
    # --- identify which column name is used for cluster ids in multi_hits_sb ---
    suffixes = _detect_suffixes_from_targets(multi_hits_sb)
    if not suffixes:
        raise KeyError("multi_hits_sb has no target_X columns (e.g., target_A).")

    # choose cluster column base name
    # Prefer cluster_id_X, fallback to medoid_cluster_X
    def _cluster_col_for_suffix(s):
        if f"cluster_id_{s}" in multi_hits_sb.columns:
            return f"cluster_id_{s}"
        return None

    cluster_cols = {s: _cluster_col_for_suffix(s) for s in suffixes}
    if all(v is None for v in cluster_cols.values()):
        raise KeyError("multi_hits_sb must contain cluster_id_X or medoid_cluster_X columns.")

    # --- collect all (target, cluster_id) pairs used anywhere in multi_hits_sb ---
    pairs = []
    for s in suffixes:
        ccol = cluster_cols[s]
        if ccol is None:
            continue
        tmp = multi_hits_sb[[f"target_{s}", ccol]].dropna()
        tmp = tmp.rename(columns={f"target_{s}": "target", ccol: "cluster_id"})
        tmp["cluster_id"] = tmp["cluster_id"].astype(str)
        pairs.append(tmp)

    used_pairs = pd.concat(pairs, ignore_index=True).drop_duplicates()

    # --- validate sb_fp_all schema ---
    req = {"target", "cluster_id", routes_col_sbfp}
    missing = req - set(sb_fp_all.columns)
    if missing:
        raise KeyError(f"sb_fp_all missing columns: {sorted(missing)}")

    sb = sb_fp_all.copy()
    sb["cluster_id"] = sb["cluster_id"].astype(str)

    # --- filter sb_fp_all to only used pairs and build merged routes table ---
    merged = used_pairs.merge(
        sb[["target", "cluster_id", routes_col_sbfp]],
        on=["target", "cluster_id"],
        how="left"
    )

    # aggregate (in case of duplicates)
    out = (
        merged.groupby(["target", "cluster_id"], dropna=False, as_index=False)
              .agg({routes_col_sbfp: lambda x: sorted({int(rid) for lst in x if isinstance(lst, (list, tuple, set))
                                                      for rid in lst})})
              .rename(columns={routes_col_sbfp: "route_ids"})
    )
    out["n_routes"] = out["route_ids"].apply(len)
    return out

def build_route_fp_df_from_cluster_routes(
    data: dict,
    cluster_routes: pd.DataFrame,
    max_radius: int = 1,
    length: int = 4096,
    route_col: str = "route_ids",
) -> pd.DataFrame:
    """
    Fingerprint RouteCGRs for routes listed per (target, cluster_id).

    Returns columns:
      target, cluster_id, route_id, packed, ones
    """
    req = {"target", "cluster_id", route_col}
    missing = req - set(cluster_routes.columns)
    if missing:
        raise KeyError(f"cluster_routes missing columns: {sorted(missing)}")

    rows = []
    for _, r in cluster_routes.iterrows():
        t = r["target"]
        cl = str(r["cluster_id"])
        route_ids = r[route_col] or []

        route_map = data[t].get("all_route_cgrs", {})
        for rid in route_ids:
            rid = int(rid)
            if rid not in route_map:
                continue
            route_cgr = route_map[rid]
            packed, ones = cgr_to_packed_fp(route_cgr, max_radius=max_radius, length=length)
            rows.append({
                "target": t,
                "cluster_id": cl,
                "route_id": rid,
                "packed": packed,
                "ones": int(ones),
            })

    return pd.DataFrame(rows)



# ---------- core: top tuples for ONE SB-hit row ----------
def top_route_tuples_for_row_cluster(
    row: pd.Series,
    suffixes: list,
    cluster_route_idx: dict,
    top_n: int = 50,
    beam_width: int = 400,
    min_pair_ti: float | None = 0.9,
    order_by_smallest_first: bool = True
):
    """
    One route per (target_X, cluster_id_X), ranked by sum of pairwise RouteCGR Tanimotos.
    Returns list of dicts with route_id_X + route_cgr_score.
    """

    # collect cluster route lists for this row
    blocks = []
    for suf in suffixes:
        t = norm_target(row[f"target_{suf}"])
        cl = norm_cluster_id(row[f"cluster_id_{suf}"])
        routes = cluster_route_idx.get((t, cl), [])
        blocks.append({"suf": suf, "target": t, "cluster_id": cl, "routes": routes})

    # if any cluster has no route fingerprints -> nothing to do
    if any(len(b["routes"]) == 0 for b in blocks):
        return []

    # process smallest route-sets first (reduces branching)
    blocks_proc = sorted(blocks, key=lambda b: len(b["routes"])) if order_by_smallest_first else blocks

    # beam elements: (chosen_list, score)
    # chosen_list: list of (suf, rid, packed, ones)
    beam = [([], 0.0)]

    for blk in blocks_proc:
        new_beam = []
        routes_j = blk["routes"]
        suf_j = blk["suf"]

        for chosen, score in beam:
            for rid_j, p_j, o_j in routes_j:
                inc = 0.0
                ok = True

                # check against already chosen routes
                for (_, _, p_i, o_i) in chosen:
                    if min_pair_ti is not None and tanimoto_upper_bound_from_ones(o_i, o_j) < min_pair_ti:
                        ok = False
                        break

                    sim = tanimoto_packed(p_i, o_i, p_j, o_j)

                    if min_pair_ti is not None and sim < min_pair_ti:
                        ok = False
                        break

                    inc += sim

                if not ok:
                    continue

                new_chosen = chosen + [(suf_j, rid_j, p_j, o_j)]
                new_beam.append((new_chosen, score + inc))

        if not new_beam:
            return []

        # keep best partials
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # convert beam to output dicts in original suffix order
    results = []
    for chosen, score in beam[:top_n]:
        chosen_map = {suf: rid for (suf, rid, _, _) in chosen}
        out = {"route_cgr_score": float(score)}
        for suf in suffixes:
            out[f"route_id_{suf}"] = chosen_map.get(suf, None)
        results.append(out)

    results.sort(key=lambda d: d["route_cgr_score"], reverse=True)
    return results



# ---------- apply to ALL SB-hit rows ----------
def expand_sb_hits_to_top_route_tuples(
    multi_hits_sb: pd.DataFrame,
    route_fp_df: pd.DataFrame,
    top_n_per_row: int = 50,
    beam_width: int = 400,
    min_pair_ti: float | None = 0.9,
    debug_missing: bool = True,
):
    """
    multi_hits_sb: SB-level hits (target_X + cluster_id_X columns).
    route_fp_df: RouteCGR fingerprints (target, cluster_id, route_id, packed, ones).

    Returns dataframe:
      all original multi_hits_sb cols + route_id_X cols + route_cgr_score
    """
    suffixes = detect_suffixes_cluster(multi_hits_sb)
    if not suffixes:
        raise KeyError("No target_X/cluster_id_X pairs found in multi_hits_sb.")

    cluster_route_idx = build_cluster_route_fp_index(route_fp_df)

    if debug_missing:
        miss_rows = 0
        for _, row in multi_hits_sb.iterrows():
            ok = True
            for suf in suffixes:
                t = norm_target(row[f"target_{suf}"])
                cl = norm_cluster_id(row[f"cluster_id_{suf}"])
                if (t, cl) not in cluster_route_idx:
                    ok = False
                    break
            if not ok:
                miss_rows += 1
        print(f"[debug] SB-hit rows with at least one missing (target,cluster_id) in route_fp_df: {miss_rows} / {len(multi_hits_sb)}")

    out_rows = []
    for idx, row in multi_hits_sb.iterrows():
        tuples = top_route_tuples_for_row_cluster(
            row=row,
            suffixes=suffixes,
            cluster_route_idx=cluster_route_idx,
            top_n=top_n_per_row,
            beam_width=beam_width,
            min_pair_ti=min_pair_ti,
            order_by_smallest_first=True
        )
        for t in tuples:
            r = row.to_dict()
            r.update(t)
            r["_match_row"] = idx
            out_rows.append(r)

    if not out_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(out_rows)
          .sort_values(["_match_row", "route_cgr_score"], ascending=[True, False])
          .reset_index(drop=True)
    )


def build_final_df_grouped_sb(
    tuples_df: pd.DataFrame,
    score_col: str = "route_cgr_score",
    count_col: str = "n_route_tuples",
) -> pd.DataFrame:
    """
    SB-only version.

    Groups route tuples by stable identity columns:
      target_X, cluster_id_X   (for all suffixes X)

    Aggregates:
      - route_id_X -> route_id_X_list (unique sorted ints)
      - score_col  -> max score for that identity group
      - count_col  -> product of list lengths across suffixes

    Special rule:
      - If the set of cluster_id_X across suffixes is identical, they will naturally
        group because cluster_id_X columns are part of the identity.
        (This is exactly what grouping by target_X+cluster_id_X achieves.)

    Returns final_df with score_col as the first column.
    """
    df = tuples_df.copy()

    suffixes = detect_suffixes_cluster(df)
    if not suffixes:
        raise KeyError("No target_X/cluster_id_X pairs found in tuples_df.")

    # identity columns = target_X + cluster_id_X
    id_cols = []
    route_cols = []
    for s in suffixes:
        tcol = f"target_{s}"
        ccol = f"cluster_id_{s}"
        if tcol in df.columns: id_cols.append(tcol)
        if ccol in df.columns: id_cols.append(ccol)

        rcol = f"route_id_{s}"
        if rcol in df.columns:
            route_cols.append(rcol)

    if not id_cols:
        raise KeyError("Missing identity columns (target_X / cluster_id_X).")
    if not route_cols:
        raise KeyError("No route_id_X columns found.")

    def unique_sorted_int_list(x):
        return sorted({int(v) for v in x if pd.notna(v)})

    # aggregate per group
    agg = {c: unique_sorted_int_list for c in route_cols}
    if score_col in df.columns:
        agg[score_col] = "max"

    g = df.groupby(id_cols, dropna=False).agg(agg).reset_index()

    # rename route_id_X -> route_id_X_list
    rename = {c: f"{c}_list" for c in route_cols}
    g = g.rename(columns=rename)

    # product of route-list sizes
    def product_len(row):
        prod = 1
        for c in route_cols:
            lst = row.get(f"{c}_list", [])
            prod *= len(lst) if isinstance(lst, list) else 0
        return int(prod)

    g[count_col] = g.apply(product_len, axis=1)

    # column order: score first, then identity, then route lists + count
    front = [score_col] if score_col in g.columns else []
    front += id_cols

    rest = [c for c in g.columns if c not in front]
    final_df = (
        g[front + rest]
        .sort_values(by=[score_col] if score_col in g.columns else [count_col], ascending=False)
        .reset_index(drop=True)
    )

    return final_df


def best_tuple_per_group_sb(tuples_df_unique: pd.DataFrame, score_col: str = "route_cgr_score"):
    """
    SB-only version.

    Keeps only the best row (highest score_col) per stable identity group:
      target_X, cluster_id_X   (for all suffixes X)

    Also moves score_col to the first column.
    """
    df = tuples_df_unique.copy()

    suffixes = detect_suffixes_cluster(df)
    if not suffixes:
        raise KeyError("No target_X/cluster_id_X pairs found in df.")

    id_cols = []
    for s in suffixes:
        tcol = f"target_{s}"
        ccol = f"cluster_id_{s}"
        if tcol in df.columns: id_cols.append(tcol)
        if ccol in df.columns: id_cols.append(ccol)

    if not id_cols:
        raise KeyError("No identity columns (target_X/cluster_id_X) found.")

    # keep best per group
    if score_col in df.columns:
        df = df.sort_values(score_col, ascending=False)
    df = df.drop_duplicates(subset=id_cols, keep="first").reset_index(drop=True)

    # move score_col to first position
    if score_col in df.columns:
        cols = [score_col] + [c for c in df.columns if c != score_col]
        df = df[cols]

    return df