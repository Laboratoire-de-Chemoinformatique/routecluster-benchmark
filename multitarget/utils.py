import pandas as pd
from contextlib import contextmanager
import time
import re
import pickle
import numpy as np


def summarize_targets(data: dict) -> pd.DataFrame:
    """
    Return a compact summary with one row per target.

    Columns:
      - target
      - n_routes   : number of winning routes in `tree.winning_nodes` (if available)
      - n_clusters : number of SB-CGR clusters in `data[target]['clusters']`
    """
    rows = []
    for target, info in data.items():
        tree = info.get("tree", None)
        clusters = info.get("clusters", {})
        n_routes = None
        if tree is not None and hasattr(tree, "winning_nodes"):
            n_routes = len(tree.winning_nodes)
        rows.append({
            "target": target,
            "n_routes": n_routes,
            "n_clusters": len(clusters),
        })
    return pd.DataFrame(rows).sort_values(["n_routes", "n_clusters"], ascending=False)


def _pickle_profile_row(label: str, obj, protocol: int):
    try:
        t0 = time.perf_counter()
        blob = pickle.dumps(obj, protocol=protocol)
        dt = time.perf_counter() - t0
        size = len(blob)
        return {
            "label": label,
            "type": type(obj).__name__,
            "seconds": dt,
            "bytes": size,
            "mb": size / (1024 * 1024),
            "error": None,
        }
    except Exception as exc:
        return {
            "label": label,
            "type": type(obj).__name__,
            "seconds": None,
            "bytes": None,
            "mb": None,
            "error": repr(exc),
        }


def profile_pickle_sections(
    data: dict,
    max_samples: int = 3,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    inspect_tree_attrs: bool = False,
    max_tree_attrs: int = 25,
) -> pd.DataFrame:
    """
    Profile pickle time/size for top-level sections of `data`.

    - max_samples: how many sample items to inspect for clusters/lists
    - inspect_tree_attrs: if True, also pickle per-attribute samples from tree.__dict__
    """
    rows = []
    for target, info in data.items():
        rows.append(_pickle_profile_row(f"{target}:info", info, protocol))
        if isinstance(info, dict):
            for key, val in info.items():
                rows.append(_pickle_profile_row(f"{target}:{key}", val, protocol))

                if key == "clusters" and isinstance(val, dict):
                    for idx, (cid, cluster) in enumerate(val.items()):
                        if idx >= max_samples:
                            break
                        rows.append(_pickle_profile_row(f"{target}:clusters[{cid}]", cluster, protocol))

                if key in ("all_route_cgrs", "all_sb_cgrs") and isinstance(val, list):
                    for idx, item in enumerate(val[:max_samples]):
                        rows.append(_pickle_profile_row(f"{target}:{key}[{idx}]", item, protocol))

                if inspect_tree_attrs and key == "tree":
                    try:
                        items = list(vars(val).items())
                    except TypeError:
                        items = []
                    for idx, (attr, aval) in enumerate(items):
                        if idx >= max_tree_attrs:
                            break
                        rows.append(_pickle_profile_row(f"{target}:tree.{attr}", aval, protocol))

    df = pd.DataFrame(rows)
    if not df.empty and "seconds" in df.columns:
        df = df.sort_values("seconds", ascending=False, na_position="last").reset_index(drop=True)
    return df


def tanimoto_upper_bound_from_ones(oa: int, ob: int) -> float:
    """Maximum possible TI given only bitcounts (subset upper bound)."""
    return (min(oa, ob) / max(oa, ob)) if max(oa, ob) else 0.0


def detect_suffixes_cluster(matches_df: pd.DataFrame):
    """Detect suffixes X where target_X and cluster_id_X exist."""
    sfx = []
    for col in matches_df.columns:
        m = re.match(r"target_(.+)$", col)
        if m:
            suf = m.group(1)
            if f"cluster_id_{suf}" in matches_df.columns:
                sfx.append(suf)
    return sorted(sfx)


# ---------- popcount + packed tanimoto ----------
_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def popcount_uint8(u8arr: np.ndarray) -> int:
    return int(_POPCNT[u8arr].sum())


# ---------- normalization ----------
def norm_target(x) -> str:
    return str(x).strip()


def norm_cluster_id(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return str(x).strip()


# ---------- indexing ----------
def build_cluster_route_fp_index(route_fp_df: pd.DataFrame):
    """
    route_fp_df must have: target, cluster_id, route_id, packed, ones
    Index key is (norm_target, norm_cluster_id)
    """
    req = {"target", "cluster_id", "route_id", "packed", "ones"}
    missing = req - set(route_fp_df.columns)
    if missing:
        raise KeyError(f"route_fp_df missing columns: {sorted(missing)}")

    idx = {}
    for _, r in route_fp_df.iterrows():
        t = norm_target(r["target"])
        cl = norm_cluster_id(r["cluster_id"])
        rid = int(r["route_id"])
        packed = r["packed"]
        ones = int(r["ones"])
        idx.setdefault((t, cl), []).append((rid, packed, ones))
    return idx
