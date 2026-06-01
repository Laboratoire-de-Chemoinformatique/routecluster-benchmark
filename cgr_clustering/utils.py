import pandas as pd
import numpy as np

from chython.containers import ReactionContainer

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Mapping


def collect_overlaps(syn, ask, aiz):
    # Compute overlaps
    only_syn = syn - ask - aiz
    only_ask = ask - syn - aiz
    only_ai = aiz - syn - ask

    syn_ask = (syn & ask) - aiz
    syn_ai = (syn & aiz) - ask
    ask_ai = (ask & aiz) - syn

    all_three = syn & ask & aiz

    # Organize into a DataFrame
    regions = {
        'Only SynPlanner': only_syn,
        'Only ASKCOS': only_ask,
        'Only AiZynthFinder': only_ai,
        'SynPlanner ∩ ASKCOS': syn_ask,
        'SynPlanner ∩ AiZynthFinder': syn_ai,
        'ASKCOS ∩ AiZynthFinder': ask_ai,
        'All Three': all_three
    }

    df = pd.DataFrame({
        'Region': list(regions.keys()),
        'Count': [len(v) for v in regions.values()],
        'Examples': [list(v) for v in regions.values()]
    })
    return df


def build_sb_to_keys(clusters: Mapping[Any, Mapping[str, Any]], field: str = "sb_cgr") -> Dict[str, List[Any]]:
    sb_to_keys = defaultdict(list)
    for k, cluster in clusters.items():
        sb_to_keys[str(cluster[field])].append(k)
    return dict(sb_to_keys)


def merge_keys_for_sbs(examples, keys_for_sb):
    merged = {"synplanner_keys": [], "askcos_keys": [], "aizynth_keys": []}

    for sb_smi in examples:
        d = keys_for_sb(sb_smi)
        merged["synplanner_keys"].extend(d.get("synplanner_keys", []))
        merged["askcos_keys"].extend(d.get("askcos_keys", []))
        merged["aizynth_keys"].extend(d.get("aizynth_keys", []))

    return merged


def make_keys_for_sb(
    synplanner_clusters: Mapping[Any, Mapping[str, Any]],
    askcos_clusters: Mapping[Any, Mapping[str, Any]],
    aizynth_clusters: Mapping[Any, Mapping[str, Any]],
    field: str = "sb_cgr",
):
    syn = build_sb_to_keys(synplanner_clusters, field=field)
    ask = build_sb_to_keys(askcos_clusters, field=field)
    ai  = build_sb_to_keys(aizynth_clusters, field=field)

    def keys_for_sb(sb_cgr_str: str):
        return {
            "synplanner_keys": syn.get(sb_cgr_str, []),
            "askcos_keys": ask.get(sb_cgr_str, []),
            "aizynth_keys": ai.get(sb_cgr_str, []),
        }

    return keys_for_sb


def compute_group_percentages(clusters):
    """
    Given a dict mapping cluster IDs (e.g. '2.1') to cluster data containing 'group_size',
    returns a dict mapping the integer prefix (e.g. 2) to the percentage of the total group_size.
    """
    # Sum sizes by prefix
    prefix_sums = {}
    for cid, data in clusters.items():
        prefix = int(cid.split('.')[0])
        size = data.get('group_size', 0)
        prefix_sums[prefix] = prefix_sums.get(prefix, 0) + size

    # Compute total
    total = sum(prefix_sums.values())
    if total == 0:
        return {prefix: 0.0 for prefix in prefix_sums}

    # Calculate percentages
    return {
        prefix: round((size / total) * 100, 2)
        for prefix, size in prefix_sums.items()
    }


def tanimoto(a, b):
    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))
    return intersection / union if union > 0 else 0.0


def diff_score_sb_cgr(cgr_1, cgr_2):

    # sb_cgr_1.clean2d()
    # sb_cgr_2.clean2d()
    # display(sb_cgr_1)
    # display(sb_cgr_2)

    mol_1_react, mol_1_prod = cgr_1.decompose()
    chython_reaction_1 = ReactionContainer([mol_1_react], [mol_1_prod])
    cgr_chython_1 = chython_reaction_1.compose()
    
    mol_2_react, mol_2_prod = cgr_2.decompose()
    chython_reaction_2 = ReactionContainer([mol_2_react], [mol_2_prod])
    cgr_chython_2 = chython_reaction_2.compose()

    # sb_cgr_fp_1 = cgr_chython_1.linear_fingerprint(max_radius=2, length=4096)
    # sb_cgr_fp_2 = cgr_chython_2.linear_fingerprint(max_radius=2, length=4096)

    sb_cgr_fp_1 = cgr_chython_1.morgan_fingerprint(max_radius=2, length=4096)
    sb_cgr_fp_2 = cgr_chython_2.morgan_fingerprint(max_radius=2, length=4096)

    sim12 = tanimoto(sb_cgr_fp_1, sb_cgr_fp_2)
    return sim12


# ---------- similarity ----------
def tanimoto_sim(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return (inter / union) if union else 0.0


def cgr_to_morgan_fp(cgr, max_radius=2, length=4096) -> np.ndarray:
    # decompose() -> (reactants, products)
    react_cgr, prod_cgr = cgr.decompose()

    rxn = ReactionContainer([react_cgr], [prod_cgr])
    cgr_chy = rxn.compose()

    fp = cgr_chy.morgan_fingerprint(max_radius=max_radius, length=length)
    return np.asarray(fp, dtype=bool)


def distance_matrix(cgrs_dict, radius=2, length=4096) -> pd.DataFrame:
    ids = list(range(len(cgrs_dict)))

    # 1) Precompute fingerprints once (much faster than recomputing per pair)
    fps = {}
    for k in ids:
        fps[k] = cgr_to_morgan_fp(cgrs_dict[k], max_radius=radius, length=length)

    # 2) Pairwise similarity (upper triangle) -> symmetric matrix
    n = len(ids)
    S = np.eye(n, dtype=np.float32)  # similarity matrix
    idx = {k: i for i, k in enumerate(ids)}

    for a_id, b_id in combinations(ids, 2):
        i, j = idx[a_id], idx[b_id]
        sim = tanimoto_sim(fps[a_id], fps[b_id])
        S[i, j] = sim
        S[j, i] = sim

    # 3) Convert to distance matrix (common choice: distance = 1 - similarity)
    D = (1.0 - S).astype(np.float32)
    np.fill_diagonal(D, 0.0)

    # Optional: nice labeled DataFrames
    D_df = pd.DataFrame(D, index=ids, columns=ids)

    return D_df
