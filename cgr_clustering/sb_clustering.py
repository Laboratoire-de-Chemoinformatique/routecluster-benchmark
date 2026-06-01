from typing import Dict, Tuple, Any
import copy

from chython import smiles
from chython.containers import CGRContainer, ReactionContainer
from chython.containers.bonds import DynamicBond
from chython.periodictable import DynamicN
from synplan.chem.reaction_routes.clustering import extract_strat_bonds
from synplan.chem.reaction_routes.route_cgr import compose_sb_cgr as _compose_sb_cgr


def _sync_dynamic_atom_state(cgr):
    # SynPlan updates CGR dictionaries during reduction. Chython's copied
    # DynamicElement objects must be updated as well or stale charges remain in SMILES.
    for atom_num, atom in cgr._atoms.items():
        atom._charge = cgr._charges[atom_num]
        atom._p_charge = cgr._p_charges[atom_num]
        atom._is_radical = cgr._radicals[atom_num]
        atom._p_is_radical = cgr._p_radicals[atom_num]
    cgr.flush_cache()
    return cgr


def compose_sb_cgr(route_cgr):
    """Reduce a RouteCGR and reconcile Chython dynamic-atom state."""
    return _sync_dynamic_atom_state(_compose_sb_cgr(route_cgr))


def compose_all_sb_cgrs(route_cgrs_dict):
    """Reduce every RouteCGR with Chython dynamic-atom reconciliation."""
    return {
        route_id: compose_sb_cgr(route_cgr)
        for route_id, route_cgr in route_cgrs_dict.items()
    }

def merge_groups(data, key1, key2):
    """
    Merge the group at key2 into the group at key1 within the provided dictionary.
    
    Parameters:
    - data (dict): Original dictionary containing the groups.
    - key1 (str): The key of the target group to preserve.
    - key2 (str): The key of the source group whose data will be merged, then deleted.
    
    Returns:
    - dict: A new dictionary with the merged result.
    """
    # Work on a shallow copy of the main dict to avoid mutating original
    merged = copy.deepcopy(data)
    
    # If key2 is not present or is empty, return the copy unchanged
    if not key2 or key2 not in merged:
        return merged
    
    group1 = merged[key1]
    group2 = merged[key2]
    
    # Merge node_ids
    group1['route_ids'].extend(group2['route_ids'])
    
    # Update group_size
    group1['group_size'] += group2['group_size']
    
    # Remove the old group
    del merged[key2]

    return merged

def fix_dict_key_order(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renumber keys "M.N" so that for each M the Ns run 1,2,3… in the order seen.
    Prints any old→new key mappings.
    Returns a new dict with values preserved.
    """
    next_minor: Dict[str, int] = {}
    new_dict: Dict[str, Any] = {}
    changes: list[Tuple[str, str]] = []

    for old_key, value in d.items():
        # split major and ignore original minor
        major, _ = old_key.split('.', 1)
        # init counter if first time seeing this major
        if major not in next_minor:
            next_minor[major] = 1

        new_key = f"{major}.{next_minor[major]}"
        next_minor[major] += 1

        if new_key != old_key:
            changes.append((old_key, new_key))

        new_dict[new_key] = value

    # report what changed
    if changes:
        print("Renamed keys:")
        for old, new in changes:
            print(f"  {old} → {new}")

    return new_dict

def remap_final_cgrs(sbp_groups, t_smiles):
    new_sbp_groups = {}
    for i, group in sbp_groups.items():
        sb_cgr = group['sb_cgr']
        rr = sb_cgr.decompose()
        r_react = rr[0]
        r_prod = rr[1]
        cgr_mol = smiles(t_smiles)
        if str(r_prod) == str(cgr_mol):
            mapping = r_prod.get_mapping(cgr_mol)
            nn = next(mapping)
            r_react.remap(nn)
            new_react = ReactionContainer(r_react.split() , [cgr_mol])
            new_cgr = new_react.compose()
            str_b = extract_strat_bonds(new_cgr)    
            group['sb_cgr'] = new_cgr
            group['strat_bonds'] = str_b
            new_sbp_groups[i] = group
    return new_sbp_groups

def _aromatic_amine_pattern():
    pattern = CGRContainer()
    carbon_1 = pattern.add_atom('C')
    carbon_main = pattern.add_atom('C')
    carbon_2 = pattern.add_atom('C')
    nitrogen = pattern.add_atom('N')
    pattern.add_bond(carbon_1, carbon_main, DynamicBond(None, 4))
    pattern.add_bond(carbon_main, carbon_2, DynamicBond(None, 4))
    pattern.add_bond(carbon_main, nitrogen, DynamicBond(1, 1))
    return pattern


def map_error_fix(sbp_groups, verbose=False):
    # To do: other way to identify mapping errors? This one works only for apatinib
    arom_amine = _aromatic_amine_pattern()
    error_clusters = []
    clean_sbp_groups = {}
    for i, group in sbp_groups.items():
        # print(i)
        sb_cgr = group['sb_cgr']
        if sb_cgr > arom_amine:
            cgr_query = sb_cgr.substructure(sb_cgr._atoms)
            n = next(arom_amine.get_mapping(cgr_query))
            error_clusters.append(i)
            cgr = sb_cgr
            fff = list(n.values())
            for atom_num in fff:
                if isinstance(cgr_query._atoms[atom_num], DynamicN):
                    num_n = atom_num
                    data = cgr_query._bonds[num_n]
                    fff.remove(num_n)
                    for a_num in fff:
                        if a_num in data.keys():
                            if data[a_num].order == 1 and data[a_num].p_order == 1:
                                num_c_main = a_num
                                fff.remove(a_num)
            num_c_1 = fff[0]
            num_c_2 = fff[1]
            # --- Fix bond (C–C aromatic) ---
            b27 = cgr.bond(num_c_1, num_c_main)
            if b27 is not None and b27.order is None and b27.p_order == 4:
                cgr.delete_bond(num_c_1,num_c_main)
                cgr.add_bond(num_c_1, num_c_main, DynamicBond(4,4))

            # --- Fix bond (C–C aromatic) ---
            b37 = cgr.bond(num_c_2, num_c_main)
            if b37 is not None and b37.order is None and b37.p_order == 4:
                cgr.delete_bond(num_c_2,num_c_main)
                cgr.add_bond(num_c_2,num_c_main, DynamicBond(4,4))

            # --- Fix bond (C–N) ---
            b78 = cgr.bond(num_c_main, num_n)
            if b78 is not None and b78.order == 1 and b78.p_order == 1:
                cgr.delete_bond(num_c_main,num_n)
                cgr.add_bond(num_c_main,num_n, DynamicBond(None,1))
            group['sb_cgr'] = cgr 
            group['strat_bonds'] = extract_strat_bonds(cgr) 
            clean_sbp_groups[i] = group
        else:
            clean_sbp_groups[i] = group
    if len(error_clusters) > 0 and verbose:
        print("Fixed error clusters:", error_clusters)
    return clean_sbp_groups


def _key_sort_value(k: str):
    """
    Turn '2.3' or '10.1' into (2, 3) or (10, 1) so we can compare keys numerically.
    If there's no dot, treat the entire string as the first number.
    """
    parts = k.split(".")
    first = int(parts[0])
    second = int(parts[1]) if len(parts) > 1 else 0
    return (first, second)

def merge_identical_sb_cgr(clusters: dict, verbose=False) -> dict:
    """
    Detect clusters that have identical 'sb_cgr' and merge them using merge_groups.
    'cluster_key_min' is the key with the lowest first number (before the dot),
    'cluster_key_max' is the corresponding highest one.
    """
    # work on a copy of the whole structure
    merged_clusters = copy.deepcopy(clusters)

    keys = list(merged_clusters.keys())
    removed_keys = set()

    # pairwise comparison of sb_cgr objects
    for i, ki in enumerate(keys):
        if ki in removed_keys or ki not in merged_clusters:
            continue

        cgr_i = merged_clusters[ki]['sb_cgr']

        for kj in keys[i + 1:]:
            if kj in removed_keys or kj not in merged_clusters:
                continue

            cgr_j = merged_clusters[kj]['sb_cgr']

            # "identical sb_cgr" – treat as equal if they compare equal
            # or are literally the same object
            try:
                identical = (cgr_i == cgr_j)
                
            except Exception:
                identical = (cgr_i is cgr_j)
            if identical and verbose:
                print(ki, kj, '- merged')

            if not identical:
                continue

            # decide which key is min / max based on the numeric part before the dot
            if _key_sort_value(ki) <= _key_sort_value(kj):
                cluster_key_min, cluster_key_max = ki, kj
            else:
                cluster_key_min, cluster_key_max = kj, ki

            # apply merge (key_max into key_min)
            merged_clusters = merge_groups(merged_clusters, cluster_key_min, cluster_key_max)

            # remember that key_max is gone
            removed_keys.add(cluster_key_max)

    return merged_clusters

def _parse_cluster_key(key: str):
    """
    Turn '2.3' or '10.1' into (2, 3) or (10, 1) so we can sort keys numerically.
    If there's no dot, treat the entire string as the first number, second = 0.
    """
    parts = key.split(".")
    first = int(parts[0])
    second = int(parts[1]) if len(parts) > 1 else 0
    return first, second


def renumber_clusters_by_strat_bonds(clusters: dict, verbose=False):
    """
    Rebuild the cluster keys so that:

      first_part == len(cluster['strat_bonds'])
      second_part is a unique 1-based index within that first_part group

    Example:
        cluster '5.4' with len(strat_bonds) == 4 becomes '4.x'
        where x is the next available integer among 4.* clusters.

    Returns:
        new_clusters: dict with new keys
        key_mapping: dict {old_key: new_key}
    """
    # group clusters by len(strat_bonds),
    # iterating in sorted order of current cluster keys
    buckets = {}  # {len_strat_bonds: [(old_key, cluster_dict), ...]}
    for old_key in sorted(clusters.keys(), key=_parse_cluster_key):
        group = clusters[old_key]
        l = len(group['strat_bonds'])
        buckets.setdefault(l, []).append((old_key, group))
    
    new_clusters = {}
    key_mapping = {}

    # Now assign new keys like "2.1", "2.2", ..., "3.1", ...
    for l in sorted(buckets.keys()):
        counter = 1
        for old_key, group in buckets[l]:
            new_key = f"{l}.{counter}"
            new_clusters[new_key] = copy.deepcopy(group)
            key_mapping[old_key] = new_key
            counter += 1

    # inspect which were misleading (old != new)
    misleading = {old: new for old, new in key_mapping.items() if old != new}
    if verbose:
        print("Old → New cluster keys (only changed ones):")
        for old, new in sorted(misleading.items(), key=lambda kv: _parse_cluster_key(kv[0])):
            print(f"{old}  ->  {new}")

    return new_clusters
