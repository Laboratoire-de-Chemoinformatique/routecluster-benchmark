from chython import smiles as smiles_chython
from chython.containers import ReactionContainer as ReactionContainerChython

from CGRtools import smiles as smiles_cgrtools
from CGRtools.containers import ReactionContainer, CGRContainer
from CGRtools.containers.bonds import DynamicBond
from CGRtools.algorithms.depict import *



def route_smi_2_cgr(pathway, reverse=False): # True for AiZynthFInder, False for ASKCOS
    """Converts a pathway of SMILES strings to a list of CGRs."""
    cgr_pathway = []
    inversed_pathway = pathway[::-1] if reverse else pathway
    for reaction_str in inversed_pathway:
        reactants = []
        product = smiles_chython(reaction_str[1])
        for reactant_smiles in reaction_str[0]:
            reactant = smiles_chython(reactant_smiles)
            reactant.kekule()
            reactant.implicify_hydrogens()
            reactant.thiele()
            reactants.append(reactant)
        reaction = ReactionContainerChython(reactants=reactants, products = [product])
        reaction.reset_mapping(keep_reactants_numbering=False)
        reaction_cgrtools = smiles_cgrtools(format(reaction, "m"))
        cgr_pathway.append(reaction_cgrtools)
    return cgr_pathway

def find_remap(lst):
    """
    Given a sorted list `lst` whose true length N is known to be len(lst),
    returns a dict mapping each value > N in lst to the missing values in 1..N.

    Example:
      L = [1,2,...,18,20,21,22,23]  # len=22
      => missing = [19]
         out_of_range = [23]
      => {23: 19}
    """
    N = len(lst)
    # 1) which values in the “ideal” 1..N are missing?
    missing = sorted(set(range(1, N+1)) - set(x for x in lst if x <= N))
    # 2) which values in lst have “overflowed” past N?
    out_of_range = sorted(x for x in lst if x > N)

    if len(missing) != len(out_of_range):
        raise ValueError(f"got {len(missing)} missing slots but {len(out_of_range)} overflow values")

    # 3) pair them up in ascending order
    return dict(zip(out_of_range, missing))

def _atom_symbol(atom):
    for attr in ("symbol", "atomic_symbol", "element"):
        val = getattr(atom, attr, None)
        if not val:
            continue
        if isinstance(val, str):
            return val
        sym = getattr(val, "symbol", None)
        if isinstance(sym, str):
            return sym
    name = atom.__class__.__name__
    if name.startswith("Dynamic"):
        name = name[len("Dynamic"):]
    return name

def _atom_hcount(atom):
    for attr in ("implicit_hydrogens", "hydrogens", "h", "hydrogen_count"):
        val = getattr(atom, attr, None)
        if val is None:
            continue
        if callable(val):
            try:
                val = val()
            except TypeError:
                continue
        if isinstance(val, (int, float)):
            return int(val)
    return 0

def _bond_order(bond):
    for attr in ("order", "p_order"):
        val = getattr(bond, attr, None)
        if isinstance(val, int):
            return val
    return None

def _is_nitrogen(atom):
    num = getattr(atom, "atomic_number", None)
    if num is None:
        num = getattr(atom, "number", None)
    if num == 7:
        return True
    return _atom_symbol(atom) == "N"

def _is_oxygen(atom):
    num = getattr(atom, "atomic_number", None)
    if num is None:
        num = getattr(atom, "number", None)
    if num == 8:
        return True
    return _atom_symbol(atom) == "O"

def _is_carbonyl_carbon(mol, atom_num):
    atom = mol._atoms.get(atom_num)
    if not atom or _atom_symbol(atom) != "C":
        return False
    bonds = getattr(mol, "_bonds", {})
    for nbr, bond in bonds.get(atom_num, {}).items():
        if _bond_order(bond) != 2:
            continue
        nbr_atom = mol._atoms.get(nbr)
        if nbr_atom and _is_oxygen(nbr_atom):
            return True
    return False

def _amide_by_carbonyl(mol):
    bonds = getattr(mol, "_bonds", {})
    mapping = {}
    for c_num, atom in mol._atoms.items():
        if _atom_symbol(atom) != "C":
            continue
        if not _is_carbonyl_carbon(mol, c_num):
            continue
        for nbr, bond in bonds.get(c_num, {}).items():
            if _bond_order(bond) != 1:
                continue
            nbr_atom = mol._atoms.get(nbr)
            if nbr_atom and _is_nitrogen(nbr_atom):
                mapping[c_num] = nbr
                break
    return mapping

def _select_n_attached_carbon(mol, n_num):
    bonds = getattr(mol, "_bonds", {})
    candidates = []
    for nbr, bond in bonds.get(n_num, {}).items():
        nbr_atom = mol._atoms.get(nbr)
        if not nbr_atom or _atom_symbol(nbr_atom) != "C":
            continue
        if _is_carbonyl_carbon(mol, nbr):
            continue
        order = _bond_order(bond)
        if order not in (1, 4):
            continue
        candidates.append((order, nbr))
    if not candidates:
        return None
    aromatic = [nbr for order, nbr in candidates if order == 4]
    if aromatic:
        return min(aromatic)
    return min(nbr for _, nbr in candidates)

def _expand_mapping_with_substructure(prod_mol, cand_mol, mapping, prod_n=None, cand_n=None):
    try:
        mapping_iter = cand_mol.get_mapping(prod_mol)
    except Exception:
        return mapping, 0

    chosen = None
    for cand_to_prod in mapping_iter:
        if cand_n is not None and prod_n is not None:
            if cand_to_prod.get(cand_n) != prod_n:
                continue
        chosen = cand_to_prod
        break

    if chosen is None:
        return mapping, 0

    prod_to_cand = {p: c for c, p in chosen.items()}
    added = 0
    for p, c in prod_to_cand.items():
        if p not in prod_mol._atoms:
            continue
        if p not in mapping:
            added += 1
        mapping[p] = c
    return mapping, added

def _has_dynamic_bond_4_0(cgr):
    for m_bond in cgr._bonds.values():
        for bond in m_bond.values():
            if isinstance(bond, DynamicBond) and bond.order == 4 and bond.p_order in (None, 0):
                return True
    return False

def _amide_nitrogens(mol):
    bonds = getattr(mol, "_bonds", {})
    amide_nums = []
    for num, atom in mol._atoms.items():
        if not _is_nitrogen(atom):
            continue
        for nbr, bond in bonds.get(num, {}).items():
            if _bond_order(bond) != 1:
                continue
            if _is_carbonyl_carbon(mol, nbr):
                amide_nums.append(num)
                break
    return amide_nums

def _amine_nitrogens(mol, amide_nums):
    amine_nums = {}
    for num, atom in mol._atoms.items():
        if not _is_nitrogen(atom) or num in amide_nums:
            continue
        h_count = _atom_hcount(atom)
        if h_count <= 0:
            continue
        amine_nums[num] = h_count
    return amine_nums

def _prep_molecules(molecules):
    for mol in molecules:
        mol.kekule()
        mol.implicify_hydrogens()
        mol.thiele()

def _find_transamidation_swap(reaction, *, expand=False):
    _prep_molecules(list(reaction.reactants) + list(reaction.products))
    react_amide_by_c = {}
    react_c_idx = {}
    react_n_info = {}
    for idx, mol in enumerate(reaction.reactants):
        amide_map = _amide_by_carbonyl(mol)
        for c_num, n_num in amide_map.items():
            react_amide_by_c[c_num] = n_num
            react_c_idx[c_num] = idx
        for num, atom in mol._atoms.items():
            if not _is_nitrogen(atom):
                continue
            react_n_info[num] = (idx, _atom_hcount(atom), mol)

    prod_amide_by_c = {}
    prod_n_all = set()
    prod_n_mol = {}
    for mol in reaction.products:
        prod_amide_by_c.update(_amide_by_carbonyl(mol))
        for num, atom in mol._atoms.items():
            if _is_nitrogen(atom):
                prod_n_all.add(num)
                prod_n_mol.setdefault(num, mol)

    for c_num, react_n in react_amide_by_c.items():
        prod_n = prod_amide_by_c.get(c_num)
        if prod_n is None:
            continue
        if prod_n != react_n:
            continue
        acyl_idx = react_c_idx.get(c_num)
        candidates = [
            n
            for n, (idx, _, _) in react_n_info.items()
            if idx != acyl_idx and n not in prod_n_all and n != react_n
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda n: (react_n_info[n][1], n),
            reverse=True,
        )
        cand_n = candidates[0]
        mapping = {prod_n: cand_n}
        prod_mol = prod_n_mol.get(prod_n)
        if prod_mol:
            prod_c = _select_n_attached_carbon(prod_mol, prod_n)
        else:
            prod_c = None
        cand_mol = react_n_info[cand_n][2]
        cand_c = _select_n_attached_carbon(cand_mol, cand_n)
        if prod_c and cand_c and cand_c not in prod_mol._atoms:
            mapping[prod_c] = cand_c
        if prod_mol and cand_mol:
            expanded, added = _expand_mapping_with_substructure(
                prod_mol,
                cand_mol,
                mapping,
                prod_n=prod_n,
                cand_n=cand_n,
            )
            if expand or added >= max(3, len(cand_mol) // 2):
                mapping = expanded
        return mapping
    return None

def _remap_product_atoms(reaction, mapping):
    if not mapping:
        return False
    for prod in reaction.products:
        keys = set(mapping) & set(prod._atoms)
        if not keys:
            continue
        prod_map = {k: v for k, v in mapping.items() if k in prod._atoms}
        existing = set(prod._atoms)
        for old, new in list(prod_map.items()):
            if new in existing and new not in prod_map:
                prod_map.pop(old)
        if not prod_map:
            continue
        prod.remap(prod_map)
        return True
    return False

def process_single_route(cgr_pathway, check_trans_error=True):
    for i, reaction in enumerate(cgr_pathway):
        if check_trans_error:
            swap_cgr = reaction.compose()
            expand_swap = _has_dynamic_bond_4_0(swap_cgr)
            swap_pair = _find_transamidation_swap(reaction, expand=expand_swap)
            if swap_pair:
                if _remap_product_atoms(reaction, swap_pair):
                    reaction.flush_cache()
        if i == 0:
            cgr = reaction.compose()
            atoms = reaction.products[0]._atoms.keys()
            if reaction.products[0].atoms_count != max(atoms):
                remapper = find_remap(list(atoms))
                temp_num = max(cgr._atoms)+1
                for key, value in remapper.items():
                    save_val = int(value)
                    cgr.remap({value: temp_num, key: value, value: key})
        else:
            curr_product = reaction.products[0]
            curr_product.kekule()
            curr_product.implicify_hydrogens()
            curr_product.thiele()
            
            for reactant in decomposed.reactants:
                reactant.kekule()
                reactant.implicify_hydrogens()
                reactant.thiele()
                try:
                    if len(reactant) == len(curr_product):
                        curr_remap = next(curr_product.get_mapping(reactant))
                        curr_cgr = reaction.compose()
                        max_num = max(cgr._atoms) + 1
                        curr_decomposed = ReactionContainer.from_cgr(curr_cgr)
                        lg_remap = {}
                        for product in curr_decomposed.products:
                            curr_max_num = max(curr_cgr._atoms) + 1
                            if curr_max_num > max_num:
                                max_num = curr_max_num
                            if len(product) == len(curr_product):
                                continue
                            else:
                                for atom_num in product:
                                    lg_remap[atom_num] = max_num
                                    max_num += 1
                        curr_cgr.remap(lg_remap)
                        curr_cgr.remap(curr_remap)
                        cgr = curr_cgr.compose(cgr)
                except:
                    pass
        decomposed = ReactionContainer.from_cgr(cgr)
    target_cgr = [cgr.substructure(c) for c in cgr.connected_components][0]
    return target_cgr
