from chython import smiles as smiles_chython
from chython.containers import ReactionContainer as ReactionContainerChython

from CGRtools import smiles as smiles_cgrtools
from CGRtools.containers import ReactionContainer, CGRContainer
from CGRtools.containers.bonds import DynamicBond
from CGRtools.algorithms.depict import *
from collections import defaultdict

from functools import partial
from math import atan2, sin, cos, hypot


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
        if not hasattr(reaction, "__cached_method_compose"):
            setattr(reaction, "__cached_method_compose", None)
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

def process_single_route(cgr_pathway):
    for i, reaction in enumerate(cgr_pathway):
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
    #target_cgr = cgr_enhance(target_cgr)
    return target_cgr