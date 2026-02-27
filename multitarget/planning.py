from synplan.chem.utils import mol_from_smiles
from synplan.chem.reaction_routes.route_cgr import *
from synplan.chem.reaction_routes.clustering import *
from synplan.chem.reaction_routes.visualisation import cgr_display
from IPython.display import display, HTML, SVG
from synplan.utils.visualisation import get_route_svg, get_route_svg_from_json, render_svg
from CGRtools.reactor.reactor import Reactor
from typing import List
from CGRtools.containers import ReactionContainer
import pickle

def multi_target_route_generation(target_smiles, tree_config,
                                  reaction_rules,
                                  building_blocks,
                                  policy_function,
                                  evaluation_function):
    data = {}
    for label, example_smiles in target_smiles:
        print(f"Planning for {label}: {example_smiles}")

        target_molecule = mol_from_smiles(
            example_smiles,
            clean2d=True,
            standardize=True,
            clean_stereo=True
            )
        
        tree = Tree(
            target=target_molecule,
            config=tree_config,
            reaction_rules=reaction_rules,
            building_blocks=building_blocks,
            expansion_function=policy_function,
            evaluation_function=evaluation_function,
        )

        tree_solved = False
        for solved, node_id in tree:
            if solved:
                tree_solved = True
        all_route_cgrs = compose_all_route_cgrs(tree)
        all_sb_cgrs = compose_all_sb_cgrs(all_route_cgrs)
        clusters = cluster_routes(all_sb_cgrs, use_strat=False)
        print(f"Number of clusters routes: {len(tree.winning_nodes)}")
        print(f"Number of clusters formed: {len(clusters)}")
        print(f"Cluster Indices:", [i for i in clusters.keys()])

        tree._tqdm = None
        tree.reaction_rules = None
        #tree.building_blocks = None
        tree.expansion_function = None
        tree.evaluator = None

        data[label] = {
            "tree": tree,
            "clusters": clusters,
            'all_route_cgrs': all_route_cgrs,
            'all_sb_cgrs': all_sb_cgrs,
        }
    return data

def load_reaction_rules(file: str) -> List[Reactor]:
    """Loads the reaction rules from a pickle file and converts them into a list of
    Reactor objects if necessary.

    :param file: The path to the pickle file that stores the reaction rules.
    :return: A list of reaction rules as Reactor objects.
    """

    with open(file, "rb") as f:
        reaction_rules = pickle.load(f)

    reaction_rules_out = []
    for x, ids in reaction_rules:
        react = x.reactants
        prods = x.products
        if len (prods) == 1:
            prod = prods[0]
            if len(prod.split()) > 1:
                n = ReactionContainer(react, prod.split())
                #print("Multiple products detected in rule; using ReactionContainer.")
            else:
                n = ReactionContainer(react, [prod])
        else:
            n = ReactionContainer(react, prods)
        reaction_rules_out.append(Reactor(n))

    return tuple(reaction_rules_out)