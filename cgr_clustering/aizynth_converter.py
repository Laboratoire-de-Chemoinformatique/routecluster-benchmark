# aizynth_converter.py

from aizynthfinder.analysis import TreeAnalysis
from aizynthfinder.analysis.routes import RouteCollection
from aizynthfinder.reactiontree import ReactionTree
from typing import List, Any, Dict
from itertools import chain
from cgr_clustering import cgr_process
from tqdm import tqdm
from route_distances.ted.distances import distance_matrix
from route_distances.validation import validate_dict


def extract_routes_from_tree(app):
    """Extracts all solved routes from the AiZynthFinder search tree."""
    tree_analysis = TreeAnalysis(app.finder.tree)
    nodes = list(tree_analysis.search_tree.graph())

    solved = [node for node in nodes if not node.children and node.state.is_solved]

    route_collections = []
    for node in solved:
        rt = node.to_reaction_tree()
        rc = RouteCollection([rt], nodes=[node])
        route_collections.append(rc)
    return route_collections

def filter_unique_routes(route_collections: RouteCollection) -> RouteCollection:
    """Filters a RouteCollection to keep only unique retrosynthetic routes."""

    all_trees = list(chain.from_iterable(rc.reaction_trees for rc in route_collections))
    all_nodes = list(chain.from_iterable(rc.nodes for rc in route_collections))
    
    single_route_collection = RouteCollection(
        all_trees,
        nodes=all_nodes,
    )

    seen_hashes = set()
    unique_trees: List[ReactionTree] = []
    unique_nodes: List[Any] = []
    unique_dicts: List[Dict[str, Any]] = []

    for tree, node, dct in zip(
        single_route_collection.reaction_trees,
        single_route_collection.nodes,
        single_route_collection.dicts
    ):
        h = tree.hash_key()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique_trees.append(tree)
        unique_nodes.append(node)
        unique_dicts.append(dct)

    return RouteCollection(
        unique_trees,
        nodes=unique_nodes,
        dicts=unique_dicts
    )

def extract_pathway_aizynthfinder(node, parent_smiles=None):
    """Recursively extracts a pathway from a reaction tree node."""
    pathway = []
    if node.get('type') == 'reaction':
        for child in node.get('children', []):
            if child.get('type') == 'mol' and 'children' in child:
                for sub in child['children']:
                    pathway.extend(extract_pathway_aizynthfinder(sub, child['smiles']))
        reactants = [c['smiles'] for c in node['children'] if c['type']=='mol'][::-1]
        pathway.append([reactants, parent_smiles])
    else:
        for child in node.get('children', []):
            pathway.extend(extract_pathway_aizynthfinder(child, node.get('smiles')))
    return pathway


def extract_one_route_cgr(data, check_trans_error=True):
    root = data['dict']
    pathway = extract_pathway_aizynthfinder(root)
    cgr_pathway = cgr_process.route_smi_2_cgr(pathway, reverse=True)
    route_cgr = cgr_process.process_single_route(cgr_pathway, check_trans_error=check_trans_error)
    return route_cgr


def extract_all_route_cgrs(route_collection, check_trans_error=True):
    route_cgrs_dict = {}
    for i, data in tqdm(enumerate(route_collection)):
        route_cgr = extract_one_route_cgr(data, check_trans_error=check_trans_error)
        route_cgrs_dict[i] = route_cgr
    return route_cgrs_dict


def compute_ted_matrix(filtered_route_collection, content="both", timeout=None, validate=True):
    """
    Parameters
    ----------
    filtered_route_collection : RouteCollection
        Your AIZynthFinder routes collection
    content : str
        "molecules", "reactions", or "both"
    timeout : int | None
        If set, raises if TED calc takes too long (see route-distances docs)
    validate : bool
        Validate that each route dict matches the expected schema.

    Returns
    -------
    np.ndarray
        Square pairwise TED distance matrix (N x N)
    """
    # RouteCollection.dicts gives dict representation of each route
    routes = list(filtered_route_collection.dicts)

    if validate:
        for k, r in enumerate(routes):
            try:
                validate_dict(r)
            except Exception as e:
                raise ValueError(f"Route {k} failed validation: {e}") from e

    # Compute NxN TED matrix
    D = distance_matrix(routes, content=content, timeout=timeout)
    return D