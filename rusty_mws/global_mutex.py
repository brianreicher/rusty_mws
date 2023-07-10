import logging
import os
import time

import numpy as np
import mwatershed as mws
import networkx as nx
from funlib.persistence import open_ds, Array, graphs


logger: logging.Logger = logging.getLogger(__name__)

def global_mutex_watershed_on_super_voxels(fragments_file:str,
                                        fragments_dataset,
                                        sample_name:str,
                                        merge_function:str="mwatershed",
                                        adj_bias:float=7., 
                                        lr_bias:float=-1.2) -> bool:
    
    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    print("Reading graph from DB ", db_name)
    start = time.time()

    graph_provider = graphs.MongoDbGraphProvider(
        db_name,
        db_host,
        mode="r+",
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
        edges_collection=sample_name + "_edges_" + merge_function,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    print("Got Graph provider")

    fragments = open_ds(fragments_file, fragments_dataset)

    print("Opened fragments")

    roi = fragments.roi

    print("Getting graph for roi %s" % roi)

    graph = graph_provider.get_graph(roi)

    print("Read graph in %.3fs" % (time.time() - start))

    if graph.number_of_nodes == 0:
        print("No nodes found in roi %s" % roi)
        return
    nodes: np.ndarray = np.array(graph.nodes)
    edges: np.ndarray = np.stack(list(graph.edges), axis=0)
    adj_scores: np.ndarray = np.array([graph.edges[tuple(e)]["adj_weight"] for e in edges]).astype(
        np.float32
    )
    lr_scores: np.ndarray = np.array([graph.edges[tuple(e)]["lr_weight"] for e in edges]).astype(
        np.float32
    )

    print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    out_dir: str = os.path.join(fragments_file, "luts_full")

    os.makedirs(out_dir, exist_ok=True)

    start = time.time()



    segment(
        nodes=nodes,
        edges=edges,
        adj_scores=adj_scores,
        lr_scores=lr_scores,
        merge_function=merge_function,
        out_dir=out_dir,
        adj_bias=adj_bias,
        lr_bias=lr_bias
    )
    
    print("Created and stored lookup tables in %.3fs" % (time.time() - start))
    return True

def segment(nodes, edges, adj_scores, lr_scores, merge_function, out_dir, adj_bias, lr_bias) -> None:

    edges: list[tuple] = [
        (adj + adj_bias, u, v)
        for adj, (u, v) in zip(adj_scores, edges)
        if not np.isnan(adj) and adj is not None
    ] + [
        (lr_adj + lr_bias, u, v)
        for lr_adj, (u, v) in zip(lr_scores, edges)
        if not np.isnan(lr_adj) and lr_adj is not None
    ]
    edges = sorted(
        edges,
        key=lambda edge: abs(edge[0]),
        reverse=True,
    )
    edges = [(bool(aff > 0), u, v) for aff, u, v in edges]
    lut = mws.cluster(edges)
    inputs, outputs = zip(*lut)

    start: float = time.time()
    print("%.3fs" % (time.time() - start))

    start = time.time()
    lut = np.array([inputs, outputs])

    print("%.3fs" % (time.time() - start))

    lookup = "seg_%s" % (merge_function)
    lookup = lookup.replace("/", "-")

    out_file: str = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, fragment_segment_lut=lut, edges=edges)


    print("%.3fs" % (time.time() - start))