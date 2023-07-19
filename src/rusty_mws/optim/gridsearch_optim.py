import random
import time
import numpy as np
from rusty_mws.rusty_segment_mws import *
from rusty_mws.global_mutex import segment
from rusty_mws.extract_seg_from_luts import extract_segmentation
from funlib.persistence import open_ds, graphs, Array
import mwatershed as mws
from tqdm import tqdm 
from funlib.evaluate import rand_voi


def grid_search(adj_bias_range:tuple, lr_bias_range:tuple, 
                      sample_name:str="htem4413041148969302336", 
                      merge_function:str="mwatershed",
                      fragments_file:str="./validation.zarr",
                      fragments_dataset:str="frag_seg",):
    
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

    edges: np.ndarray = np.stack(list(graph.edges), axis=0)
    adj_scores: np.ndarray = np.array([graph.edges[tuple(e)]["adj_weight"] for e in edges]).astype(
        np.float32
    )
    lr_scores: np.ndarray = np.array([graph.edges[tuple(e)]["lr_weight"] for e in edges]).astype(
        np.float32
    )

    scores: list = []
    print("Running grid search . . .")
    index: int = 0
    for a_bias in tqdm(np.arange(adj_bias_range[0], adj_bias_range[1] + 0.1, 0.1)):
        index+=1
        start_time: float = time.time()
        for l_bias in np.arange(lr_bias_range[0], lr_bias_range[1] + 0.1, 0.1):
            n_seg_run: int = get_num_segs(edges, adj_scores, lr_scores, a_bias, l_bias)
            if 6000<n_seg_run<14000:
                scores.append((a_bias, l_bias, n_seg_run))
        np.savez_compressed("./gridsearch_biases.npz", grid=np.array(sorted(scores, key=lambda x: x[2])))
        print(f"Completed {index}th iteration in {time.time()-start_time} sec")
    print("Completed grid search")


def get_num_segs(edges, adj_scores, lr_scores, adj_bias, lr_bias) -> None:

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

    lut = np.array([inputs, outputs])

    return len(np.unique(lut[1]))
