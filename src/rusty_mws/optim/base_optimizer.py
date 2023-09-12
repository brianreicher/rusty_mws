import time
import os
import numpy as np
from funlib.persistence import open_ds, graphs, Array
from ..algo import segment, extract_segmentation
from funlib.evaluate import rand_voi


class OptimizerBase:
    def __init__(
        self,
        fragments_file: str,
        fragments_dataset: str,
        seg_file: str,
        seg_dataset: str,
        seeds_file: str,
        seeds_dataset: str,
        sample_name: str,
        adj_bias_range: tuple,
        lr_bias_range: tuple,
        db_host: str = "mongodb://localhost:27017",
        db_name: str = "seg",
        merge_function: str = "mwatershed",
    ) -> None:
        # set bias ranges
        self.adj_bias_range: tuple = adj_bias_range
        self.lr_bias_range: tuple = lr_bias_range

        # db hosting
        self.sample_name: str = sample_name
        self.graph_provider = graphs.MongoDbGraphProvider(
            db_name=db_name,
            host=db_host,
            mode="r+",
            nodes_collection=f"{self.sample_name}_nodes",
            meta_collection=f"{self.sample_name}_meta",
            edges_collection=self.sample_name + "_edges_" + merge_function,
            position_attribute=["center_z", "center_y", "center_x"],
        )
        self.merge_function: str = merge_function

        # set the seeds and frags arrays
        self.fragments_file: str = fragments_file
        self.fragments_dataset: str = fragments_dataset
        self.seg_file: str = seg_file
        self.seg_dataset: str = seg_dataset
        self.seeds_file: str = seeds_file
        self.seeds_dataset: str = seeds_dataset

        self.frags: Array = open_ds(filename=fragments_file, ds_name=fragments_dataset)
        seeds: Array = open_ds(filename=seeds_file, ds_name=seeds_dataset)
        seeds = seeds.to_ndarray(self.frags.roi)
        self.seeds: np.ndarray = np.asarray(a=seeds, dtype=np.uint64)

        # handle db fetch
        print("Reading graph from DB ", db_name)
        start: float = time.time()

        print("Got Graph provider")

        roi = self.frags.roi

        print("Getting graph for roi %s" % roi)
        graph = self.graph_provider.get_graph(roi=roi)

        print("Read graph in %.3fs" % (time.time() - start))

        if graph.number_of_nodes == 0:
            print("No nodes found in roi %s" % roi)
            return

        self.edges: np.ndarray = np.stack(arrays=list(graph.edges), axis=0)
        self.adj_scores: np.ndarray = np.array(
            object=[graph.edges[tuple(e)]["adj_weight"] for e in self.edges]
        ).astype(dtype=np.float32)
        self.lr_scores: np.ndarray = np.array(
            object=[graph.edges[tuple(e)]["lr_weight"] for e in self.edges]
        ).astype(dtype=np.float32)

        self.out_dir: str = os.path.join(self.fragments_file, "luts_full")
        os.makedirs(name=self.out_dir, exist_ok=True)

    def evaluate_weight_biases(
        self,
        adj_bias: float,
        lr_bias: float,
        edges: np.ndarray,
        adj_scores: np.ndarray,
        lr_scores: np.ndarray,
        out_dir: str,
    ) -> np.floating:
        segment(
            edges=edges,
            adj_scores=adj_scores,
            lr_scores=lr_scores,
            merge_function=self.merge_function,
            out_dir=out_dir,
            adj_bias=adj_bias,
            lr_bias=lr_bias,
        )
        extract_segmentation(
            fragments_file=self.fragments_file, 
            fragments_dataset=self.fragments_dataset, 
            seg_file=self.sample_name,
            seg_dataset=self.seg_dataset,
        )

        seg: Array = open_ds(filename=self.seg_file, ds_name=self.seg_ds)

        seg: np.ndarray = seg.to_ndarray()

        seg: np.ndarray = np.asarray(seg, dtype=np.uint64)

        score_dict: dict = rand_voi(self.seeds, seg, True)

        print([score_dict[f"voi_split"], score_dict["voi_merge"]])
        return np.mean(a=[score_dict[f"voi_split"], score_dict["voi_merge"]])

