import random
import time
import numpy as np
from ..rusty_segment_mws import *
from ..algo.global_mutex_agglom import segment
from ..algo.extract_seg_from_luts import extract_segmentation
from funlib.persistence import open_ds, graphs, Array
import mwatershed as mws
from tqdm import tqdm


class GridSearchOptimizer():
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

        # set the seeds and frags arrays
        self.fragments_file: str = fragments_file
        self.fragments_dataset: str = fragments_dataset
        self.seg_file: str = seg_file
        self.seg_dataset: str = seg_dataset
        self.seeds_file: str = seeds_file
        self.seeds_dataset: str = seeds_dataset

        self.frags: Array = open_ds(filename=fragments_file, ds_name=fragments_dataset)
        seeds: Array = open_ds(filename=seeds_file, ds_name=seeds_dataset)
        seeds = self.seeds.to_ndarray(self.frags.roi)
        self.seeds: np.ndarray = np.asarray(a=seeds, dtype=np.uint64)

        # handle db fetch
        print("Reading graph from DB ", self.db_name)
        start: float = time.time()

        print("Got Graph provider")

        roi = self.frags.roi

        print("Getting graph for roi %s" % roi)
        graph = self.graph_provider.get_graph(roi)

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

    def grid_search(
        self,
        eval_method:str="rand_voi",
        seg_range: tuple=(6000,14000),
    ) -> list:

        scores: list = []
        temp_edges: np.ndarray = self.edges
        temp_adj_scores: np.ndarray = self.adj_scores
        temp_lr_scores: np.ndarray = self.lr_scores
        print("Running grid search . . .")
        index: int = 0
        for a_bias in tqdm(np.arange(self.adj_bias_range[0], self.adj_bias_range[1] + 0.1, 0.1)):
            index += 1
            start_time: float = time.time()
            for l_bias in np.arange(self.lr_bias_range[0], self.lr_bias_range[1] + 0.1, 0.1):
                if eval_method.lower() == "rand_voi":
                    pass
                else
                    n_seg_run: int = self.get_num_segs(temp_edges, temp_adj_scores, temp_lr_scores, a_bias, l_bias)
                    if n_seg_run in seg_range:
                        scores.append((a_bias, l_bias, n_seg_run))
            np.savez_compressed(
                file="./gridsearch_biases.npz", grid=np.array(object=sorted(scores, key=lambda x: x[2]))
            )
            print(f"Completed {index}th iteration in {time.time()-start_time} sec")
        print("Completed grid search")
        return sorted(scores, key=lambda x: x[2], reverse=True)[:len(5)]

    @staticmethod
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
            seg_file=self.sample_name
        )

        seg: Array = open_ds(filename=self.seg_file, ds_name=self.seg_ds)

        seg: np.ndarray = seg.to_ndarray()

        seg: np.ndarray = np.asarray(seg, dtype=np.uint64)

        score_dict: dict = rand_voi(self.seeds, seg, True)

        print([score_dict[f"voi_split"], score_dict["voi_merge"]])
        return np.mean(a=[score_dict[f"voi_split"], score_dict["voi_merge"]])
