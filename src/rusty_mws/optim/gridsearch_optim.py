import time
import numpy as np
from .base_optimizer import OptimizerBase
import mwatershed as mws
from tqdm import tqdm


class GridSearchOptimizer(OptimizerBase):
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
        super().__init__(
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            seg_file=seg_file,
            seg_dataset=seg_dataset,
            seeds_file=seeds_file,
            seeds_dataset=seeds_dataset,
            sample_name=sample_name,
            adj_bias_range=adj_bias_range,
            lr_bias_range=lr_bias_range,
            db_host=db_host,
            db_name=db_name,
            merge_function=merge_function,
        )

    def grid_search(
        self,
        eval_method: str = "rand_voi",
        seg_range: tuple = (6000, 14000),
    ) -> list:
        scores: list = []
        temp_edges: np.ndarray = self.edges
        temp_adj_scores: np.ndarray = self.adj_scores
        temp_lr_scores: np.ndarray = self.lr_scores
        print("Running grid search . . .")
        index: int = 0
        for a_bias in tqdm(
            np.arange(self.adj_bias_range[0], self.adj_bias_range[1] + 0.1, 0.1)
        ):
            index += 1
            start_time: float = time.time()
            for l_bias in np.arange(
                self.lr_bias_range[0], self.lr_bias_range[1] + 0.1, 0.1
            ):
                if eval_method.lower() == "rand_voi":
                    fitness: np.floating = self.evaluate_weight_biases(
                        adj_bias=a_bias,
                        lr_bias=l_bias,
                        edges=temp_edges,
                        adj_scores=temp_adj_scores,
                        lr_scores=temp_lr_scores,
                        out_dir=self.out_dir,
                    )
                    scores.append((a_bias, l_bias, fitness))
                else:
                    n_seg_run: int = self.get_num_segs(
                        edges=temp_edges,
                        adj_scores=temp_adj_scores,
                        lr_scores=temp_lr_scores,
                        adj_bias=a_bias,
                        lr_bias=l_bias,
                    )
                    if n_seg_run in seg_range:
                        scores.append((a_bias, l_bias, n_seg_run))
            np.savez_compressed(
                file="./gridsearch_biases.npz",
                grid=np.array(object=sorted(scores, key=lambda x: x[2])),
            )
            print(f"Completed {index}th iteration in {time.time()-start_time} sec")
        print("Completed grid search")
        return sorted(scores, key=lambda x: x[2], reverse=True)[: len(5)]

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
