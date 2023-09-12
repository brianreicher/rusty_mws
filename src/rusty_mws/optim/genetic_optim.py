import random
import os
import time
import numpy as np
from ..algo import segment, extract_segmentation
from funlib.persistence import open_ds, graphs, Array

from funlib.evaluate import rand_voi


class GeneticOptimizer():
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

    @staticmethod
    def crossover(parent1, parent2) -> tuple:
        # Perform crossover by blending the weight biases of the parents
        alpha: float = random.uniform(0.0, 1.0)  # Blend factor

        adj_bias_parent1, lr_bias_parent1 = parent1[0], parent1[1]
        adj_bias_parent2, lr_bias_parent2 = parent2[0], parent2[1]

        # Blend the weight biases
        adj_bias_child: float = alpha * adj_bias_parent1 + (1 - alpha) * adj_bias_parent2
        lr_bias_child: float = alpha * lr_bias_parent1 + (1 - alpha) * lr_bias_parent2

        return adj_bias_child, lr_bias_child

    @staticmethod
    def mutate(individual: tuple, mutation_rate:float=0.1, mutation_strength:float=0.1) -> tuple:
        # Perform mutation by adding random noise to the weight biases
        adj_bias, lr_bias = individual

        # Mutate the weight biases with a certain probability
        if random.uniform(a=0.0, b=1.0) < mutation_rate:
            # Add random noise to the weight biases
            adj_bias += random.uniform(a=-mutation_strength, b=mutation_strength)
            lr_bias += random.uniform(a=-mutation_strength, b=mutation_strength)

        return adj_bias, lr_bias

    def optimize(
        self,
        num_generations: int,
        population_size: int,
    ) -> list:
        # Initialize the population
        population: list = []
        for _ in range(population_size):
            adj_bias: float = random.uniform(*self.adj_bias_range)
            lr_bias: float = random.uniform(*self.lr_bias_range)
            population.append((adj_bias, lr_bias))

        # evo loop
        for generation in range(num_generations):
            print("Generation:", generation)

            # Evaluate the fitness of each individual in the population
            fitness_values: list = []
            temp_edges: np.ndarray = self.edges
            temp_adj_scores: np.ndarray = self.adj_scores
            temp_lr_scores: np.ndarray = self.lr_scores

            for adj_bias, lr_bias in population:
                print("BIASES:", adj_bias, lr_bias)
                fitness: np.floating = self.evaluate_weight_biases(
                    adj_bias=adj_bias,
                    lr_bias=lr_bias,
                    edges=temp_edges,
                    adj_scores=temp_adj_scores,
                    lr_scores=temp_lr_scores,
                    out_dir=self.out_dir,
                )
                fitness_values.append((adj_bias, lr_bias, fitness))

            # Sort individuals by fitness (descending order)
            fitness_values.sort(key=lambda x: x[2], reverse=True)

            # Select parents for the next generation
            parents: list = fitness_values[: population_size // 2]
            parents: list = [parent[:2] for parent in parents]

            # Create the next generation through crossover and mutation
            offspring = []
            for _ in range(population_size - len(parents)):
                parent1 = random.choice(seq=parents)
                parent2 = random.choice(seq=parents)
                child: tuple = self.crossover(parent1=parent1, parent2=parent2)
                child: tuple = self.mutate(individual=child)
                offspring.append(child)

            # Combine parents and offspring to form the new population
            population = parents + offspring

            fvals: list = sorted(
                fitness_values, key=lambda x: x[2], reverse=True
            )  # [:len(population)//2]

            # Extract the baises from the fitness values
            adj: list = [x[0] for x in fvals]
            lr: list = [x[1] for x in fvals]
            score: list = [x[2] for x in fvals]

            # Save the biases as an npz file
            np.savez(file=f"./optimal_biases_{generation}.npz", adj=adj, lr=lr, score=score)

        # Return the best weight biases found in the last generation
        best_biases: list = sorted(fitness_values, key=lambda x: x[2], reverse=True)[
            : len(population)
        ]
        return best_biases

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
