import random
import time
import numpy as np
from rusty_mws.rusty_segment_mws import *
from ..algo import segment, extract_segmentation
from funlib.persistence import open_ds, graphs, Array
import mwatershed as mws
from tqdm import tqdm 
from funlib.evaluate import rand_voi

from .optimizer import Optimizer

class GeneticOptimizer(Optimizer):

    def __init__(self, param_space:dict, adj_bias_range:tuple, lr_bias_range:tuple) -> None:
        super().__init__(param_space)

        # set bias ranges
        self.adj_bias_range: tuple = adj_bias_range
        self.lr_bias_range: tuple = lr_bias_range

    @staticmethod
    def crossover(parent1, parent2) -> tuple:
        # Perform crossover by blending the weight biases of the parents
        alpha = random.uniform(0.0, 1.0)  # Blend factor

        adj_bias_parent1, lr_bias_parent1 = parent1[0], parent1[1]
        adj_bias_parent2, lr_bias_parent2 = parent2[0], parent2[1]

        # Blend the weight biases
        adj_bias_child = alpha * adj_bias_parent1 + (1 - alpha) * adj_bias_parent2
        lr_bias_child = alpha * lr_bias_parent1 + (1 - alpha) * lr_bias_parent2

        return adj_bias_child, lr_bias_child

    @staticmethod
    def mutate(individual, mutation_rate=0.1, mutation_strength=0.1) -> tuple:
        # Perform mutation by adding random noise to the weight biases
        adj_bias, lr_bias = individual

        # Mutate the weight biases with a certain probability
        if random.uniform(0.0, 1.0) < mutation_rate:
            # Add random noise to the weight biases
            adj_bias += random.uniform(-mutation_strength, mutation_strength)
            lr_bias += random.uniform(-mutation_strength, mutation_strength)

        return adj_bias, lr_bias


    def evo_algo(population_size, num_generations, adj_bias_range, lr_bias_range,
                seg_file="./validation.zarr", seg_ds="pred_seg", rasters_file="../../data/xpress-challenge.zarr",
                fragments_file="./validation.zarr", fragments_dataset="frag_seg",
                rasters_ds="volumes/validation_gt_rasters", sample_name:str="htem39454661040933637", merge_function="mwatershed"):
        # Initialize the population
        population: list = []
        for _ in range(population_size):
            adj_bias = random.uniform(*adj_bias_range)
            lr_bias = random.uniform(*lr_bias_range)
            population.append((adj_bias, lr_bias))

        # set the rasters array
        frag: Array = open_ds(fragments_file, fragments_dataset)
        rasters: Array = open_ds(rasters_file, rasters_ds)

        print("Loading rasters . . .")
        rasters = rasters.to_ndarray(frag.roi)
        rasters = np.asarray(rasters, np.uint64)
        
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

        out_dir: str = os.path.join(fragments_file, "luts_full")
        os.makedirs(out_dir, exist_ok=True)

        # evo loop
        for generation in range(num_generations):
            print("Generation:", generation)

            # Evaluate the fitness of each individual in the population
            fitness_values = []
            for adj_bias, lr_bias in population:
                print("BIASES:", adj_bias, lr_bias)
                fitness = self.evaluate_weight_biases(adj_bias, lr_bias, rasters, seg_file,
                                                seg_ds, sample_name, edges, adj_scores,
                                                    lr_scores, merge_function, out_dir, fragments_file, fragments_dataset)
                fitness_values.append((adj_bias, lr_bias, fitness))


            # Sort individuals by fitness (descending order)
            fitness_values.sort(key=lambda x: x[2], reverse=True)

            # Select parents for the next generation
            parents = fitness_values[:population_size//2]
            parents = [parent[:2] for parent in parents]


            # Create the next generation through crossover and mutation
            offspring = []
            for _ in range(population_size - len(parents)):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = crossover(parent1, parent2) 
                child = mutate(child) 
                offspring.append(child)

            # Combine parents and offspring to form the new population
            population = parents + offspring

            fvals = sorted(fitness_values, key=lambda x: x[2], reverse=True) #[:len(population)//2]

            # Extract the baises from the fitness values
            adj = [x[0] for x in fvals]
            lr = [x[1] for x in fvals]
            score = [x[2] for x in fvals]

            # Save the biases as an npz file
            np.savez(f"./optimal_biases_{generation}.npz", adj=adj, lr=lr, score=score)

        # Return the best weight biases found in the last generation
        best_biases = sorted(fitness_values, key=lambda x: x[2], reverse=True)[:len(population)]
        return best_biases

    def evaluate_weight_biases(adj_bias, 
                            lr_bias, 
                            rasters, 
                            seg_file, 
                            seg_ds, 
                            sample_name, 
                            edges, 
                            adj_scores, 
                            lr_scores, 
                            merge_function, 
                            out_dir, 
                            fragments_file, 
                            fragments_dataset) -> np.floating:
        # Call the function that performs the agglomeration step with the given weight biases
        segment(edges, adj_scores, lr_scores, merge_function, out_dir, adj_bias, lr_bias)
        extract_segmentation(fragments_file, fragments_dataset, sample_name)

        seg: Array = open_ds(filename=seg_file, ds_name=seg_ds)

        seg: np.ndarray = seg.to_ndarray()

        seg: np.ndarray = np.asarray(seg, dtype=np.uint64)
        
        score_dict: dict = rand_voi(rasters, seg, True)
        print([score_dict[f"voi_split"], score_dict["voi_merge"]])
        return np.mean(a=[score_dict[f"voi_split"], score_dict["voi_merge"]])
