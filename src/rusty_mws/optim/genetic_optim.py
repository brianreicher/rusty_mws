import random
import os
import time
import numpy as np

from rusty_mws.optim.base_optimizer import OptimizerBase
from ..algo import segment, extract_segmentation
from funlib.persistence import open_ds, graphs, Array

from funlib.evaluate import rand_voi


class GeneticOptimizer(OptimizerBase):
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
                        fragments_file,
                        fragments_dataset,
                        seg_file,
                        seg_dataset,
                        seeds_file,
                        seeds_dataset,
                        sample_name,
                        adj_bias_range,
                        lr_bias_range,
                        db_host,
                        db_name,
                        merge_function,)

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
            offspring: list = []
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
