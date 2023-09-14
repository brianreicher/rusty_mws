import random
import numpy as np

from .base_optimizer import OptimizerBase


class ParticleSwarmOptimizer(OptimizerBase):
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

    def initialize_particles(self, population_size) -> list:
        particles = []
        for _ in range(population_size):
            position = (
                random.uniform(self.adj_bias_range[0], self.adj_bias_range[1]),
                random.uniform(self.lr_bias_range[0], self.lr_bias_range[1]),
            )
            velocity = (
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            )
            personal_best_position = position
            personal_best_score = float("inf")
            particles.append(
                {
                    "position": position,
                    "velocity": velocity,
                    "personal_best_position": personal_best_position,
                    "personal_best_score": personal_best_score,
                }
            )
        return particles

    def evaluate_particle(self, particle):
        adj_bias, lr_bias = particle["position"]
        score = self.evaluate_weight_biases(
            adj_bias, lr_bias, self.edges, self.adj_scores, self.lr_scores, self.out_dir
        )
        return score

    def optimize(
        self, num_generations: int, population_size: int
    ) -> list:
        particles = self.initialize_particles(population_size)
        global_best_position = None
        global_best_score = float("inf")

        for generation in range(num_generations):
            print("Generation:", generation)

            for particle in particles:
                # Evaluate the particle's position
                score = self.evaluate_particle(particle)

                # Update personal best
                if score < particle["personal_best_score"]:
                    particle["personal_best_score"] = score
                    particle["personal_best_position"] = particle["position"]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particle["position"]

                # Update particle velocity and position
                inertia_term = np.multiply(
                    self.inertia_weight, particle["velocity"]
                )
                cognitive_term = np.multiply(
                    self.c1 * random.random(), np.subtract(
                        particle["personal_best_position"], particle["position"]
                    )
                )
                social_term = np.multiply(
                    self.c2 * random.random(), np.subtract(
                        global_best_position, particle["position"]
                    )
                )
                particle["velocity"] = np.add(
                    inertia_term, np.add(cognitive_term, social_term)
                )
                particle["position"] = np.add(
                    particle["position"], particle["velocity"]
                )

            print(f"Iteration {generation}: Best Score = {global_best_score}")

        return global_best_position
