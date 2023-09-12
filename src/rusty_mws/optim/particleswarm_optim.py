import random
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
                        merge_function=merge_function,)

    def initialize_population(self, population_size):
        self.swarm = []
        for _ in range(population_size):
            particle = {}
            for param_name, (min_val, max_val) in self.param_space.items():
                particle[param_name] = random.uniform(min_val, max_val)
            particle["velocity"] = {
                param_name: 0.0 for param_name in self.param_space.keys()
            }
            particle["best_position"] = particle.copy()
            self.swarm.append(particle)

    def update_population(self, offspring):
        # Replace the entire population with the new offspring
        self.swarm = offspring

    def update_particle_velocity(self, particle, global_best_position):
        for param_name in self.param_space.keys():
            cognitive_component = (
                self.cognitive_weight
                * random.random()
                * (particle["best_position"][param_name] - particle[param_name])
            )
            social_component = (
                self.social_weight
                * random.random()
                * (global_best_position[param_name] - particle[param_name])
            )
            particle["velocity"][param_name] = (
                self.inertia_weight * particle["velocity"][param_name]
                + cognitive_component
                + social_component
            )

    def move_particle(self, particle):
        for param_name in self.param_space.keys():
            particle[param_name] += particle["velocity"][param_name]

            # Ensure particles stay within the parameter space
            min_val, max_val = self.param_space[param_name]
            particle[param_name] = max(min_val, min(particle[param_name], max_val))

    def update_swarm(self, global_best_position):
        for particle in self.swarm:
            self.update_particle_velocity(particle, global_best_position)
            self.move_particle(particle)

    def evaluate_fitness(self, particle):
        """
        Implement this method to evaluate the fitness of a particle's position.
        It should return a scalar value, where lower values indicate better fitness.
        """
        # Example: You can define a fitness function here based on your optimization problem.
        # For demonstration purposes, let's assume we want to minimize the sum of squared values of parameters.
        fitness = sum(val**2 for val in particle.values())
        return fitness

    def optimize(self, num_generations=100):
        self.initialize_population(self.swarm_size)

        global_best_position = min(self.swarm, key=lambda x: self.evaluate_fitness(x))

        for _ in range(num_generations):
            self.update_swarm(global_best_position)

            for particle in self.swarm:
                if self.evaluate_fitness(particle) < self.evaluate_fitness(
                    global_best_position
                ):
                    global_best_position = particle.copy()

        return global_best_position
