from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, param_space) -> None:
        """Initialize the optimizer with the parameter space.

        Args:
            param_space (dict): 
                A dictionary representing the search space of the parameters.
                Keys are parameter names, and values are tuples (min_val, max_val).
        """
        self.param_space: dict = param_space

    @abstractmethod
    def initialize_population(self, population_size) -> None:
        """Initialize the population of solutions for the optimization.

        Args:
            population_size (integer): 
                The number of individuals in the population.
        """
        pass

    @abstractmethod
    def update_population(self, offspring) -> None:
        """Update the population with the newly created offspring.

        Args:
            offspring (list): 
                A list of offspring individuals.
        """
        pass

    @abstractmethod
    def optimize(self, num_generations) -> None:
        """Perform the optimization process for a given number of generations.

        Args:
            num_generations (integer): 
                The number of generations to run the optimization.
        """
        pass
