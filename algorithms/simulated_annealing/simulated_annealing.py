import math
import random
from problems import Problem
from problems.n_queens import Nqueens


class SimulatedAnnealing:

    def __init__(self, general_functions: Problem):
        self.problem = general_functions

    # helper functions
    @staticmethod
    def __execute_with_probability(state, future_state, energy_change, temperature):
        def probability_function():
            return math.exp(energy_change / temperature)

        probability = probability_function()

        if random.uniform(0, 1) < probability:
            state = future_state

        return state

    # find solution
    def find_solution(self, initial_temperature: float, n: int, cooling_factor: float,
                      minimum_temperature: float):
        problem = self.problem

        def simulated_annealing():
            temperature = initial_temperature
            while temperature > minimum_temperature:
                for _ in range(n):
                    future_state = problem.get_random_future_state()
                    energy_change = (problem.heuristic(problem.get_current_state()) -
                                     problem.heuristic(future_state))
                    if energy_change > 0:
                        problem.update_current_state(future_state)
                    else:
                        problem.update_current_state(self.__execute_with_probability(problem.get_current_state(),
                                                                                     future_state,
                                                                                     energy_change, temperature))
                temperature = temperature * cooling_factor

        while not (problem.validate_state()):
            problem.update_current_state(problem.get_initial_state())
            simulated_annealing()

        return problem.get_current_state()


# Test
n_queens = Nqueens(4)
sim_ann = SimulatedAnnealing(n_queens)
print(sim_ann.find_solution(initial_temperature=5, n=50,
                            cooling_factor=0.1, minimum_temperature=0.1))
