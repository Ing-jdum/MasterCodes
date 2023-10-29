import math
import random
from general_functions import Nqueens


class NqueensSimulatedAnnealing:
    def __init__(self, n: int):
        self.n = n
        self.n_queens = Nqueens(n)
        self.initial_state = self.n_queens.initial_state

    # helper functions
    def __generate_random_future_state(self, queens):
        future_state = queens
        while future_state == queens:
            random_queen = random.randint(0, self.n - 1)
            random_movement = random.choice([1, -1])
            future_state = self.n_queens.transition(queens, random_queen, random_movement)
        return future_state

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
        def simulated_annealing(state=self.initial_state):
            temperature = initial_temperature
            while temperature > minimum_temperature:
                for i in range(n):
                    future_state = self.__generate_random_future_state(state)
                    energy_change = Nqueens.threats(state) - Nqueens.threats(future_state)
                    if energy_change > 0:
                        state = future_state
                    else:
                        state = self.__execute_with_probability(state, future_state, energy_change, temperature)
                temperature = temperature * cooling_factor
            return state

        result = self.initial_state
        while Nqueens.threats(result) != 0:
            result = simulated_annealing()

        return result


sim_ann = NqueensSimulatedAnnealing(4)
print(sim_ann.find_solution(initial_temperature=5, n=100,
                            cooling_factor=0.1, minimum_temperature=0.1))
