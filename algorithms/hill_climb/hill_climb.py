import math
import random
from problems import Problem
from problems.n_queens import Nqueens


class HillClimb:

    def __init__(self, general_functions: Problem):
        self.problem = general_functions

    # find solution
    def find_solution(self, n: int):
        problem = self.problem

        def hill_climb():
            for _ in range(n):
                future_state = problem.get_random_future_state()
                energy_change = (problem.heuristic(problem.get_current_state()) -
                                 problem.heuristic(future_state))
                if energy_change >= 0:
                    problem.update_current_state(future_state)
                print(problem.get_current_state())
                print(problem.validate_state())

        while not (problem.validate_state()):
            problem.update_current_state(problem.get_initial_state())
            hill_climb()

        return problem.get_current_state()


# Test
n_queens = Nqueens(4)
hill_climb = HillClimb(n_queens)
print(hill_climb.find_solution(n=20))
