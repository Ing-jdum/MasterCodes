from problem_interface import Problem
import logging
import random


class Nqueens(Problem):
    def __init__(self, n: int):
        self.n = n
        self.length = n - 1
        self.initial_state = [0] * n
        self.current_state = self.initial_state.copy()

    def transition(self, queen_idx: int, movement: int):
        future_state = self.current_state.copy()

        if movement not in [-1, 1]:
            logging.info("invalid input")
        elif queen_idx not in range(self.n):
            logging.info("invalid queen")
        elif not (0 <= self.current_state[queen_idx] + movement <= self.length):
            logging.info("invalid movement")
        else:
            future_state[queen_idx] = self.current_state[queen_idx] + movement

        return future_state

    def get_random_future_state(self):
        while True:
            random_queen = random.randint(0, self.length)
            random_movement = random.choice([1, -1])
            future_state = self.transition(random_queen, random_movement)
            if future_state != self.current_state:
                return future_state

    def get_all_future_states(self):
        future_states = [
            self.transition(queen, movement)
            for queen in range(self.n)
            for movement in [1, -1]
            if self.transition(queen, movement) != self.current_state
        ]
        return future_states

    @staticmethod
    def threatens(queens_state, idx_1, idx_2):
        return (
            0 if idx_1 == idx_2 else
            1 if queens_state[idx_1] == queens_state[idx_2] or abs(idx_1 - idx_2) == abs(
                queens_state[idx_1] - queens_state[idx_2]) else
            0
        )

    @classmethod
    def heuristic(cls, queens_state):
        return sum(
            Nqueens.threatens(queens_state, idx_1, idx_2)
            for idx_1 in range(len(queens_state) - 1)
            for idx_2 in range(idx_1 + 1, len(queens_state))
        )

    def validate_state(self):
        return Nqueens.heuristic(self.current_state) == 0

    def get_initial_state(self):
        return self.initial_state

    def get_current_state(self):
        return self.current_state

    def update_current_state(self, state):
        self.current_state = state
