import logging


class Nqueens:

    def __init__(self, n: int):
        self.n = n
        self.initial_state = self.__create_world()

    @staticmethod
    def transition(queens, queen_idx: int, movement: int):
        if movement not in [-1, 1]:
            logging.info("invalid input")
            return
        if queen_idx not in range(len(queens)):
            logging.info("invalid queen")
            return

        future_state = queens.copy()
        if 0 <= queens[queen_idx]+movement <= len(queens)-1:
            future_state[queen_idx] = queens[queen_idx] + movement
        else:
            logging.info("invalid movement")

        return future_state

    def  __create_world(self):
        return [0] * self.n

    @staticmethod
    def threatens(queens, idx_1, idx_2):
        if idx_1 == idx_2:
            return 0
        if queens[idx_1] == queens[idx_2] or abs(idx_1 - idx_2) == abs(queens[idx_1] - queens[idx_2]):
            return 1
        else:
            return 0

    @staticmethod
    def threats(queens):
        sum_threats = 0
        for idx_1 in range(len(queens) - 1):
            for idx_2 in range(idx_1 + 1, len(queens)):
                sum_threats += Nqueens.threatens(queens, idx_1, idx_2)
        return sum_threats

    @staticmethod
    def validate_state(queens):
        if Nqueens.threats(queens) > 0:
            return False
        else:
            return True
