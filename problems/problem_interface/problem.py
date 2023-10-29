from abc import ABC, abstractmethod


class Problem(ABC):

    @abstractmethod
    def get_random_future_state(self):
        pass

    @abstractmethod
    def get_all_future_states(self):
        pass

    @abstractmethod
    def validate_state(self):
        pass

    @classmethod
    def heuristic(cls, state):
        pass

    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def update_current_state(self, state):
        pass


