from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def policy(self, state, log_dir: str = None):
        pass