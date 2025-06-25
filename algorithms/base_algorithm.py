from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def policy(self, state):
        pass