from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass