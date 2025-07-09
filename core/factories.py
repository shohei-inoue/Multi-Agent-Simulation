"""
Factory patterns for creating components.
Provides centralized factory methods for all major components.
"""

from typing import Dict, Any, Type, Optional
from abc import ABC, abstractmethod
import tensorflow as tf

from core.interfaces import Configurable, Stateful, Renderable


class BaseFactory(ABC):
    """Base factory class"""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str, cls: Type):
        """Register a class with the factory"""
        self._registry[name] = cls
    
    def create(self, name: str, **kwargs) -> Any:
        """Create an instance"""
        if name not in self._registry:
            raise ValueError(f"Unknown type: {name}")
        return self._registry[name](**kwargs)
    
    def get_available_types(self) -> list:
        """Get list of available types"""
        return list(self._registry.keys())


class AlgorithmFactory(BaseFactory):
    """Factory for algorithms"""
    
    def __init__(self):
        super().__init__()
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Setup default algorithm types"""
        from algorithms.vfh_fuzzy import AlgorithmVfhFuzzy
        self.register("vfh_fuzzy", AlgorithmVfhFuzzy)


class AgentFactory(BaseFactory):
    """Factory for agents"""
    
    def __init__(self):
        super().__init__()
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Setup default agent types"""
        from agents.agent_a2c import A2CAgent
        self.register("a2c", A2CAgent)
    
    def create_agent(self, agent_type: str, **kwargs):
        """Create agent with proper setup"""
        agent = self.create(agent_type, **kwargs)
        
        # Setup optimizer if needed
        if "optimizer" in kwargs and isinstance(kwargs["optimizer"], str):
            agent.optimizer = self._create_optimizer(
                kwargs["optimizer"], 
                kwargs.get("learning_rate", 0.001)
            )
        
        return agent
    
    def _create_optimizer(self, optimizer_type: str, learning_rate: float):
        """Create optimizer"""
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop
        }
        
        if optimizer_type not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizers[optimizer_type](learning_rate=learning_rate)


class ModelFactory(BaseFactory):
    """Factory for models"""
    
    def __init__(self):
        super().__init__()
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Setup default model types"""
        from models.actor_critic import ModelActorCritic
        self.register("actor-critic", ModelActorCritic)


class EnvironmentFactory(BaseFactory):
    """Factory for environments"""
    
    def __init__(self):
        super().__init__()
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Setup default environment types"""
        from envs.env import Env
        self.register("exploration", Env)


# Global factory instances
algorithm_factory = AlgorithmFactory()
agent_factory = AgentFactory()
model_factory = ModelFactory()
environment_factory = EnvironmentFactory()


def create_component(component_type: str, factory_type: str, **kwargs):
    """Generic component creation"""
    factories = {
        "algorithm": algorithm_factory,
        "agent": agent_factory,
        "model": model_factory,
        "environment": environment_factory
    }
    
    if factory_type not in factories:
        raise ValueError(f"Unknown factory type: {factory_type}")
    
    return factories[factory_type].create(component_type, **kwargs) 