"""
Core interfaces for the project.
Defines common interfaces that all components should implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Protocol
import numpy as np
import tensorflow as tf


class Renderable(Protocol):
    """Interface for objects that can be rendered"""
    
    def render(self, ax=None, **kwargs) -> None:
        """Render the object"""
        ...


class Loggable(Protocol):
    """Interface for objects that can be logged"""
    
    def get_log_data(self) -> Dict[str, Any]:
        """Get data for logging"""
        ...
    
    def save_log(self, path: str) -> None:
        """Save log data to file"""
        ...


class Configurable(Protocol):
    """Interface for objects that can be configured"""
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration"""
        ...


class Stateful(Protocol):
    """Interface for objects that maintain state"""
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        ...
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state"""
        ...
    
    def reset_state(self) -> None:
        """Reset to initial state"""
        ...


class Observable(Protocol):
    """Interface for objects that can be observed"""
    
    def get_observation(self) -> np.ndarray:
        """Get current observation"""
        ...
    
    def get_observation_space(self):
        """Get observation space"""
        ...


class Actionable(Protocol):
    """Interface for objects that can be observed"""
    
    def take_action(self, action: Any) -> Tuple[Any, bool]:
        """Take an action and return result and success flag"""
        ...
    
    def get_action_space(self):
        """Get action space"""
        ...


class Learnable(Protocol):
    """Interface for objects that can learn"""
    
    def train_step(self, *args, **kwargs):
        """Perform a training step"""
        ...
    
    def save_model(self, path: str) -> None:
        """Save the model"""
        ...
    
    def load_model(self, path: str) -> None:
        """Load the model"""
        ...


class SwarmMember(Protocol):
    """Interface for swarm members"""
    
    def get_position(self) -> np.ndarray:
        """Get current position"""
        ...
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity"""
        ...
    
    def get_role(self) -> str:
        """Get current role"""
        ...
    
    def set_role(self, role: str) -> None:
        """Set role"""
        ...
    
    def get_swarm_id(self) -> int:
        """Get swarm ID"""
        ...
    
    def set_swarm_id(self, swarm_id: int) -> None:
        """Set swarm ID"""
        ... 