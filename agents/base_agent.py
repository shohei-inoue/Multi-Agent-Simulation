"""
Base agent class that defines the interface for all agents.
Implements common interfaces for consistency across the project.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf

from core.interfaces import Configurable, Stateful, Loggable, Learnable


class BaseAgent(Configurable, Stateful, Loggable, Learnable):
    """Base class for all agents in the system"""
    
    def __init__(self, env, algorithm, model=None, **kwargs):
        self.env = env
        self.algorithm = algorithm
        self.model = model
        self._config = {}
        self._state = {}
        
        # Initialize configuration
        self._init_config(**kwargs)
    
    def _init_config(self, **kwargs):
        """Initialize agent configuration"""
        self._config.update(kwargs)
    
    @abstractmethod
    def get_action(self, state: Dict[str, Any], episode: int = 0, log_dir: Optional[str] = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """Get action from current state"""
        pass
    
    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the agent"""
        pass
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self._config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        self._config.update(config)
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self._state.copy()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self._state.update(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self._state.clear()
    
    # Loggable interface implementation
    def get_log_data(self) -> Dict[str, Any]:
        """Get data for logging"""
        return {
            "config": self.get_config(),
            "state": self.get_state(),
            "model_info": self._get_model_info() if self.model else None
        }
    
    def save_log(self, path: str):
        """Save log data to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.get_log_data(), f, indent=2, default=str)
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        if not self.model:
            return {}
        
        return {
            "model_type": type(self.model).__name__,
            "trainable_params": len(self.model.trainable_variables) if hasattr(self.model, 'trainable_variables') else 0
        }
    
    # Learnable interface implementation
    def train_step(self, *args, **kwargs):
        """Perform a training step"""
        if hasattr(self, 'model') and self.model:
            return self.model.train_step(*args, **kwargs)
        raise NotImplementedError("Training not implemented for this agent")
    
    def save_model(self, path: str):
        """Save the model"""
        if hasattr(self, 'model') and self.model:
            self.model.save(path)
        else:
            raise NotImplementedError("Model saving not implemented for this agent")
    
    def load_model(self, path: str):
        """Load the model"""
        if hasattr(self, 'model') and self.model:
            self.model = tf.keras.models.load_model(path)
        else:
            raise NotImplementedError("Model loading not implemented for this agent")
    
    # Additional utility methods
    def get_action_space(self):
        """Get action space from environment"""
        return self.env.action_space if hasattr(self.env, 'action_space') else None
    
    def get_observation_space(self):
        """Get observation space from environment"""
        return self.env.observation_space if hasattr(self.env, 'observation_space') else None