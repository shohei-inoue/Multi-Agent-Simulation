"""
Base algorithm class that defines the interface for all algorithms.
Implements common interfaces for consistency across the project.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import tensorflow as tf

from core.interfaces import Configurable, Stateful, Loggable, Renderable


class BaseAlgorithm(Configurable, Stateful, Loggable, Renderable):
    """Base class for all algorithms in the system"""
    
    def __init__(self, env=None, **kwargs):
        self.env = env
        self._config = {}
        self._state = {}
        self._render_flag = False
        self._render_components = {}
        
        # Initialize configuration
        self._init_config(**kwargs)
    
    def _init_config(self, **kwargs):
        """Initialize algorithm configuration"""
        self._config.update(kwargs)
    
    @abstractmethod
    def policy(self, state: Dict[str, Any], sampled_params: List[float], 
               episode: int = 0, log_dir: Optional[str] = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """Generate policy action from state"""
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
            "algorithm_type": type(self).__name__
        }
    
    def save_log(self, path: str):
        """Save log data to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.get_log_data(), f, indent=2, default=str)
    
    # Renderable interface implementation
    def render(self, ax=None, **kwargs):
        """Render the algorithm state"""
        if not self._render_flag:
            return
        
        # Default rendering implementation
        if ax is not None:
            self._render_to_axis(ax, **kwargs)
    
    def _render_to_axis(self, ax, **kwargs):
        """Render to matplotlib axis"""
        # Override in subclasses for specific rendering
        pass
    
    def set_render_flag(self, flag: bool):
        """Set rendering flag"""
        self._render_flag = flag
    
    def add_render_component(self, name: str, component: Any):
        """Add a renderable component"""
        self._render_components[name] = component
    
    # Additional utility methods
    def update_params(self, **params):
        """Update algorithm parameters"""
        self._config.update(params)
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a specific parameter"""
        return self._config.get(key, default)
    
    def set_param(self, key: str, value: Any):
        """Set a specific parameter"""
        self._config[key] = value
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate input state"""
        # Override in subclasses for specific validation
        return True
    
    def preprocess_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess state before policy computation"""
        # Override in subclasses for specific preprocessing
        return state
    
    def postprocess_action(self, action: tf.Tensor) -> tf.Tensor:
        """Postprocess action after policy computation"""
        # Override in subclasses for specific postprocessing
        return action