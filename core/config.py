"""
Core configuration management for the project.
Centralizes all configuration and parameter management.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    simulation_id: Optional[str] = None
    log_dir: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.simulation_id is None:
            import uuid
            self.simulation_id = f"sim_{self.timestamp}_{uuid.uuid4().hex[:6]}"
        
        if self.log_dir is None:
            self.log_dir = f"./logs/{self.simulation_id}"


@dataclass
class SystemConfig:
    """System-wide configuration"""
    enable_gpu: bool = True
    enable_visualization: bool = False
    enable_gif_save: bool = True
    max_workers: int = 4
    debug_mode: bool = False
    
    def __post_init__(self):
        # GPU設定の自動検出
        if self.enable_gpu:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("⚠️  GPU not detected, falling back to CPU")
                self.enable_gpu = False


class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self):
        self.simulation = SimulationConfig()
        self.system = SystemConfig()
        self._params = {}
    
    def set_param(self, key: str, value: Any):
        """Set a parameter"""
        self._params[key] = value
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter"""
        return self._params.get(key, default)
    
    def create_log_directories(self):
        """Create necessary log directories"""
        log_dir = Path(self.simulation.log_dir)
        subdirs = ["metrics", "gifs", "models", "tensorboard", "csvs"]
        
        for subdir in subdirs:
            (log_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return log_dir
    
    def get_log_path(self, subdir: str, filename: str) -> Path:
        """Get log file path"""
        return Path(self.simulation.log_dir) / subdir / filename


# Global configuration instance
config = ConfigManager() 