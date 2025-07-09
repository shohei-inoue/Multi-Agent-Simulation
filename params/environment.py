from pydantic import BaseModel
from typing import Dict, Any
import json

class MapParam(BaseModel):
    width  : int = 150
    height : int = 60
    seed   : int = 42

class ObstacleParam(BaseModel):
    probability : float = 0.005
    maxSize     : float = 10
    value       : int   = 1000

class EnvironmentParam(BaseModel):
    map: MapParam = MapParam()
    obstacle: ObstacleParam = ObstacleParam()
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "map": self.map.dict(),
            "obstacle": self.obstacle.dict()
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "map" in config:
            self.map = MapParam(**config["map"])
        if "obstacle" in config:
            self.obstacle = ObstacleParam(**config["obstacle"])
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.get_config()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self.set_config(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self.map = MapParam()
        self.obstacle = ObstacleParam()
    
    # Loggable interface implementation
    def get_log_data(self) -> Dict[str, Any]:
        """Get data for logging"""
        return {
            "config": self.get_config(),
            "state": self.get_state()
        }
    
    def save_log(self, path: str):
        """Save log data to file"""
        with open(path, 'w') as f:
            json.dump(self.get_log_data(), f, indent=2, default=str)