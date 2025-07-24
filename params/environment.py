from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class MapParam:
    width: int = 200
    height: int = 200
    seed: int = 42

@dataclass
class ObstacleParam:
    probability: float = 0.005
    maxSize: float = 10
    value: int = 1000

@dataclass
class EnvironmentParam:
    map: Optional[MapParam] = None
    obstacle: Optional[ObstacleParam] = None

    def __post_init__(self):
        if self.map is None:
            self.map = MapParam()
        if self.obstacle is None:
            self.obstacle = ObstacleParam()

    def to_dict(self):
        return {
            "map": self.map.__dict__,
            "obstacle": self.obstacle.__dict__
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            map=MapParam(**data["map"]) if "map" in data else None,
            obstacle=ObstacleParam(**data["obstacle"]) if "obstacle" in data else None
        )

    def copy(self):
        return EnvironmentParam(
            map=MapParam(**self.map.__dict__),
            obstacle=ObstacleParam(**self.obstacle.__dict__)
        )

    # Configurable interface implementation
    def get_config(self):
        """Get current configuration"""
        return self.to_dict()
    
    def set_config(self, config):
        """Set configuration"""
        if "map" in config:
            self.map = MapParam(**config["map"])
        if "obstacle" in config:
            self.obstacle = ObstacleParam(**config["obstacle"])
    
    # Stateful interface implementation
    def get_state(self):
        """Get current state"""
        return self.get_config()
    
    def set_state(self, state):
        """Set state"""
        self.set_config(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self.map = MapParam()
        self.obstacle = ObstacleParam()
    
    # Loggable interface implementation
    def get_log_data(self):
        """Get data for logging"""
        return {
            "config": self.get_config(),
            "state": self.get_state()
        }
    
    def save_log(self, path: str):
        """Save log data to file"""
        with open(path, 'w') as f:
            json.dump(self.get_log_data(), f, indent=2, default=str)