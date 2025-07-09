from enum import Enum
from typing import Dict, Any
import json


class Movement:
    min: float = 2.0
    max: float = 3.0


class Boids:
    min: float = 2.0
    max: float = 3.0

class Avoidance:
    min: float = 90.0
    max: float = 270.0


class BoidsType(Enum):
    """
    boids判断用のタイプ
    """
    NONE  = 0
    OUTER = 1
    INNER = 2


class Offset:
    position              : float     = 5.0
    step                  : int       = 0
    amount_of_movement    : float     = 0.0
    direction_angle       : float     = 0.0
    collision_flag        : bool      = False
    boids_flag            : BoidsType = BoidsType.NONE
    estimated_probability : float     = 0.0
    one_explore_step      : int       = 60


class RobotParam:
    movement  : Movement  = Movement()
    boids     : Boids     = Boids()
    avoidance : Avoidance = Avoidance()
    offset    : Offset    = Offset()
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "movement": {
                "min": self.movement.min,
                "max": self.movement.max
            },
            "boids": {
                "min": self.boids.min,
                "max": self.boids.max
            },
            "avoidance": {
                "min": self.avoidance.min,
                "max": self.avoidance.max
            },
            "offset": {
                "position": self.offset.position,
                "step": self.offset.step,
                "amount_of_movement": self.offset.amount_of_movement,
                "direction_angle": self.offset.direction_angle,
                "collision_flag": self.offset.collision_flag,
                "boids_flag": self.offset.boids_flag.value,
                "estimated_probability": self.offset.estimated_probability,
                "one_explore_step": self.offset.one_explore_step
            }
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "movement" in config:
            self.movement.min = config["movement"].get("min", 2.0)
            self.movement.max = config["movement"].get("max", 3.0)
        if "boids" in config:
            self.boids.min = config["boids"].get("min", 2.0)
            self.boids.max = config["boids"].get("max", 3.0)
        if "avoidance" in config:
            self.avoidance.min = config["avoidance"].get("min", 90.0)
            self.avoidance.max = config["avoidance"].get("max", 270.0)
        if "offset" in config:
            offset_config = config["offset"]
            self.offset.position = offset_config.get("position", 5.0)
            self.offset.step = offset_config.get("step", 0)
            self.offset.amount_of_movement = offset_config.get("amount_of_movement", 0.0)
            self.offset.direction_angle = offset_config.get("direction_angle", 0.0)
            self.offset.collision_flag = offset_config.get("collision_flag", False)
            self.offset.boids_flag = BoidsType(offset_config.get("boids_flag", 0))
            self.offset.estimated_probability = offset_config.get("estimated_probability", 0.0)
            self.offset.one_explore_step = offset_config.get("one_explore_step", 60)
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.get_config()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self.set_config(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self.movement = Movement()
        self.boids = Boids()
        self.avoidance = Avoidance()
        self.offset = Offset()
    
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