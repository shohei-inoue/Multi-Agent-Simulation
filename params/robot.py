from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import json


class BoidsType(Enum):
    """
    boids判断用のタイプ
    """
    NONE  = 0
    OUTER = 1
    INNER = 2


@dataclass
class Movement:
    min: float = 2.0
    max: float = 3.0


@dataclass
class Boids:
    min: float = 2.0
    max: float = 3.0


@dataclass
class Avoidance:
    min: float = 90.0
    max: float = 270.0


@dataclass
class Offset:
    position              : float     = 5.0
    step                  : int       = 0
    amount_of_movement    : float     = 0.0
    direction_angle       : float     = 0.0
    collision_flag        : bool      = False
    boids_flag            : "BoidsType" = BoidsType.NONE
    estimated_probability : float     = 0.0
    one_explore_step      : int       = 60  # 1探査エリアあたりの探索ステップ数

    def get_boids_flag_value(self):
        return self.boids_flag.value if isinstance(self.boids_flag, BoidsType) else 0


@dataclass
class RobotParam:
    movement: Optional[Movement] = None
    boids: Optional[Boids] = None
    avoidance: Optional[Avoidance] = None
    offset: Optional[Offset] = None

    def __post_init__(self):
        if self.movement is None:
            self.movement = Movement()
        if self.boids is None:
            self.boids = Boids()
        if self.avoidance is None:
            self.avoidance = Avoidance()
        if self.offset is None:
            self.offset = Offset()

    def to_dict(self):
        offset_dict = {k: v for k, v in (self.offset.__dict__ if self.offset else {}).items() if k != "boids_flag"}
        offset_dict["boids_flag"] = self.offset.get_boids_flag_value() if self.offset else 0
        return {
            "movement": self.movement.__dict__ if self.movement else {},
            "boids": self.boids.__dict__ if self.boids else {},
            "avoidance": self.avoidance.__dict__ if self.avoidance else {},
            "offset": offset_dict
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            movement=Movement(**data["movement"]) if "movement" in data else None,
            boids=Boids(**data["boids"]) if "boids" in data else None,
            avoidance=Avoidance(**data["avoidance"]) if "avoidance" in data else None,
            offset=Offset(
                **{k: v for k, v in data["offset"].items() if k != "boids_flag"},
                boids_flag=BoidsType(data["offset"].get("boids_flag", 0))
            ) if "offset" in data else None
        )

    def copy(self):
        return RobotParam(
            movement=Movement(**self.movement.__dict__),
            boids=Boids(**self.boids.__dict__),
            avoidance=Avoidance(**self.avoidance.__dict__),
            offset=Offset(**self.offset.__dict__)
        )

    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.to_dict()
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "movement" in config:
            self.movement = Movement(**config["movement"])
        if "boids" in config:
            self.boids = Boids(**config["boids"])
        if "avoidance" in config:
            self.avoidance = Avoidance(**config["avoidance"])
        if "offset" in config:
            self.offset = Offset(**config["offset"])
    
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