from typing import Dict, Any
import json
from dataclasses import dataclass, asdict, field

@dataclass
class Boundary:
    inner: float = 0.0
    outer: float = 10.0

@dataclass
class MV:
    mean: float = 0.0
    variance: float = 10.0

@dataclass
class InitialCoordinate:
    x: float = 10.0
    y: float = 10.0

@dataclass
class ExploreParam:
    boundary: Boundary = field(default_factory=Boundary)
    mv: MV = field(default_factory=MV)
    coordinate: InitialCoordinate = field(default_factory=InitialCoordinate)
    robotNum: int = 10
    finishRate: float = 0.8
    finishStep: int = 40
    initialSwarmNum: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            boundary=Boundary(**data["boundary"]) if "boundary" in data else Boundary(),
            mv=MV(**data["mv"]) if "mv" in data else MV(),
            coordinate=InitialCoordinate(**data["coordinate"]) if "coordinate" in data else InitialCoordinate(),
            robotNum=data.get("robotNum", 10),
            finishRate=data.get("finishRate", 0.8),
            finishStep=data.get("finishStep", 40),
            initialSwarmNum=data.get("initialSwarmNum", 1)
        )

    def copy(self):
        return ExploreParam(
            boundary=Boundary(**self.boundary.__dict__),
            mv=MV(**self.mv.__dict__),
            coordinate=InitialCoordinate(**self.coordinate.__dict__),
            robotNum=self.robotNum,
            finishRate=self.finishRate,
            finishStep=self.finishStep,
            initialSwarmNum=self.initialSwarmNum
        )

    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.to_dict()
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        self.boundary = Boundary(**config["boundary"])
        self.mv = MV(**config["mv"])
        self.coordinate = InitialCoordinate(**config["coordinate"])
        self.robotNum = config["robotNum"]
        self.finishRate = config["finishRate"]
        self.finishStep = config["finishStep"]
        self.initialSwarmNum = config["initialSwarmNum"]
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.get_config()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self.set_config(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self.boundary = Boundary()
        self.mv = MV()
        self.coordinate = InitialCoordinate()
        self.robotNum = 10
        self.finishRate = 0.8
        self.finishStep = 40
        self.initialSwarmNum = 1
    
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