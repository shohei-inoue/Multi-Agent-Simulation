from pydantic import BaseModel
from typing import Dict, Any
import json

class Boundary(BaseModel):
    inner: float = 0.0
    outer: float = 10.0

class MV(BaseModel):
    mean     : float = 0.0
    variance : float = 10.0

class InitialCoordinate(BaseModel):
    x: float = 10.0
    y: float = 10.0

class ExploreParam(BaseModel):
    boundary   : Boundary          = Boundary()
    mv         : MV                = MV()
    coordinate : InitialCoordinate = InitialCoordinate()
    robotNum   : int               = 10
    finishRate : float             = 0.8
    finishStep : int               = 40
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "boundary": self.boundary.dict(),
            "mv": self.mv.dict(),
            "coordinate": self.coordinate.dict(),
            "robotNum": self.robotNum,
            "finishRate": self.finishRate,
            "finishStep": self.finishStep
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "boundary" in config:
            self.boundary = Boundary(**config["boundary"])
        if "mv" in config:
            self.mv = MV(**config["mv"])
        if "coordinate" in config:
            self.coordinate = InitialCoordinate(**config["coordinate"])
        if "robotNum" in config:
            self.robotNum = config["robotNum"]
        if "finishRate" in config:
            self.finishRate = config["finishRate"]
        if "finishStep" in config:
            self.finishStep = config["finishStep"]
    
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