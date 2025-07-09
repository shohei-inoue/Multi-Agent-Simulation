"""
Simulation parameter configuration.
Centralizes all parameter management for the simulation.
"""

from params.environment import EnvironmentParam
from params.robot import RobotParam
from params.explore import ExploreParam
from params.agent import AgentParam
from core.logging import get_component_logger
from typing import Dict, Any
import json


class Param:
    environment : EnvironmentParam = EnvironmentParam()
    agent       : AgentParam       = AgentParam()
    robot       : RobotParam       = RobotParam()
    explore     : ExploreParam     = ExploreParam()
    
    def __init__(self):
        self.logger = get_component_logger("param")
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "environment": self.environment.get_config(),
            "agent": self.agent.get_config(),
            "robot": self.robot.get_config(),
            "explore": self.explore.get_config()
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "environment" in config:
            self.environment.set_config(config["environment"])
        if "agent" in config:
            self.agent.set_config(config["agent"])
        if "robot" in config:
            self.robot.set_config(config["robot"])
        if "explore" in config:
            self.explore.set_config(config["explore"])
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            "environment": self.environment.get_state(),
            "agent": self.agent.get_state(),
            "robot": self.robot.get_state(),
            "explore": self.explore.get_state()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        if "environment" in state:
            self.environment.set_state(state["environment"])
        if "agent" in state:
            self.agent.set_state(state["agent"])
        if "robot" in state:
            self.robot.set_state(state["robot"])
        if "explore" in state:
            self.explore.set_state(state["explore"])
    
    def reset_state(self):
        """Reset to initial state"""
        self.environment.reset_state()
        self.agent.reset_state()
        self.robot.reset_state()
        self.explore.reset_state()
    
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