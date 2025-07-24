"""
Simulation parameter configuration.
Centralizes all parameter management for the simulation.
"""

from params.environment import EnvironmentParam
from params.robot import RobotParam
from params.explore import ExploreParam
from params.agent import AgentParam
from core.logging import get_component_logger
from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass, asdict, field
from params.reward import RewardConfig
from params.robot_logging import RobotLoggingConfig


@dataclass
class SimulationParam:
    agent: Optional[AgentParam] = None
    environment: Optional[EnvironmentParam] = None
    explore: Optional[ExploreParam] = None
    reward: Optional[RewardConfig] = None
    robot_logging: Optional[RobotLoggingConfig] = None
    robot_params: List[RobotParam] = field(default_factory=list)

    def __post_init__(self):
        if self.agent is None:
            self.agent = AgentParam()
        if self.environment is None:
            self.environment = EnvironmentParam()
        if self.explore is None:
            self.explore = ExploreParam()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.robot_logging is None:
            self.robot_logging = RobotLoggingConfig()
        if self.robot_params is None:
            self.robot_params = []
        
        # 初期SwarmAgentParamを作成
        if self.agent and not self.agent.swarm_agent_params:
            from params.swarm_agent import SwarmAgentParam
            initial_swarm_param = SwarmAgentParam()
            self.agent.swarm_agent_params = [initial_swarm_param]

    def to_dict(self):
        return {
            "agent": self.agent.to_dict() if self.agent else None,
            "environment": self.environment.to_dict() if self.environment else None,
            "explore": self.explore.to_dict() if self.explore else None,
            "reward": self.reward.to_dict() if self.reward else None,
            "robot_logging": self.robot_logging.to_dict() if self.robot_logging else None,
            "robot_params": [r.to_dict() for r in self.robot_params]
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            agent=AgentParam.from_dict(data["agent"]) if "agent" in data else AgentParam(),
            environment=EnvironmentParam.from_dict(data["environment"]) if "environment" in data else EnvironmentParam(),
            explore=ExploreParam.from_dict(data["explore"]) if "explore" in data else ExploreParam(),
            reward=RewardConfig.from_dict(data["reward"]) if "reward" in data else RewardConfig(),
            robot_logging=RobotLoggingConfig.from_dict(data["robot_logging"]) if "robot_logging" in data else RobotLoggingConfig(),
            robot_params=[RobotParam.from_dict(r) for r in data.get("robot_params", [])]
        )

    def copy(self):
        return SimulationParam(
            agent=self.agent.copy() if self.agent else None,
            environment=self.environment.copy() if self.environment else None,
            explore=self.explore.copy() if self.explore else None,
            reward=self.reward.copy() if self.reward else None,
            robot_logging=self.robot_logging.copy() if self.robot_logging else None,
            robot_params=[r.copy() for r in self.robot_params]
        )


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