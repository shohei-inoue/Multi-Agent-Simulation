from typing import List, Optional, Literal, Dict, Any
import json
from dataclasses import dataclass, field, asdict
from params.system_agent import SystemAgentParam
from params.swarm_agent import SwarmAgentParam
from params.learning import LearningParameter

@dataclass
class ContinuousRange:
    min: float
    max: float


@dataclass
class ActionSpace:
    name: str
    type: Literal["continuous", "discrete"]
    range: Optional[ContinuousRange]
    values: Optional[List[float]]


@dataclass
class ObservationSpace:
    name: str
    isSet: bool
    type: Literal["vector", "scalar"]
    low: float
    high: float
    initialize: float

@dataclass
class Reward:
    name: str
    isSet: bool
    value: float


@dataclass
class AgentParam:
    episodeNum: int = 1000
    maxStepsPerEpisode: int = 100
    system_agent_param: Optional[SystemAgentParam] = None
    swarm_agent_params: List[SwarmAgentParam] = field(default_factory=list)

    def __post_init__(self):
        if self.system_agent_param is None:
            self.system_agent_param = SystemAgentParam()
        if self.swarm_agent_params is None:
            self.swarm_agent_params = []

    def to_dict(self):
        return {
            "episodeNum": self.episodeNum,
            "maxStepsPerEpisode": self.maxStepsPerEpisode,
            "system_agent_param": self.system_agent_param,
            "swarm_agent_params": [s.__dict__ for s in self.swarm_agent_params]
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            episodeNum=data.get("episodeNum", 100),
            maxStepsPerEpisode=data.get("maxStepsPerEpisode", 50),
            system_agent_param=SystemAgentParam(**data["system_agent_param"]) if "system_agent_param" in data else None,
            swarm_agent_params=[SwarmAgentParam(**s) for s in data.get("swarm_agent_params", [])]
        )

    def copy(self):
        return AgentParam(
            episodeNum=self.episodeNum,
            maxStepsPerEpisode=self.maxStepsPerEpisode,
            system_agent_param=self.system_agent_param,
            swarm_agent_params=[s for s in self.swarm_agent_params]
        )

    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        config = {
            "episodeNum": self.episodeNum,
            "maxStepsPerEpisode": self.maxStepsPerEpisode,
            "system_agent_param": self.system_agent_param,
            "swarm_agent_params": [s.to_dict() for s in self.swarm_agent_params]
        }
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "episodeNum" in config:
            self.episodeNum = config["episodeNum"]
        if "maxStepsPerEpisode" in config:
            self.maxStepsPerEpisode = config["maxStepsPerEpisode"]
        if "system_agent_param" in config:
            self.system_agent_param = SystemAgentParam(**config["system_agent_param"])
        if "swarm_agent_params" in config:
            self.swarm_agent_params = [SwarmAgentParam(**s) for s in config["swarm_agent_params"]]
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.get_config()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self.set_config(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self.episodeNum = 100
        self.maxStepsPerEpisode = 50
        self.system_agent_param = SystemAgentParam()
        self.swarm_agent_params = []
    
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


# パラメータの例
# {
#   "episodeNum": 100,
#   "maxStepsPerEpisode": 50,
#   "system_agent_param": {
#     "agent_id": "system_agent",
#     "monitoring_enabled": true,
#     "branch_condition": {
#       "branch_enabled": true,
#       "branch_threshold": 0.3,
#       "min_directions": 2,
#       "min_followers_for_branch": 3,
#       "branch_learning_inheritance": true,
#       "branch_leader_selection_method": "highest_score",
#       "branch_follower_selection_method": "random",
#       "swarm_creation_cooldown": 5.0,
#       "next_swarm_id": 2,
#       "branch_algorithm": "random"
#     },
#     "integration_condition": {
#       "integration_enabled": true,
#       "integration_threshold": 0.7,
#       "min_swarms_for_integration": 2,
#       "integration_learning_merge": true,
#       "integration_target_selection": "nearest",
#       "integration_learning_merge_method": "weighted_average",
#       "swarm_merge_cooldown": 3.0,
#       "integration_algorithm": "nearest"
#     },
#     "debug": {
#       "enable_debug_log": false,
#       "log_system_events": true,
#       "log_swarm_management": true,
#       "log_learning_events": true,
#       "log_branch_events": true,
#       "log_integration_events": true
#     },
#     "performance_monitoring_enabled": true,
#     "performance_threshold": 0.5,
#     "performance_evaluation_interval": 10.0,
#     "thread_timeout": 1.0,
#     "max_queue_size": 100,
#     "reward_weights": {
#       "exploration_efficiency": 10.0,
#       "swarm_count_balance": 2.0,
#       "mobility_score": 5.0,
#       "learning_transfer_success": 3.0,
#       "system_stability": 2.0,
#       "collision_penalty": -5.0,
#       "energy_efficiency": 1.0,
#       "branch_success": 2.0,
#       "integration_success": 1.5
#     },
#     "learningParameter": {
#       "type": "system_agent",
#       "model": "actor-critic",
#       "optimizer": "adam",
#       "gamma": 0.99,
#       "learningLate": 0.001,
#       "nStep": 10,
#       "inherit_learning_info": true,
#       "merge_learning_info": true
#     }
#   },
#   "swarm_agent_params": [
#     {
#       "algorithm": "vfh_fuzzy",
#       "swarm_id": 1,
#       "isLearning": true,
#       "learningParameter": {
#         "type": "swarm_agent",
#         "model": "actor-critic",
#         "optimizer": "adam",
#         "gamma": 0.99,
#         "learningLate": 0.001,
#         "nStep": 5,
#         "inherit_learning_info": true,
#         "merge_learning_info": true
#       },
#       "debug": {
#         "enable_debug_log": false,
#         "log_movement_events": true,
#         "log_learning_events": true
#       }
#     }
#   ]
# }