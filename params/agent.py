from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import json


class ContinuousRange(BaseModel):
    min: float
    max: float


class ActionSpace(BaseModel):
    name: str
    type: Literal["continuous", "discrete"]
    range: Optional[ContinuousRange]
    values: Optional[List[float]]


class ObservationSpace(BaseModel):
    name: str
    isSet: bool
    type: Literal["vector", "scalar"]
    low: float
    high: float
    initialize: float


class Reward(BaseModel):
    name: str
    isSet: bool
    value: float


class LearningParameter(BaseModel):
    type: Literal["a2c"]
    model: Literal["actor-critic"]
    optimizer: Literal["adam"]
    gamma: float
    learningLate: float
    episodeNum: int
    nStep: int
    # actionSpace: List[ActionSpace]
    # observationSpace: List[ObservationSpace]
    # reward: List[Reward]


class AgentParam(BaseModel):
    algorithm:                     Literal["vfh_fuzzy"] = "vfh_fuzzy"
    maxStepsPerEpisode:            int = 40
    isLearning:                    bool = True
    learningParameter:             Optional[LearningParameter] = LearningParameter(
        type="a2c",
        model="actor-critic",
        optimizer="adam",
        gamma=0.99,
        learningLate=0.001,
        episodeNum=50,
        nStep=5,
    )
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        config = {
            "algorithm": self.algorithm,
            "maxStepsPerEpisode": self.maxStepsPerEpisode,
            "isLearning": self.isLearning
        }
        if self.learningParameter:
            config["learningParameter"] = self.learningParameter.dict()
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        if "algorithm" in config:
            self.algorithm = config["algorithm"]
        if "maxStepsPerEpisode" in config:
            self.maxStepsPerEpisode = config["maxStepsPerEpisode"]
        if "isLearning" in config:
            self.isLearning = config["isLearning"]
        if "learningParameter" in config:
            self.learningParameter = LearningParameter(**config["learningParameter"])
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.get_config()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self.set_config(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self.algorithm = "vfh_fuzzy"
        self.maxStepsPerEpisode = 40
        self.isLearning = True
        self.learningParameter = LearningParameter(
            type="a2c",
            model="actor-critic",
            optimizer="adam",
            gamma=0.99,
            learningLate=0.001,
            episodeNum=50,
            nStep=5,
        )
    
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