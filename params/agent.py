from pydantic import BaseModel
from typing import List, Optional, Literal

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