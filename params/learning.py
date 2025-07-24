from typing import Literal
from dataclasses import dataclass, asdict


@dataclass
class LearningParameter:
    type: Literal["a2c"]
    model: Literal["actor-critic"]
    optimizer: Literal["adam"]
    gamma: float
    learningLate: float
    nStep: int
    # 学習情報の管理
    inherit_learning_info: bool = True  # 学習情報を引き継ぐか
    merge_learning_info: bool = True  # 学習情報を統合するか

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def copy(self):
        return LearningParameter(**asdict(self)) 