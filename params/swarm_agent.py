"""
SwarmAgent用のパラメータ設定
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LearningParameter:
    """学習パラメータ"""
    type: str = "swarm_agent"
    model: str = "actor-critic"
    optimizer: str = "adam"
    gamma: float = 0.99
    learningLate: float = 0.001
    nStep: int = 5
    inherit_learning_info: bool = True
    merge_learning_info: bool = True


@dataclass
class SwarmDebugParam:
    """デバッグパラメータ"""
    enable_debug_log: bool = False
    log_movement_events: bool = True
    log_learning_events: bool = True


@dataclass
class SwarmAgentParam:
    """SwarmAgent用のパラメータ"""
    
    # 基本設定
    algorithm: str = "vfh_fuzzy"
    swarm_id: str = "swarm_default"
    
    # 学習設定（元の設計を尊重）
    isLearning: bool = True
    learningParameter: Optional[LearningParameter] = None
    
    # デバッグ設定
    debug: Optional[SwarmDebugParam] = None
    
    # VFH-Fuzzyパラメータ（学習可能）
    th: float = 0.5        # 閾値
    k_e: float = 10.0      # 探査向上性重み
    k_c: float = 5.0       # 衝突回避重み
    
    # 移動パラメータ
    movement_min: float = 2.0
    movement_max: float = 3.0
    boids_min: float = 2.0
    boids_max: float = 3.0
    avoidance_min: float = 2.0
    avoidance_max: float = 3.0
    
    # 群制御パラメータ
    ideal_distance: float = 5.0
    exploration_radius: float = 10.0
    max_followers: int = 20
    
    def __post_init__(self):
        if self.learningParameter is None:
            self.learningParameter = LearningParameter()
        if self.debug is None:
            self.debug = SwarmDebugParam()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返す"""
        return {
            "algorithm": self.algorithm,
            "swarm_id": self.swarm_id,
            "isLearning": self.isLearning,
            "learningParameter": self.learningParameter.__dict__ if self.learningParameter else None,
            "debug": self.debug.__dict__ if self.debug else None,
            "th": self.th,
            "k_e": self.k_e,
            "k_c": self.k_c,
            "movement_min": self.movement_min,
            "movement_max": self.movement_max,
            "boids_min": self.boids_min,
            "boids_max": self.boids_max,
            "avoidance_min": self.avoidance_min,
            "avoidance_max": self.avoidance_max,
            "ideal_distance": self.ideal_distance,
            "exploration_radius": self.exploration_radius,
            "max_followers": self.max_followers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmAgentParam':
        """辞書から作成"""
        # ネストしたオブジェクトを適切に処理
        learning_param_data = data.get('learningParameter')
        learning_param = LearningParameter(**learning_param_data) if learning_param_data else None
        
        debug_param_data = data.get('debug')
        debug_param = SwarmDebugParam(**debug_param_data) if debug_param_data else None
        
        # 基本データをコピー
        base_data = {k: v for k, v in data.items() 
                    if k not in ['learningParameter', 'debug']}
        
        return cls(
            learningParameter=learning_param,
            debug=debug_param,
            **base_data
        )
    
    def copy(self) -> 'SwarmAgentParam':
        """コピーを作成"""
        return SwarmAgentParam(
            algorithm=self.algorithm,
            swarm_id=self.swarm_id,
            isLearning=self.isLearning,
            learningParameter=self.learningParameter.copy() if self.learningParameter else None,
            debug=self.debug.copy() if self.debug else None,
            th=self.th,
            k_e=self.k_e,
            k_c=self.k_c,
            movement_min=self.movement_min,
            movement_max=self.movement_max,
            boids_min=self.boids_min,
            boids_max=self.boids_max,
            avoidance_min=self.avoidance_min,
            avoidance_max=self.avoidance_max,
            ideal_distance=self.ideal_distance,
            exploration_radius=self.exploration_radius,
            max_followers=self.max_followers
        )
