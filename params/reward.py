from pydantic import BaseModel
from typing import Literal

class ExplorationRewardConfig(BaseModel):
    """探査報酬の設定"""
    # 基本報酬
    base_reward: float = 5.0
    
    # 上昇量に応じた倍率
    improvement_multiplier: float = 10.0
    
    # 報酬計算の非線形性（0.5=平方根、1.0=線形、2.0=二乗）
    nonlinearity: float = 0.5
    
    # 最大報酬の上限（無限大を防ぐ）
    max_reward: float = 50.0
    
    # 最小報酬の下限
    min_reward: float = -10.0

class SwarmExplorationRewardConfig(BaseModel):
    """群ロボット探査専用の報酬設定"""
    # 探査率向上の基本報酬
    exploration_base_reward: float = 10.0
    
    # 探査率上昇量の重み（非常に重要）
    exploration_improvement_weight: float = 50.0
    
    # 未探査領域への接近報酬
    unexplored_area_bonus: float = 3.0
    
    # 群ロボットの分散度報酬（協調行動）
    swarm_dispersion_bonus: float = 2.0
    
    # 効率的な移動報酬（距離効率）
    movement_efficiency_bonus: float = 1.0
    
    # 再訪問ペナルティ（探査効率を重視）
    revisit_penalty_weight: float = 5.0
    
    # 衝突ペナルティ（安全性）
    collision_penalty_weight: float = 15.0
    
    # 目標達成報酬（全探索完了）
    completion_bonus: float = 100.0
    
    # 時間効率報酬（早期完了ボーナス）
    time_efficiency_bonus: float = 20.0

class RewardConfig(BaseModel):
    """報酬設定の全体設定"""
    # デフォルト報酬（時間ペナルティ）
    default: float = -1.0
    
    # 探査報酬設定
    exploration: ExplorationRewardConfig = ExplorationRewardConfig()
    
    # 群ロボット探査専用設定
    swarm_exploration: SwarmExplorationRewardConfig = SwarmExplorationRewardConfig()
    
    # 再訪問ペナルティ
    revisit_penalty: float = -2.0
    
    # 衝突ペナルティ
    collision_penalty: float = -10.0
    
    # 目標達成報酬
    clear_target_rate: float = 50.0
    
    # 未達成ペナルティ
    none_finish_penalty: float = -50.0 