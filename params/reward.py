from dataclasses import dataclass, asdict, field

@dataclass
class SwarmRewardConfig:
    exploration_base_reward: float = 10.0  # 探索率向上の基本報酬
    exploration_improvement_weight: float = 50.0  # 探索率上昇量の重み（新しい領域を探索したときの報酬の大きさ）
    unexplored_area_bonus: float = 3.0  # 未探査領域への接近ボーナス
    swarm_dispersion_bonus: float = 2.0  # 群ロボットの分散度（協調行動）に対するボーナス
    movement_efficiency_bonus: float = 1.0  # 効率的な移動（距離効率）に対するボーナス
    revisit_penalty_weight: float = 5.0  # 再訪問ペナルティの重み
    collision_penalty_weight: float = 15.0  # 衝突ペナルティの重み
    completion_bonus: float = 100.0  # 全探索完了時のボーナス
    time_efficiency_bonus: float = 20.0  # 早期完了（時間効率）ボーナス

    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    def copy(self):
        return SwarmRewardConfig(**asdict(self))

@dataclass
class SystemRewardConfig:
    swarm_count_balance: float = 2.0  # 群数バランス（群が適切な数で維持されているか）に対する報酬
    branch_success: float = 2.0  # 分岐成功時の報酬
    integration_success: float = 1.5  # 統合成功時の報酬
    system_stability: float = 2.0  # システム全体の安定性に対する報酬
    exploration_efficiency: float = 10.0  # 全体の探索効率に対する報酬
    low_performance_penalty: float = -5.0  # 低性能群（例：探索効率が悪い群）へのペナルティ
    # 必要に応じて追加
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    def copy(self):
        return SystemRewardConfig(**asdict(self))

@dataclass
class RewardConfig:
    default: float = -1.0  # デフォルト報酬（例：時間経過ごとに与えるペナルティなど）
    revisit_penalty: float = -2.0  # 既に訪れた場所を再訪した場合のペナルティ
    collision_penalty: float = -10.0  # ロボット同士や障害物との衝突時のペナルティ
    clear_target_rate: float = 50.0  # 目標（例：全探索完了）を達成した場合のボーナス
    none_finish_penalty: float = -50.0  # 目標未達成で終了した場合のペナルティ
    swarm: SwarmRewardConfig = field(default_factory=SwarmRewardConfig)  # 群ロボット用報酬設定
    system: SystemRewardConfig = field(default_factory=SystemRewardConfig)  # システム全体用報酬設定

    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data):
        return cls(
            default=data.get("default", -1.0),
            revisit_penalty=data.get("revisit_penalty", -2.0),
            collision_penalty=data.get("collision_penalty", -10.0),
            clear_target_rate=data.get("clear_target_rate", 50.0),
            none_finish_penalty=data.get("none_finish_penalty", -50.0),
            swarm=SwarmRewardConfig.from_dict(data["swarm"]) if "swarm" in data else SwarmRewardConfig(),
            system=SystemRewardConfig.from_dict(data["system"]) if "system" in data else SystemRewardConfig()
        )
    def copy(self):
        return RewardConfig(
            default=self.default,
            revisit_penalty=self.revisit_penalty,
            collision_penalty=self.collision_penalty,
            clear_target_rate=self.clear_target_rate,
            none_finish_penalty=self.none_finish_penalty,
            swarm=self.swarm.copy(),
            system=self.system.copy()
        ) 