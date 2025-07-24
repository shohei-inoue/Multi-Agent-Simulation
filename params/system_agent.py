"""
SystemAgent用のパラメータ定義
システムエージェント（監視役）の設定を管理
"""

from dataclasses import dataclass, field
from typing import Optional
from params.learning import LearningParameter


@dataclass
class SystemAgentDebugParam:
    enable_debug_log: bool = False
    log_system_events: bool = True
    log_swarm_management: bool = True
    log_learning_events: bool = True
    log_branch_events: bool = True
    log_integration_events: bool = True


@dataclass
class BranchConditionParam:
    branch_enabled: bool = True
    branch_threshold: float = 0.3
    min_directions: int = 2
    min_followers_for_branch: int = 3
    branch_learning_inheritance: bool = True
    branch_leader_selection_method: str = "highest_score"
    branch_follower_selection_method: str = "random"
    swarm_creation_cooldown: float = 5.0
    next_swarm_id: int = 2
    branch_algorithm: str = "mobility_based"  # "random", "mobility_based"


@dataclass
class IntegrationConditionParam:
    integration_enabled: bool = True
    integration_threshold: float = 0.2  # 統合閾値を下げて、より低いmobility_scoreで統合
    min_swarms_for_integration: int = 2
    integration_learning_merge: bool = True
    integration_target_selection: str = "nearest"
    integration_learning_merge_method: str = "weighted_average"
    swarm_merge_cooldown: float = 15.0  # クールダウンを15秒に延長
    integration_algorithm: str = "nearest"  # "nearest", "largest", "performance_based"


@dataclass
class SystemAgentParam:
    """SystemAgent用のパラメータ"""
    
    # 監視・制御パラメータ
    update_interval: float = 1.0  # 監視更新間隔
    monitoring_enabled: bool = True  # 監視機能の有効化
    
    # 学習パラメータ
    learning_type: str = "actor_critic"  # "actor_critic", "ppo", "sac"
    learning_rate: float = 0.001
    gamma: float = 0.99
    n_steps: int = 10
    max_steps_per_episode: int = 100
    
    # 群管理パラメータ
    max_swarms: int = 10  # 最大群数
    min_swarms: int = 1  # 最小群数
    
    # 分岐条件のパラメータ（SwarmAgentから移動）
    branch_threshold: float = 0.3  # 分岐閾値（mobility_score）
    min_directions: int = 2  # 最小方向数
    min_followers_for_branch: int = 3  # 分岐に必要な最小フォロワー数
    
    # 統合条件のパラメータ（SwarmAgentから移動）
    integration_threshold: float = 0.7  # 統合閾値（mobility_score）
    min_swarms_for_integration: int = 2  # 統合に必要な最小群数
    
    # 分岐パラメータ
    branch_enabled: bool = True  # 分岐機能の有効化
    branch_learning_inheritance: bool = True  # 学習情報の引き継ぎ
    branch_leader_selection_method: str = "highest_score"  # "highest_score", "random", "round_robin"
    branch_follower_selection_method: str = "random"  # "random", "nearest", "highest_mobility"
    
    # 統合パラメータ
    integration_enabled: bool = True  # 統合機能の有効化
    integration_learning_merge: bool = True  # 学習情報の統合
    integration_target_selection: str = "nearest"  # "nearest", "largest", "highest_performance"
    integration_learning_merge_method: str = "weighted_average"  # "weighted_average", "best", "simple_average"
    
    # 性能監視パラメータ
    performance_monitoring_enabled: bool = True
    performance_threshold: float = 0.5  # 低性能群の閾値
    performance_evaluation_interval: float = 10.0  # 性能評価間隔
    
    # スレッド管理パラメータ
    thread_timeout: float = 1.0  # スレッドタイムアウト
    max_queue_size: int = 100  # 最大キューサイズ
    
    # 報酬設計パラメータ
    reward_weights: Optional[dict] = None  # 報酬の重み
    
    # デバッグ・ログ設定
    enable_debug_log: bool = False
    log_system_events: bool = True
    log_swarm_management: bool = True
    log_learning_events: bool = True
    log_branch_events: bool = True  # 分岐イベントのログ
    log_integration_events: bool = True  # 統合イベントのログ
    
    # 条件・デバッグをまとめたクラス
    branch_condition: BranchConditionParam = field(default_factory=BranchConditionParam)
    integration_condition: IntegrationConditionParam = field(default_factory=IntegrationConditionParam)
    debug: SystemAgentDebugParam = field(default_factory=SystemAgentDebugParam)
    
    # 学習パラメータ
    learningParameter: Optional[LearningParameter] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.reward_weights is None:
            self.reward_weights = {
                'exploration_efficiency': 10.0,
                'swarm_count_balance': 2.0,
                'mobility_score': 5.0,
                'learning_transfer_success': 3.0,
                'system_stability': 2.0,
                'collision_penalty': -5.0,
                'energy_efficiency': 1.0,
                'branch_success': 2.0,  # 分岐成功の報酬
                'integration_success': 1.5  # 統合成功の報酬
            }
        if self.learningParameter is None:
            self.learningParameter = LearningParameter(
        type="a2c",
        model="actor-critic",
        optimizer="adam",
        gamma=0.99,
        learningLate=0.001,
        nStep=10,
                inherit_learning_info=True,
                merge_learning_info=True
            )
    
    def validate(self) -> bool:
        """パラメータの妥当性を検証"""
        if self.update_interval <= 0.0:
            return False
        if self.learning_rate <= 0.0:
            return False
        if self.gamma < 0.0 or self.gamma > 1.0:
            return False
        if self.max_swarms < self.min_swarms:
            return False
        if self.max_swarms < 1:
            return False
        if self.min_swarms < 1:
            return False
        if self.thread_timeout <= 0.0:
            return False
        if self.max_queue_size < 1:
            return False
        if self.performance_threshold < 0.0 or self.performance_threshold > 1.0:
            return False
        if self.branch_threshold < 0.0 or self.branch_threshold > 1.0:
            return False
        if self.integration_threshold < 0.0 or self.integration_threshold > 1.0:
            return False
        if self.min_directions < 1:
            return False
        if self.min_followers_for_branch < 1:
            return False
        if self.min_swarms_for_integration < 2:
            return False
        return True
    
    def to_dict(self) -> dict:
        """パラメータを辞書形式で取得"""
        return {
            'update_interval': self.update_interval,
            'monitoring_enabled': self.monitoring_enabled,
            'learning_type': self.learning_type,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'n_steps': self.n_steps,
            'max_steps_per_episode': self.max_steps_per_episode,
            'max_swarms': self.max_swarms,
            'min_swarms': self.min_swarms,
            'branch_threshold': self.branch_threshold,
            'min_directions': self.min_directions,
            'min_followers_for_branch': self.min_followers_for_branch,
            'integration_threshold': self.integration_threshold,
            'min_swarms_for_integration': self.min_swarms_for_integration,
            'branch_enabled': self.branch_enabled,
            'branch_learning_inheritance': self.branch_learning_inheritance,
            'branch_leader_selection_method': self.branch_leader_selection_method,
            'branch_follower_selection_method': self.branch_follower_selection_method,
            'integration_enabled': self.integration_enabled,
            'integration_learning_merge': self.integration_learning_merge,
            'integration_target_selection': self.integration_target_selection,
            'integration_learning_merge_method': self.integration_learning_merge_method,
            'performance_monitoring_enabled': self.performance_monitoring_enabled,
            'performance_threshold': self.performance_threshold,
            'performance_evaluation_interval': self.performance_evaluation_interval,
            'thread_timeout': self.thread_timeout,
            'max_queue_size': self.max_queue_size,
            'reward_weights': self.reward_weights,
            'enable_debug_log': self.enable_debug_log,
            'log_system_events': self.log_system_events,
            'log_swarm_management': self.log_swarm_management,
            'log_learning_events': self.log_learning_events,
            'log_branch_events': self.log_branch_events,
            'log_integration_events': self.log_integration_events
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SystemAgentParam':
        """辞書からパラメータを作成"""
        return cls(**data)
    
    def copy(self) -> 'SystemAgentParam':
        """パラメータのコピーを作成"""
        return SystemAgentParam(
            update_interval=self.update_interval,
            monitoring_enabled=self.monitoring_enabled,
            learning_type=self.learning_type,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            max_steps_per_episode=self.max_steps_per_episode,
            max_swarms=self.max_swarms,
            min_swarms=self.min_swarms,
            branch_threshold=self.branch_threshold,
            min_directions=self.min_directions,
            min_followers_for_branch=self.min_followers_for_branch,
            integration_threshold=self.integration_threshold,
            min_swarms_for_integration=self.min_swarms_for_integration,
            branch_enabled=self.branch_enabled,
            branch_learning_inheritance=self.branch_learning_inheritance,
            branch_leader_selection_method=self.branch_leader_selection_method,
            branch_follower_selection_method=self.branch_follower_selection_method,
            integration_enabled=self.integration_enabled,
            integration_learning_merge=self.integration_learning_merge,
            integration_target_selection=self.integration_target_selection,
            integration_learning_merge_method=self.integration_learning_merge_method,
            performance_monitoring_enabled=self.performance_monitoring_enabled,
            performance_threshold=self.performance_threshold,
            performance_evaluation_interval=self.performance_evaluation_interval,
            thread_timeout=self.thread_timeout,
            max_queue_size=self.max_queue_size,
            reward_weights=self.reward_weights.copy() if self.reward_weights else None,
            enable_debug_log=self.enable_debug_log,
            log_system_events=self.log_system_events,
            log_swarm_management=self.log_swarm_management,
            log_learning_events=self.log_learning_events,
            log_branch_events=self.log_branch_events,
            log_integration_events=self.log_integration_events,
            branch_condition=self.branch_condition,
            integration_condition=self.integration_condition,
            debug=self.debug,
            learningParameter=self.learningParameter
        )
    
    def get_reward_weight(self, key: str) -> float:
        """報酬の重みを取得"""
        if self.reward_weights is None:
            return 1.0
        return self.reward_weights.get(key, 1.0)
    
    def set_reward_weight(self, key: str, weight: float):
        """報酬の重みを設定"""
        if self.reward_weights is None:
            self.reward_weights = {}
        self.reward_weights[key] = weight
    
    def get_next_swarm_id(self) -> int:
        """次の群IDを取得"""
        return self.branch_condition.next_swarm_id
    
    def increment_swarm_id(self):
        """群IDをインクリメント（分岐時に使用）"""
        self.branch_condition.next_swarm_id += 1
    
    def check_branch_condition(self, direction_count: int, mobility_score: float, follower_count: int) -> bool:
        """分岐条件をチェック"""
        return (self.branch_enabled and
                direction_count >= self.min_directions and
                mobility_score >= self.branch_threshold and
                follower_count >= self.min_followers_for_branch)
    
    def check_integration_condition(self, mobility_score: float, swarm_count: int) -> bool:
        """統合条件をチェック"""
        return (self.integration_enabled and
                mobility_score >= self.integration_threshold and
                swarm_count >= self.min_swarms_for_integration)



