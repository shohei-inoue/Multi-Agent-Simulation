"""
SystemAgent用のパラメータ設定
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LearningParameter:
    """学習パラメータ"""
    type: str = "system_agent"
    model: str = "actor-critic"
    optimizer: str = "adam"
    gamma: float = 0.99
    learningLate: float = 0.001
    nStep: int = 10
    inherit_learning_info: bool = True
    merge_learning_info: bool = True
    updateInterval: float = 1.0


@dataclass
class BranchConditionParam:
    """分岐条件パラメータ"""
    branch_enabled: bool = True
    branch_threshold: float = 0.3
    min_directions: int = 2
    min_followers_for_branch: int = 3
    branch_learning_inheritance: bool = True
    branch_leader_selection_method: str = "highest_score"
    branch_follower_selection_method: str = "random"
    swarm_creation_cooldown: float = 5.0
    next_swarm_id: int = 2
    branch_algorithm: str = "random"


@dataclass
class IntegrationConditionParam:
    """統合条件パラメータ"""
    integration_enabled: bool = True
    integration_threshold: float = 0.7
    min_swarms_for_integration: int = 2
    integration_learning_merge: bool = True
    integration_target_selection: str = "nearest"
    integration_learning_merge_method: str = "weighted_average"
    swarm_merge_cooldown: float = 3.0
    integration_algorithm: str = "nearest"


@dataclass
class SystemAgentDebugParam:
    """デバッグパラメータ"""
    enable_debug_log: bool = False
    log_system_events: bool = True
    log_swarm_management: bool = True
    log_learning_events: bool = True
    log_branch_events: bool = True
    log_integration_events: bool = True


@dataclass
class SystemAgentParam:
    """SystemAgent用のパラメータ"""
    
    # 基本設定
    agent_id: str = "system_agent"
    monitoring_enabled: bool = True
    
    # 学習設定（元の設計を尊重）
    learningParameter: Optional[LearningParameter] = None
    
    # 分岐・統合設定
    branch_condition: Optional[BranchConditionParam] = None
    integration_condition: Optional[IntegrationConditionParam] = None
    
    # デバッグ設定
    debug: Optional[SystemAgentDebugParam] = None
    
    # 性能監視設定
    performance_monitoring_enabled: bool = True
    performance_threshold: float = 0.5
    performance_evaluation_interval: float = 10.0
    
    # スレッド設定
    thread_timeout: float = 1.0
    max_queue_size: int = 100
    
    # 報酬重み設定
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "exploration_efficiency": 10.0,
        "swarm_count_balance": 2.0,
        "mobility_score": 5.0,
        "learning_transfer_success": 3.0,
        "system_stability": 2.0,
        "collision_penalty": -5.0,
        "energy_efficiency": 1.0,
        "branch_success": 2.0,
        "integration_success": 1.5
    })
    
    def __post_init__(self):
        if self.learningParameter is None:
            self.learningParameter = LearningParameter()
        if self.branch_condition is None:
            self.branch_condition = BranchConditionParam()
        if self.integration_condition is None:
            self.integration_condition = IntegrationConditionParam()
        if self.debug is None:
            self.debug = SystemAgentDebugParam()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返す"""
        return {
            "agent_id": self.agent_id,
            "monitoring_enabled": self.monitoring_enabled,
            "learningParameter": self.learningParameter.__dict__ if self.learningParameter else None,
            "branch_condition": self.branch_condition.__dict__ if self.branch_condition else None,
            "integration_condition": self.integration_condition.__dict__ if self.integration_condition else None,
            "debug": self.debug.__dict__ if self.debug else None,
            "performance_monitoring_enabled": self.performance_monitoring_enabled,
            "performance_threshold": self.performance_threshold,
            "performance_evaluation_interval": self.performance_evaluation_interval,
            "thread_timeout": self.thread_timeout,
            "max_queue_size": self.max_queue_size,
            "reward_weights": self.reward_weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemAgentParam':
        """辞書から作成"""
        # ネストしたオブジェクトを適切に処理
        learning_param_data = data.get('learningParameter')
        learning_param = LearningParameter(**learning_param_data) if learning_param_data else None
        
        branch_condition_data = data.get('branch_condition')
        branch_condition = BranchConditionParam(**branch_condition_data) if branch_condition_data else None
        
        integration_condition_data = data.get('integration_condition')
        integration_condition = IntegrationConditionParam(**integration_condition_data) if integration_condition_data else None
        
        debug_data = data.get('debug')
        debug = SystemAgentDebugParam(**debug_data) if debug_data else None
        
        # 基本データをコピー
        base_data = {k: v for k, v in data.items() 
                    if k not in ['learningParameter', 'branch_condition', 'integration_condition', 'debug']}
        
        return cls(
            learningParameter=learning_param,
            branch_condition=branch_condition,
            integration_condition=integration_condition,
            debug=debug,
            **base_data
        )
    
    def copy(self) -> 'SystemAgentParam':
        """コピーを作成"""
        return SystemAgentParam(
            agent_id=self.agent_id,
            monitoring_enabled=self.monitoring_enabled,
            learningParameter=self.learningParameter.copy() if self.learningParameter else None,
            branch_condition=self.branch_condition.copy() if self.branch_condition else None,
            integration_condition=self.integration_condition.copy() if self.integration_condition else None,
            debug=self.debug.copy() if self.debug else None,
            performance_monitoring_enabled=self.performance_monitoring_enabled,
            performance_threshold=self.performance_threshold,
            performance_evaluation_interval=self.performance_evaluation_interval,
            thread_timeout=self.thread_timeout,
            max_queue_size=self.max_queue_size,
            reward_weights=self.reward_weights.copy()
        )



