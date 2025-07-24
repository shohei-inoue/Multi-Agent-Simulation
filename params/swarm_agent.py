"""
SwarmAgent用のパラメータ定義
群エージェントの設定を管理（移動方向決定に専念）
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, List
from params.learning import LearningParameter


@dataclass
class SwarmDebugParam:
    """SwarmAgent用のデバッグ・ログ設定"""
    enable_debug_log: bool = False
    log_movement_events: bool = True  # 移動イベントのログ
    log_learning_events: bool = True  # 学習イベントのログ
    
    def to_dict(self) -> dict:
        """パラメータを辞書形式で取得"""
        return {
            'enable_debug_log': self.enable_debug_log,
            'log_movement_events': self.log_movement_events,
            'log_learning_events': self.log_learning_events
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SwarmDebugParam':
        """辞書からパラメータを作成"""
        return cls(**data)
    
    def copy(self) -> 'SwarmDebugParam':
        """パラメータのコピーを作成"""
        return SwarmDebugParam(
            enable_debug_log=self.enable_debug_log,
            log_movement_events=self.log_movement_events,
            log_learning_events=self.log_learning_events
        )


@dataclass
class SwarmAgentParam:
    """SwarmAgent用のパラメータ"""
    
    # アルゴリズム設定
    algorithm: Literal["vfh_fuzzy"] = "vfh_fuzzy"
    
    # 学習設定
    isLearning: bool = True  # 学習を有効にするか
    learningParameter: Optional[LearningParameter] = None  # 学習パラメータ
    
    # デバッグ・ログ設定
    debug: Optional[SwarmDebugParam] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.learningParameter is None:
            self.learningParameter = LearningParameter(
                type="a2c",
                model="actor-critic",
                optimizer="adam",
                gamma=0.99,
                learningLate=0.001,
                nStep=5,
                inherit_learning_info=True,
                merge_learning_info=True
            )
        if self.debug is None:
            self.debug = SwarmDebugParam()
    
    def validate(self) -> bool:
        """パラメータの妥当性を検証"""
        # 必要ならここでLearningParameterの値を直接チェック
        # 例: if self.isLearning and self.learningParameter is not None:
        #         if self.learningParameter.gamma < 0 or self.learningParameter.gamma > 1:
        #             return False
        return True
    
    def to_dict(self) -> dict:
        """パラメータを辞書形式で取得"""
        return {
            'algorithm': self.algorithm,
            'isLearning': self.isLearning,
            'learningParameter': self.learningParameter.to_dict() if self.learningParameter else None,
            'debug': self.debug.to_dict() if self.debug else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SwarmAgentParam':
        """辞書からパラメータを作成"""
        # LearningParameterの復元
        learning_param_data = data.get('learningParameter')
        if learning_param_data:
            data['learningParameter'] = LearningParameter.from_dict(learning_param_data)
        
        # SwarmDebugParamの復元
        debug_data = data.get('debug')
        if debug_data:
            data['debug'] = SwarmDebugParam.from_dict(debug_data)
        
        return cls(**data)
    
    def copy(self) -> 'SwarmAgentParam':
        """パラメータのコピーを作成"""
        return SwarmAgentParam(
            algorithm=self.algorithm,
            isLearning=self.isLearning,
            learningParameter=self.learningParameter.copy() if self.learningParameter else None,
            debug=self.debug.copy() if self.debug else None
        )
    
    def get_learning_rate(self) -> float:
        """学習率を取得（学習が無効の場合は0.0を返す）"""
        if not self.isLearning or self.learningParameter is None:
            return 0.0
        return self.learningParameter.learningLate
    
    def get_gamma(self) -> float:
        """割引率を取得（学習が無効の場合は0.0を返す）"""
        if not self.isLearning or self.learningParameter is None:
            return 0.0
        return self.learningParameter.gamma
    
    def get_n_steps(self) -> int:
        """nステップ数を取得（学習が無効の場合は0を返す）"""
        if not self.isLearning or self.learningParameter is None:
            return 0
        return self.learningParameter.nStep
    
    def get_inherit_learning_info(self) -> bool:
        """学習情報の引き継ぎ設定を取得（学習が無効の場合はFalseを返す）"""
        if not self.isLearning or self.learningParameter is None:
            return False
        return self.learningParameter.inherit_learning_info
    
    def get_merge_learning_info(self) -> bool:
        """学習情報の統合設定を取得（学習が無効の場合はFalseを返す）"""
        if not self.isLearning or self.learningParameter is None:
            return False
        return self.learningParameter.merge_learning_info

@dataclass
class ContinuousRange:
    min: float
    max: float
