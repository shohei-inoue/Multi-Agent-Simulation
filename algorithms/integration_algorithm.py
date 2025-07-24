"""
統合処理のアルゴリズム
群の統合先選択、統合方法決定を行う
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IntegrationResult:
    """統合処理の結果"""
    target_swarm_id: int  # 統合先の群ID
    integration_score: float  # 統合の品質スコア
    integration_method: str  # 統合方法


class BaseIntegrationAlgorithm:
    """統合アルゴリズムの基底クラス"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def select_integration_target(self, source_swarm, all_swarms: List, 
                                source_swarm_id: int) -> Optional[int]:
        """統合先の群を選択"""
        raise NotImplementedError
    
    def determine_integration_method(self, source_swarm, target_swarm) -> str:
        """統合方法を決定"""
        raise NotImplementedError
    
    def execute_integration(self, source_swarm, all_swarms: List, 
                          source_swarm_id: int) -> Optional[IntegrationResult]:
        """統合処理を実行"""
        # 統合先を選択
        target_swarm_id = self.select_integration_target(source_swarm, all_swarms, source_swarm_id)
        if target_swarm_id is None:
            return None
        
        # 統合先の群を取得
        target_swarm = None
        for swarm in all_swarms:
            if swarm.swarm_id == target_swarm_id:
                target_swarm = swarm
                break
        
        if target_swarm is None:
            return None
        
        # 統合方法を決定
        integration_method = self.determine_integration_method(source_swarm, target_swarm)
        
        # 統合スコアを計算
        integration_score = self._calculate_integration_score(source_swarm, target_swarm)
        
        return IntegrationResult(
            target_swarm_id=target_swarm_id,
            integration_score=integration_score,
            integration_method=integration_method
        )
    
    def _calculate_integration_score(self, source_swarm, target_swarm) -> float:
        """統合の品質スコアを計算"""
        # 簡易的なスコア計算
        # 実際の実装では、より複雑な評価指標を使用
        
        # 群のサイズバランスを評価
        source_size = source_swarm.get_robot_count()
        target_size = target_swarm.get_robot_count()
        
        # サイズが近いほど高いスコア
        size_diff = abs(source_size - target_size)
        max_size = max(source_size, target_size)
        size_score = 1.0 - (size_diff / max_size) if max_size > 0 else 0.0
        
        # 距離を評価（距離が近いほど高いスコア）
        distance = self._calculate_swarm_distance(source_swarm, target_swarm)
        distance_score = max(0.0, 1.0 - (distance / 20.0))  # 20.0を最大距離とする
        
        # スコアを統合
        score = size_score * 0.4 + distance_score * 0.6
        return min(1.0, max(0.0, score))
    
    def _calculate_swarm_distance(self, swarm1, swarm2) -> float:
        """2つの群の距離を計算"""
        # 群の中心座標を計算
        center1 = self._calculate_swarm_center(swarm1)
        center2 = self._calculate_swarm_center(swarm2)
        
        return float(np.linalg.norm(center1 - center2))
    
    def _calculate_swarm_center(self, swarm) -> np.ndarray:
        """群の中心座標を計算"""
        all_robots = swarm.get_all_robots()
        if not all_robots:
            return np.array([0.0, 0.0])
        
        coordinates = np.array([robot.coordinate for robot in all_robots])
        return np.mean(coordinates, axis=0)


class NearestIntegrationAlgorithm(BaseIntegrationAlgorithm):
    """最も近い群に統合するアルゴリズム"""
    
    def select_integration_target(self, source_swarm, all_swarms: List, 
                                source_swarm_id: int) -> Optional[int]:
        """最も近い群を統合先として選択"""
        min_distance = float('inf')
        nearest_swarm_id = None
        
        for swarm in all_swarms:
            if swarm.swarm_id == source_swarm_id:
                continue
            
            distance = self._calculate_swarm_distance(source_swarm, swarm)
            if distance < min_distance:
                min_distance = distance
                nearest_swarm_id = swarm.swarm_id
        
        return nearest_swarm_id
    
    def determine_integration_method(self, source_swarm, target_swarm) -> str:
        """統合方法を決定（簡易的に'merge'を返す）"""
        return "merge"


class LargestIntegrationAlgorithm(BaseIntegrationAlgorithm):
    """最も大きな群に統合するアルゴリズム"""
    
    def select_integration_target(self, source_swarm, all_swarms: List, 
                                source_swarm_id: int) -> Optional[int]:
        """最も大きな群を統合先として選択"""
        max_size = -1
        largest_swarm_id = None
        
        for swarm in all_swarms:
            if swarm.swarm_id == source_swarm_id:
                continue
            
            size = swarm.get_robot_count()
            if size > max_size:
                max_size = size
                largest_swarm_id = swarm.swarm_id
        
        return largest_swarm_id
    
    def determine_integration_method(self, source_swarm, target_swarm) -> str:
        """統合方法を決定"""
        # 統合先が大きい場合は'merge'、小さい場合は'absorb'
        if target_swarm.get_robot_count() > source_swarm.get_robot_count():
            return "merge"
        else:
            return "absorb"


class PerformanceBasedIntegrationAlgorithm(BaseIntegrationAlgorithm):
    """性能ベースの統合アルゴリズム"""
    
    def select_integration_target(self, source_swarm, all_swarms: List, 
                                source_swarm_id: int) -> Optional[int]:
        """性能が最も高い群を統合先として選択"""
        best_score = -1.0
        best_swarm_id = None
        
        for swarm in all_swarms:
            if swarm.swarm_id == source_swarm_id:
                continue
            
            # 群の性能スコアを計算
            performance_score = self._calculate_swarm_performance(swarm)
            if performance_score > best_score:
                best_score = performance_score
                best_swarm_id = swarm.swarm_id
        
        return best_swarm_id
    
    def determine_integration_method(self, source_swarm, target_swarm) -> str:
        """統合方法を決定"""
        # 性能に基づいて統合方法を決定
        source_performance = self._calculate_swarm_performance(source_swarm)
        target_performance = self._calculate_swarm_performance(target_swarm)
        
        if target_performance > source_performance:
            return "merge"  # 高性能な群に統合
        else:
            return "absorb"  # 低性能な群を吸収
    
    def _calculate_swarm_performance(self, swarm) -> float:
        """群の性能スコアを計算"""
        # 簡易的な性能計算
        # 実際の実装では、探査効率、移動効率などを考慮
        
        # 群のサイズ
        size_score = min(1.0, swarm.get_robot_count() / 10.0)
        
        # 探査率（仮想的な値）
        exploration_score = getattr(swarm, 'exploration_rate', 0.0)
        
        # 性能スコアを統合
        performance = size_score * 0.6 + exploration_score * 0.4
        return min(1.0, max(0.0, performance))


def create_integration_algorithm(algorithm_type: str, **kwargs) -> BaseIntegrationAlgorithm:
    """統合アルゴリズムを作成"""
    if algorithm_type == "nearest":
        return NearestIntegrationAlgorithm(**kwargs)
    elif algorithm_type == "largest":
        return LargestIntegrationAlgorithm(**kwargs)
    elif algorithm_type == "performance_based":
        return PerformanceBasedIntegrationAlgorithm(**kwargs)
    else:
        raise ValueError(f"Unknown integration algorithm type: {algorithm_type}") 