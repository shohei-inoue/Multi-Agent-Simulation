"""
分岐処理のアルゴリズム
新しい群の作成、leader選択、follower分割、初期アクション決定を行う
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BranchResult:
    """分岐処理の結果"""
    new_leader: Any  # 新しいleaderロボット
    new_followers: List[Any]  # 新しいfollowerロボットのリスト
    initial_theta: float  # 新しい群の初期移動方向
    branch_score: float  # 分岐の品質スコア


class BaseBranchAlgorithm:
    """分岐アルゴリズムの基底クラス"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def select_branch_leader(self, source_swarm, valid_directions: List, 
                           mobility_scores: List[float]) -> Optional[Any]:
        """新しいleaderを選択"""
        raise NotImplementedError
    
    def select_branch_followers(self, source_swarm, new_leader, 
                               valid_directions: List) -> List[Any]:
        """新しいfollowerを選択"""
        raise NotImplementedError
    
    def determine_initial_action(self, new_leader, valid_directions: List, 
                               mobility_scores: List[float]) -> float:
        """新しい群の初期アクション（theta）を決定"""
        raise NotImplementedError
    
    def execute_branch(self, source_swarm, valid_directions: List, 
                      mobility_scores: List[float]) -> Optional[BranchResult]:
        """分岐処理を実行"""
        # 新しいleaderを選択
        new_leader = self.select_branch_leader(source_swarm, valid_directions, mobility_scores)
        if new_leader is None:
            return None
        
        # 新しいfollowerを選択
        new_followers = self.select_branch_followers(source_swarm, new_leader, valid_directions)
        
        # 初期アクションを決定
        initial_theta = self.determine_initial_action(new_leader, valid_directions, mobility_scores)
        
        # 分岐スコアを計算
        branch_score = self._calculate_branch_score(new_leader, new_followers, valid_directions)
        
        return BranchResult(
            new_leader=new_leader,
            new_followers=new_followers,
            initial_theta=initial_theta,
            branch_score=branch_score
        )
    
    def _calculate_branch_score(self, new_leader, new_followers: List, 
                               valid_directions: List) -> float:
        """分岐の品質スコアを計算"""
        # 簡易的なスコア計算
        # 実際の実装では、より複雑な評価指標を使用
        follower_count = len(new_followers)
        direction_count = len(valid_directions)
        
        # follower数と方向数のバランスを評価
        score = (follower_count / 10.0) * 0.6 + (direction_count / 5.0) * 0.4
        return min(1.0, max(0.0, score))


class RandomBranchAlgorithm(BaseBranchAlgorithm):
    """ランダム分岐アルゴリズム"""
    
    def select_branch_leader(self, source_swarm, valid_directions: List, 
                           mobility_scores: List[float]) -> Optional[Any]:
        """ランダムに新しいleaderを選択"""
        if not source_swarm.followers:
            return None
        
        # ランダムにfollowerを新しいleaderとして選択
        return np.random.choice(source_swarm.followers)
    
    def select_branch_followers(self, source_swarm, new_leader, 
                               valid_directions: List) -> List[Any]:
        """ランダムに新しいfollowerを選択（最低3体を保証）"""
        remaining_followers = [f for f in source_swarm.followers if f != new_leader]
        
        if not remaining_followers:
            return []
        
        # 最低3体のfollowerを保証
        # 残りのfollowerが6体以上の場合：半分に分割
        # 残りのfollowerが6体未満の場合：3体を新しい群に割り当て
        if len(remaining_followers) >= 6:
            split_count = len(remaining_followers) // 2
        else:
            # 最低3体を保証
            split_count = 3
        
        # 選択数を制限
        split_count = min(split_count, len(remaining_followers))
        
        selected_indices = np.random.choice(
            len(remaining_followers), 
            size=split_count, 
            replace=False
        )
        
        return [remaining_followers[i] for i in selected_indices]
    
    def determine_initial_action(self, new_leader, valid_directions: List, 
                               mobility_scores: List[float]) -> float:
        """valid_directionsからランダムに初期アクションを選択"""
        if not valid_directions:
            return np.random.uniform(0, 2 * np.pi)
        
        # valid_directionsからランダムに選択
        selected_direction = np.random.choice(valid_directions)
        return selected_direction.get("angle", np.random.uniform(0, 2 * np.pi))


class MobilityBasedBranchAlgorithm(BaseBranchAlgorithm):
    """mobility_scoreベースの分岐アルゴリズム"""
    
    def select_branch_leader(self, source_swarm, valid_directions: List, 
                           mobility_scores: List[float]) -> Optional[Any]:
        """mobility_scoreに基づいて新しいleaderを選択"""
        if not source_swarm.followers:
            return None
        
        # mobility_scoreが最も高いfollowerを新しいleaderとして選択
        best_follower = None
        best_score = -1.0
        
        for i, follower in enumerate(source_swarm.followers):
            if i < len(mobility_scores):
                score = mobility_scores[i]
            else:
                # mobility_scoresが不足している場合は簡易計算
                score = 0.5  # デフォルトスコア
            
            if score > best_score:
                best_score = score
                best_follower = follower
        
        return best_follower
    
    def select_branch_followers(self, source_swarm, new_leader, 
                               valid_directions: List) -> List[Any]:
        """mobility_scoreに基づいてfollowerを選択（最低3体を保証）"""
        remaining_followers = [f for f in source_swarm.followers if f != new_leader]
        
        if not remaining_followers:
            return []
        
        # 最低3体のfollowerを保証
        # 残りのfollowerが6体以上の場合：半分に分割
        # 残りのfollowerが6体未満の場合：3体を新しい群に割り当て
        if len(remaining_followers) >= 6:
            split_count = len(remaining_followers) // 2
        else:
            # 最低3体を保証
            split_count = 3
        
        # 選択数を制限
        split_count = min(split_count, len(remaining_followers))
        
        # 簡易的にランダム選択（実際の実装ではmobility_scoreでソート）
        selected_indices = np.random.choice(
            len(remaining_followers), 
            size=split_count, 
            replace=False
        )
        
        return [remaining_followers[i] for i in selected_indices]
    
    def determine_initial_action(self, new_leader, valid_directions: List, 
                               mobility_scores: List[float]) -> float:
        """valid_directionsからスコアに基づいて初期アクションを選択"""
        if not valid_directions:
            return np.random.uniform(0, 2 * np.pi)
        
        # スコアに基づいて重み付け選択
        scores = np.array([d.get("score", 1.0) for d in valid_directions])
        if scores.sum() > 0:
            probs = scores / scores.sum()
            selected_idx = np.random.choice(len(valid_directions), p=probs)
            return valid_directions[selected_idx].get("angle", np.random.uniform(0, 2 * np.pi))
        else:
            # スコアが全て0の場合はランダム選択
            selected_direction = np.random.choice(valid_directions)
            return selected_direction.get("angle", np.random.uniform(0, 2 * np.pi))


def create_branch_algorithm(algorithm_type: str, **kwargs) -> BaseBranchAlgorithm:
    """分岐アルゴリズムを作成"""
    if algorithm_type == "random":
        return RandomBranchAlgorithm(**kwargs)
    elif algorithm_type == "mobility_based":
        return MobilityBasedBranchAlgorithm(**kwargs)
    else:
        raise ValueError(f"Unknown branch algorithm type: {algorithm_type}") 