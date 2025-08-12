"""
群エージェントクラス
移動方向の決定のみを行い、分岐・統合の判定は行わない
分岐時には新しい群を生成、統合時には他の群に取り込まれる
"""

from agents.base_agent import BaseAgent
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Optional
from utils.utils import flatten_state
from params.swarm_agent import SwarmAgentParam


class SwarmAgent(BaseAgent):
    """
    群エージェント - 移動方向の決定のみを行う
    分岐・統合の判定・トリガーは行わない
    """
    def __init__(self, env, algorithm, model, action_space, param: SwarmAgentParam, system_agent=None, swarm_id=None):
        super().__init__(env, algorithm, model, action_space=action_space)
        self.action_space = action_space
        self.param = param
        self.isLearning = param.isLearning
        self.learningParameter = param.learningParameter
        self.debug = param.debug
        self.system_agent = system_agent
        self.swarm_id = swarm_id

    def get_action(self, state: Dict[str, Any], episode: int = 0, log_dir: Optional[str] = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        アクションを取得（学習ログ機能付き）
        Args:
            state: 現在の状態
            episode: エピソード番号
            log_dir: ログディレクトリ
        Returns:
            action: アクション（theta）
            action_info: アクション情報
        """
        # 学習なしモードの場合
        if not self.isLearning or self.model is None:
            # デフォルトのパラメータを使用してアルゴリズムを実行
            default_params = np.array([0.5, 10.0, 5.0])  # th, k_e, k_c
            theta, valid_directions = self.algorithm.policy(state, default_params)
            
            # 分岐・統合条件の判定
            follower_scores = state.get("follower_mobility_scores", [])
            follower_count = len(follower_scores)
            avg_mobility = np.mean(follower_scores) if follower_count > 0 else 0.0
            
            # デバッグ情報を詳細に出力
            print(f"🔍 SwarmAgent {self.swarm_id}（学習なし）: follower_scores詳細分析")
            print(f"   - follower_scores: {follower_scores}")
            print(f"   - follower_count: {follower_count}")
            print(f"   - avg_mobility: {avg_mobility:.6f}")
            print(f"   - 各スコアの詳細:")
            for i, score in enumerate(follower_scores):
                print(f"     [{i}]: {score:.6f}")
            
            print(f"📊 SwarmAgent {self.swarm_id}（学習なし）: follower_scores={follower_scores}, follower_count={follower_count}, avg_mobility={avg_mobility:.3f}, valid_directions={len(valid_directions)}")
            
            # SystemAgentから閾値を取得
            branch_threshold = 0.5  # デフォルト値
            integration_threshold = 0.3  # デフォルト値
            
            if self.system_agent and hasattr(self.system_agent, 'model'):
                if hasattr(self.system_agent.model, 'get_branch_threshold'):
                    branch_threshold = self.system_agent.model.get_branch_threshold() or 0.5
                if hasattr(self.system_agent.model, 'get_integration_threshold'):
                    integration_threshold = self.system_agent.model.get_integration_threshold() or 0.3
            
            should_branch = (
                follower_count >= 3 and
                valid_directions and len(valid_directions) >= 2 and
                avg_mobility >= branch_threshold
            )
            should_integrate = avg_mobility < integration_threshold
            
            # system_agentに送るstateを生成
            system_state = {
                "theta": theta,
                "valid_directions": valid_directions,
                "swarm_id": self.swarm_id,
                "follower_count": follower_count,
                "swarm_count": state.get("swarm_count", 1),
                "swarm_mobility_score": follower_scores,
                "avg_mobility": avg_mobility
            }
            
            # 分岐判定
            if self.system_agent and should_branch:
                print(f"🔥 SwarmAgent {self.swarm_id}: 分岐条件満たし - check_branch呼び出し")
                self.system_agent.check_branch(system_state)
            elif self.system_agent:
                print(f"🔍 SwarmAgent {self.swarm_id}: 分岐条件チェック - follower_count={follower_count}(要求≥3), valid_directions={len(valid_directions)}(要求≥2), avg_mobility={avg_mobility:.3f}(要求≥{branch_threshold})")
            
            # 統合判定
            if self.system_agent and should_integrate:
                print(f"🔥 SwarmAgent {self.swarm_id}: 統合条件満たし - check_integration呼び出し")
                self.system_agent.check_integration(system_state)
            elif self.system_agent:
                print(f"🔍 SwarmAgent {self.swarm_id}: 統合条件チェック - avg_mobility={avg_mobility:.3f}(要求<{integration_threshold})")
            
            return {"theta": theta}, {
                'theta': theta,
                'valid_directions': valid_directions
            }
        
        # 学習ありモードの場合
        assert self.model is not None, "model must not be None"
        
        # 状態をテンソルに変換
        state_vec = tf.convert_to_tensor([flatten_state(state)], dtype=tf.float32)
        
        # モデルから学習パラメータとアクションパラメータを取得
        learning_mu, learning_std, theta_mu, theta_std, value = self.model(state_vec)
        
        # 学習パラメータをサンプリング
        learning_params = self.model.sample_learning_params(learning_mu, learning_std)
        
        # アルゴリズムに学習パラメータを渡してポリシーを実行
        theta, valid_directions = self.algorithm.policy(state, learning_params)
        
        # アクション（theta）をサンプリング
        action_theta = self.model.sample_action(theta_mu, theta_std)
        
        # 学習ログを記録
        if log_dir and hasattr(self, 'logger'):
            self._log_learning_metrics(episode, {
                'learning_th': float(learning_params[0]),
                'learning_k_e': float(learning_params[1]),
                'learning_k_c': float(learning_params[2]),
                'action_theta': float(action_theta),
                'value': float(value),
                'valid_directions_count': len(valid_directions)
            }, log_dir)
        
        # thetaがnumpy.float64の場合はfloatに変換
        theta_value = float(theta) if hasattr(theta, 'numpy') else theta
        
        action = {"theta": theta_value, "valid_directions": valid_directions}
        
        if self.debug and self.debug.enable_debug_log:
            print(f"SwarmAgent action | theta: {theta_value:.3f} ({np.rad2deg(theta_value):.1f}[deg])")
            print(f"SwarmAgent params | th: {learning_params[0]:.3f}, k_e: {learning_params[1]:.3f}, k_c: {learning_params[2]:.3f}")
            print(f"SwarmAgent valid_directions: {len(valid_directions)}")
        
        # 分岐・統合条件の判定
        follower_scores = state.get("follower_mobility_scores", [])
        follower_count = len(follower_scores)
        avg_mobility = np.mean(follower_scores) if follower_count > 0 else 0.0
        
        # SystemAgentから閾値を取得
        branch_threshold = 0.5  # デフォルト値
        integration_threshold = 0.3  # デフォルト値
        
        if self.system_agent and hasattr(self.system_agent, 'model'):
            if hasattr(self.system_agent.model, 'get_branch_threshold'):
                branch_threshold = self.system_agent.model.get_branch_threshold() or 0.5
            if hasattr(self.system_agent.model, 'get_integration_threshold'):
                integration_threshold = self.system_agent.model.get_integration_threshold() or 0.3
        
        should_branch = (
            follower_count >= 3 and
            valid_directions and len(valid_directions) >= 2 and
            avg_mobility >= branch_threshold
        )
        should_integrate = avg_mobility < integration_threshold
        
        # system_agentに送るstateを生成
        system_state = {
            "theta": theta_value,
            "valid_directions": valid_directions,
            "swarm_id": self.swarm_id,
            "follower_count": follower_count,
            "swarm_count": state.get("swarm_count", 1),
            "swarm_mobility_score": follower_scores,
            "avg_mobility": avg_mobility
        }
        
        # 分岐判定
        if self.system_agent and should_branch:
            print(f"🔥 SwarmAgent {self.swarm_id}: 分岐条件満たし - check_branch呼び出し（学習モード）")
            self.system_agent.check_branch(system_state)
        elif self.system_agent:
            print(f"🔍 SwarmAgent {self.swarm_id}: 分岐条件チェック（学習モード） - follower_count={follower_count}(要求≥3), valid_directions={len(valid_directions)}(要求≥2), avg_mobility={avg_mobility:.3f}(要求≥{branch_threshold})")
        
        # 統合判定
        if self.system_agent and should_integrate:
            print(f"🔥 SwarmAgent {self.swarm_id}: 統合条件満たし - check_integration呼び出し（学習モード）")
            self.system_agent.check_integration(system_state)
        elif self.system_agent:
            print(f"🔍 SwarmAgent {self.swarm_id}: 統合条件チェック（学習モード） - avg_mobility={avg_mobility:.3f}(要求<{integration_threshold})")
        
        return {"theta": theta_value}, {
            'theta': float(action_theta) if not isinstance(action_theta, tuple) else float(action_theta[0]),
            'learning_params': learning_params.numpy().tolist() if hasattr(learning_params, 'numpy') else list(learning_params),
            'valid_directions': valid_directions,
            'value': float(value)
        }

    def _log_learning_metrics(self, episode: int, metrics: Dict[str, float], log_dir: str):
        """学習メトリクスをログに記録"""
        try:
            from utils.logger import create_experiment_logger
            logger = create_experiment_logger(log_dir, "swarm_learning")
            logger.log_learning_progress(episode, "swarm", metrics)
            logger.close()
        except ImportError:
            # ログ機能が利用できない場合は無視
            pass

    def train(self, *args, **kwargs):
        """
        学習ロジックは必要に応じて実装
        """
        pass 