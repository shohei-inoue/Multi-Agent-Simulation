"""
システムエージェントクラス（全体情報監視役）
SwarmAgentの行動で呼び出され、全体情報を監視
分岐時には新しいSwarmAgentを生成、統合時には群を取り込む
"""

from agents.base_agent import BaseAgent
import numpy as np
from typing import Dict, Any, Tuple, Optional
from params.system_agent import SystemAgentParam
import time
import copy


class SystemAgent(BaseAgent):
    """システムエージェント（全体情報監視役）- 群の分岐・統合を管理"""
    
    def __init__(self, env, algorithm, model, action_space, param: SystemAgentParam):
        super().__init__(env, algorithm, model, action_space=action_space)
        self.param = param
        self.isLearning = param.learningParameter is not None
        self.learningParameter = param.learningParameter
        self.debug = param.debug
        self.branchCondition = param.branch_condition
        self.integrationCondition = param.integration_condition
        
        # 群管理
        self.swarm_agents = {}  # swarm_id -> swarm_agent
        self.next_swarm_id = 0
        
        # 学習情報の管理
        self.learning_history = {}  # swarm_id -> 学習情報のマッピング
        
        # 分岐・統合のクールダウン（ステップベース）
        self.last_branch_step = -999  # 最初から分岐可能にする
        self.last_integration_step = -999  # 最初から統合可能にする
        self.current_step = 0
        
        # システム状態
        self.current_swarm_count = 1
        self.branch_threshold = 0.5  # 分岐閾値（固定）
        self.integration_threshold = 0.3  # 統合閾値（固定）
        
        # 学習による閾値調整のためのパラメータ
        self.adaptive_branch_threshold = True  # 適応的分岐閾値
        self.adaptive_integration_threshold = True  # 適応的統合閾値
        
        # 固定閾値設定
        self.use_fixed_thresholds = getattr(param, 'use_fixed_thresholds', False)
        self.fixed_branch_threshold = getattr(param, 'fixed_branch_threshold', 0.5)
        self.fixed_integration_threshold = getattr(param, 'fixed_integration_threshold', 0.3)

    def reset_episode(self):
        """エピソード開始時にクールダウンをリセット"""
        self.last_branch_step = -999  # エピソード開始時から分岐可能にする
        self.last_integration_step = -999  # エピソード開始時から統合可能にする
        self.current_step = 0
        if self.debug and self.debug.log_branch_events:
            print("SystemAgent: エピソードリセット - ステップカウンターリセット")

    def update_step(self):
        """ステップカウンターを更新"""
        self.current_step += 1
        if self.debug and self.debug.log_branch_events and self.current_step % 10 == 0:
            print(f"📊 SystemAgent ステップ更新: current_step={self.current_step}, last_branch_step={self.last_branch_step}, last_integration_step={self.last_integration_step}")

    def check_branch(self, system_state: Dict[str, Any]) -> bool:
        """
        分岐条件をチェックし、条件を満たせば分岐を実行
        Returns: 分岐が実行されたかどうか
        """
        # 分岐が無効になっている場合は分岐を実行しない
        if not self.branchCondition.branch_enabled:
            return False
        
        # クールダウンチェックを無効化（毎ステップで分岐可能）
        # if self.current_step - self.last_branch_step < self.branchCondition.swarm_creation_cooldown:
        #     if self.debug and self.debug.log_branch_events:
        #         print(f"🕒 分岐クールダウン中: {self.current_step - self.last_branch_step}/{self.branchCondition.swarm_creation_cooldown}ステップ")
        #     return False
        
        # 分岐条件チェック
        follower_count = system_state.get("follower_count", 0)
        valid_directions = system_state.get("valid_directions", [])
        avg_mobility = system_state.get("avg_mobility", 0.0)
        
        # 分岐閾値の決定
        if hasattr(self, 'use_fixed_thresholds') and self.use_fixed_thresholds:
            # 固定閾値を使用（学習による閾値変更を防ぐ）
            effective_threshold = self.fixed_branch_threshold
        elif self.adaptive_branch_threshold and hasattr(self, 'model') and self.model is not None:
            # 学習モデルがある場合は、現在のmobilityに基づいて閾値を調整
            # mobilityが高い場合は閾値も高く、低い場合は閾値も低くする
            
            # 学習の進行度を考慮した閾値調整
            # 学習が進むほど分岐を促進する（mobilityが高くても分岐しやすくする）
            learning_progress_factor = 0.6  # 学習が進むと閾値を60%に下げる
            
            # 基本の適応閾値
            adaptive_threshold = max(0.2, min(0.8, avg_mobility * 0.8 + 0.3))
            
            # 学習促進のための閾値調整
            learning_enhanced_threshold = max(0.15, adaptive_threshold * learning_progress_factor)
            
            effective_threshold = learning_enhanced_threshold
        else:
            # 学習モデルがない場合は固定閾値を使用
            effective_threshold = self.branch_threshold
        
        # 基本分岐条件
        basic_branch_condition = (
            follower_count >= 3 and
            len(valid_directions) >= 2 and
            avg_mobility >= effective_threshold
        )
        
        # 追加分岐条件：学習が進んでいる場合の分岐促進
        learning_branch_condition = False
        if hasattr(self, 'model') and self.model is not None:
            # 学習モデルがある場合、追加の分岐条件をチェック
            # 学習によりmobilityが改善されている場合、より柔軟な分岐条件を適用
            
            # 1. 十分なfollowerがいる場合
            # 2. 有効な方向が複数ある場合
            # 3. 学習による改善を考慮した分岐条件
            learning_branch_condition = (
                follower_count >= 2 and  # より緩い条件
                len(valid_directions) >= 2 and
                (avg_mobility >= 0.2 or  # 低いmobilityでも分岐可能
                 (follower_count >= 4 and len(valid_directions) >= 3))  # 十分なリソースがある場合
            )
        
        # 学習による分岐促進：学習が進んでいる場合は基本条件を緩和
        enhanced_basic_condition = False
        if hasattr(self, 'model') and self.model is not None:
            # 学習モデルがある場合、基本条件も緩和
            enhanced_basic_condition = (
                follower_count >= 2 and  # 3→2に緩和
                len(valid_directions) >= 2 and
                avg_mobility >= max(0.2, effective_threshold * 0.6)  # 閾値を60%に緩和
            )
        
        # Config C用の初期分岐促進条件（mobility_scoreが低くても分岐可能）
        initial_branch_condition = False
        if hasattr(self, 'use_fixed_thresholds') and self.use_fixed_thresholds:
            # 固定閾値使用時は、初期段階でも分岐を促進
            initial_branch_condition = (follower_count >= 2 and len(valid_directions) >= 2 and avg_mobility >= 0.05)
        
        should_branch = basic_branch_condition or learning_branch_condition or enhanced_basic_condition or initial_branch_condition
        
        # デバッグ出力を追加
        if self.debug and self.debug.log_branch_events:
            print(f"🔍 分岐条件チェック (Step {self.current_step}): ")
            if hasattr(self, 'use_fixed_thresholds') and self.use_fixed_thresholds:
                print(f"   固定閾値使用: {effective_threshold:.3f}")
            else:
                print(f"   適応閾値使用: {effective_threshold:.3f}")
            print(f"   基本条件: follower_count={follower_count}(≥3), valid_directions={len(valid_directions)}(≥2), avg_mobility={avg_mobility:.3f}(≥{effective_threshold:.3f})")
            print(f"   学習条件: follower_count={follower_count}(≥2), avg_mobility={avg_mobility:.3f}(≥0.2) or (follower≥4 and directions≥3)")
            print(f"   強化条件: follower_count={follower_count}(≥2), avg_mobility={avg_mobility:.3f}(≥{max(0.2, effective_threshold * 0.6):.3f})")
            if hasattr(self, 'use_fixed_thresholds') and self.use_fixed_thresholds:
                print(f"   初期促進: follower_count={follower_count}(≥2), avg_mobility={avg_mobility:.3f}(≥0.05)")
            print(f"   結果: [基本:{basic_branch_condition}, 学習:{learning_branch_condition}, 強化:{enhanced_basic_condition}, 初期促進:{initial_branch_condition}]")
        
        if should_branch:
            # 分岐実行
            if self.debug and self.debug.log_branch_events:
                print(f"🌟 分岐実行 (Step {self.current_step})")
            self._execute_branch(system_state)
            self.last_branch_step = self.current_step
            return True
        else:
            if self.debug and self.debug.log_branch_events:
                print(f"❌ 分岐条件不満足 (Step {self.current_step})")
                print(f"   詳細分析:")
                print(f"     基本条件: follower_count={follower_count}(要求≥3), valid_directions={len(valid_directions)}(要求≥2), avg_mobility={avg_mobility:.3f}(要求≥{effective_threshold:.3f})")
                if hasattr(self, 'model') and self.model is not None:
                    print(f"     学習条件: follower_count={follower_count}(要求≥2), avg_mobility={avg_mobility:.3f}(要求≥0.2) or (follower≥4 and directions≥3)")
                    print(f"     強化条件: follower_count={follower_count}(要求≥2), avg_mobility={avg_mobility:.3f}(要求≥{max(0.2, effective_threshold * 0.6):.3f})")
                    print(f"     学習促進: 閾値を{effective_threshold:.3f}に調整（基本閾値{self.branch_threshold:.3f}の{effective_threshold/self.branch_threshold*100:.1f}%）")
                if hasattr(self, 'use_fixed_thresholds') and self.use_fixed_thresholds:
                    print(f"     初期促進: follower_count={follower_count}(要求≥2), avg_mobility={avg_mobility:.3f}(要求≥0.05)")
        
        return False

    def check_integration(self, system_state: Dict[str, Any]) -> bool:
        """
        統合条件をチェックし、条件を満たせば統合を実行
        Returns: 統合が実行されたかどうか
        """
        # 統合が無効になっている場合は統合を実行しない
        if not self.integrationCondition.integration_enabled:
            return False
        
        # クールダウンチェックを無効化（毎ステップで統合可能）
        # if self.current_step - self.last_integration_step < self.integrationCondition.swarm_merge_cooldown:
        #     if self.debug and self.debug.log_branch_events:
        #         print(f"🕒 統合クールダウン中: {self.current_step - self.last_integration_step}/{self.integrationCondition.swarm_merge_cooldown}ステップ")
        #     return False
        
        # 統合条件チェック
        avg_mobility = system_state.get("avg_mobility", 0.0)
        swarm_count = system_state.get("swarm_count", 1)
        
        # より厳格な統合条件
        # 1. 最低2群以上必要
        # 2. mobility_scoreが閾値未満（低い性能の場合のみ）
        # 3. 群数が多すぎる場合（5群以上）も統合を促進
        # 4. 探査領域が重複している群のみ統合対象とする
        # 5. ランダム要素を追加して統合頻度を制御
        base_condition = (
            swarm_count >= self.integrationCondition.min_swarms_for_integration and
            (avg_mobility < self.integration_threshold or swarm_count >= 5)
        )
        
        # 探査領域の重複チェック（統合対象の群が存在するかチェック）
        has_overlapping_swarms = False
        if base_condition and hasattr(self.env, 'check_exploration_area_overlap'):
            swarm_id = system_state.get("swarm_id")
            if swarm_id is not None:
                # 統合対象となる群が存在するかチェック
                for target_swarm_id in self.swarm_agents.keys():
                    if target_swarm_id != swarm_id:
                        if self.env.check_exploration_area_overlap(swarm_id, target_swarm_id):
                            has_overlapping_swarms = True
                            break
        
        # 統合の確率を制御（20%の確率で統合を実行）
        random_factor = np.random.random()
        should_integrate = base_condition and has_overlapping_swarms and random_factor < 0.2
        
        # デバッグ出力を追加
        if self.debug and self.debug.log_branch_events:
            print(f"🔍 統合条件チェック (Step {self.current_step}): "
                  f"swarm_count={swarm_count}(≥{self.integrationCondition.min_swarms_for_integration}), "
                  f"avg_mobility={avg_mobility:.3f}(<{self.integration_threshold} or ≥5群), "
                  f"has_overlapping={has_overlapping_swarms}, "
                  f"random={random_factor:.3f}(<0.2)")
        
        if should_integrate:
            # 統合実行
            if self.debug and self.debug.log_branch_events:
                print(f"🔗 統合実行 (Step {self.current_step})")
            self._execute_integration(system_state)
            self.last_integration_step = self.current_step
            return True
        else:
            if self.debug and self.debug.log_branch_events:
                print(f"❌ 統合条件不満足 (Step {self.current_step})")
        
        return False

    def _execute_branch(self, system_state: Dict[str, Any]):
        """分岐処理を実行（新しいSwarmAgentを生成）"""
        swarm_id = system_state.get("swarm_id")
        if swarm_id is None:
            if self.debug and self.debug.enable_debug_log:
                print("SystemAgent: swarm_id is None, skipping branch")
            return
            
        valid_directions = system_state.get("valid_directions", [])
        
        if self.debug and self.debug.enable_debug_log:
            print(f"SystemAgent: Executing branch for swarm {swarm_id}")
        
        # 1. 環境から新しい群IDを取得
        if hasattr(self.env, 'get_next_swarm_id'):
            new_swarm_id = self.env.get_next_swarm_id()
        else:
            # フォールバック: 独自のID管理
            new_swarm_id = self.next_swarm_id
            self.next_swarm_id += 1
        
        # 2. 元の群から学習情報を引き継ぐ
        learning_info = self._inherit_learning_info(swarm_id)
        
        # 3. 新しいSwarmAgentを作成
        new_swarm_agent = self._create_new_swarm_agent(
            swarm_id, 
            new_swarm_id, 
            learning_info
        )
        
        # 新しいSwarmAgentがNoneの場合は分岐を中止
        if new_swarm_agent is None:
            if self.debug and self.debug.enable_debug_log:
                print(f"SystemAgent: Failed to create new swarm agent for swarm {swarm_id}")
            return
        
        # 4. 新しい群エージェントを登録
        self.swarm_agents[new_swarm_id] = new_swarm_agent
        self.current_swarm_count += 1
        
        if self.debug and self.debug.enable_debug_log:
            print(f"SystemAgent: Registered new swarm agent {new_swarm_id}")
        
        # 5. 環境の分岐処理を呼び出し（アルゴリズムタイプを渡す）
        if hasattr(self.env, 'handle_swarm_branch'):
            self.env.handle_swarm_branch(swarm_id, new_swarm_id, valid_directions)
        
        if self.debug and self.debug.enable_debug_log:
            print(f"SystemAgent: Branch completed - new swarm {new_swarm_id} created")

    def _execute_integration(self, system_state: Dict[str, Any]):
        """統合処理を実行（群を取り込む）"""
        swarm_id = system_state.get("swarm_id")
        if swarm_id is None:
            if self.debug and self.debug.enable_debug_log:
                print("SystemAgent: swarm_id is None, skipping integration")
            return
        
        if self.debug and self.debug.enable_debug_log:
            print(f"SystemAgent: Executing integration for swarm {swarm_id}")
        
        # 1. 最も近い群を探す（アルゴリズムタイプを使用）
        target_swarm_id = self._find_nearest_swarm(swarm_id)
        
        if target_swarm_id is None:
            if self.debug and self.debug.enable_debug_log:
                print(f"SystemAgent: No target swarm found for integration")
            return
        
        # 2. 学習情報を統合
        self._merge_learning_info(swarm_id, target_swarm_id)
        
        # 3. 統合元の群エージェントを削除
        if swarm_id in self.swarm_agents:
            del self.swarm_agents[swarm_id]
            self.current_swarm_count -= 1
        
        # 4. 環境の統合処理を呼び出し
        if hasattr(self.env, 'handle_swarm_integration'):
            self.env.handle_swarm_integration(swarm_id, target_swarm_id)
        
        if self.debug and self.debug.enable_debug_log:
            print(f"SystemAgent: Integration completed - swarm {swarm_id} merged into {target_swarm_id}")

    def _find_nearest_swarm(self, source_swarm_id: int) -> Optional[int]:
        """最も近い群を探す（探査領域が重複している群のみ）"""
        if source_swarm_id not in self.swarm_agents:
            return None
        
        source_agent = self.swarm_agents[source_swarm_id]
        
        # 統合アルゴリズムを使用して統合先を決定
        from algorithms.integration_algorithm import create_integration_algorithm
        
        # 設定からアルゴリズムタイプを取得
        algorithm_type = getattr(self.param.integration_condition, 'integration_algorithm', 'nearest')
        integration_algorithm = create_integration_algorithm(algorithm_type)
        
        # 探査領域が重複している最も近い群を探す
        min_distance = float('inf')
        nearest_swarm_id = None
        
        for target_swarm_id, target_agent in self.swarm_agents.items():
            if target_swarm_id == source_swarm_id:
                continue
            
            # 探査領域の重複をチェック
            if hasattr(self.env, 'check_exploration_area_overlap'):
                if not self.env.check_exploration_area_overlap(source_swarm_id, target_swarm_id):
                    continue  # 探査領域が重複していない場合はスキップ
            
            # 距離計算（簡易版）
            distance = self._calculate_swarm_distance(source_agent, target_agent)
            if distance < min_distance:
                min_distance = distance
                nearest_swarm_id = target_swarm_id
        
        return nearest_swarm_id

    def _calculate_swarm_distance(self, swarm1, swarm2) -> float:
        """2つの群の距離を計算（簡易版）"""
        # 実際の実装では、群の中心座標などを使用
        # 現在は簡易的にランダム値を返す
        return np.random.uniform(1.0, 10.0)

    def _inherit_learning_info(self, source_swarm_id: int) -> Dict[str, Any]:
        """学習情報を引き継ぐ"""
        if source_swarm_id in self.learning_history:
            original_info = self.learning_history[source_swarm_id]
            inherited_info = copy.deepcopy(original_info)
            inherited_info['inherited_from'] = source_swarm_id
            inherited_info['inherit_time'] = time.time()
            return inherited_info
        else:
            return {
                'model_weights': None,
                'optimizer_state': None,
                'training_history': [],
                'inherited_from': source_swarm_id,
                'inherit_time': time.time()
            }

    def _merge_learning_info(self, source_swarm_id: int, target_swarm_id: int):
        """学習情報を統合"""
        if source_swarm_id in self.learning_history and target_swarm_id in self.learning_history:
            source_info = self.learning_history[source_swarm_id]
            target_info = self.learning_history[target_swarm_id]
            
            merged_info = {
                'model_weights': self._merge_model_weights(
                    source_info.get('model_weights'),
                    target_info.get('model_weights')
                ),
                'optimizer_state': target_info.get('optimizer_state'),
                'training_history': target_info['training_history'] + source_info['training_history'],
                'merged_from': source_swarm_id,
                'merge_time': time.time()
            }
            
            self.learning_history[target_swarm_id] = merged_info

    def _merge_model_weights(self, source_weights, target_weights):
        """モデルの重みを統合"""
        if source_weights is None or target_weights is None:
            return target_weights
        # 現在は統合先を優先
        return target_weights

    def _create_new_swarm_agent(self, source_swarm_id: int, new_swarm_id: int, 
                               learning_info: Dict[str, Any]):
        """新しい群エージェントを作成"""
        if source_swarm_id not in self.swarm_agents:
            print(f"Warning: Source swarm {source_swarm_id} not found in registered agents")
            return None
        
        source_agent = self.swarm_agents[source_swarm_id]
        
        # source_agentがNoneの場合のチェック
        if source_agent is None:
            print(f"Warning: Source agent for swarm {source_swarm_id} is None")
            return None
        
        # 新しいSwarmAgentを作成（ファクトリを使用）
        try:
            from agents.agent_factory import create_branched_swarm_agent
            new_agent = create_branched_swarm_agent(source_agent, new_swarm_id, learning_info)
            return new_agent
        except Exception as e:
            print(f"Error creating branched swarm agent: {e}")
            return None

    def register_swarm_agent(self, swarm_agent, swarm_id: int):
        """群エージェントを登録"""
        self.swarm_agents[swarm_id] = swarm_agent
        self.current_swarm_count = len(self.swarm_agents)

    def unregister_swarm_agent(self, swarm_id: int):
        """群エージェントを登録解除"""
        if swarm_id in self.swarm_agents:
            del self.swarm_agents[swarm_id]
            self.current_swarm_count = len(self.swarm_agents)

    def get_action(self, state: Dict[str, Any], episode: int = 0, log_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        アクションを取得（学習ログ機能付き）
        Args:
            state: 現在の状態
            episode: エピソード番号
            log_dir: ログディレクトリ
        Returns:
            action: アクション（action_type, target_swarm_id）
            action_info: アクション情報
        """
        if self.model is not None:
            # 状態をテンソル形式に変換
            import tensorflow as tf
            import numpy as np
            
            # 簡易的な状態ベクトルを作成（実際の実装では適切な観測空間を使用）
            state_vector = np.array([
                state.get('episode', 0),
                state.get('step', 0),
                state.get('swarm_count', 0),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            state_tensor = tf.convert_to_tensor([state_vector], dtype=tf.float32)
            
            # モデルから閾値とアクションを取得
            threshold_mu, threshold_std, action_type_probs, target_swarm_probs, value = self.model(state_tensor)
            
            # 閾値をサンプリング
            thresholds = self.model.sample_thresholds(threshold_mu, threshold_std)
            
            # アクションタイプとターゲット群をサンプリング
            action_type = tf.random.categorical(tf.math.log(action_type_probs), 1)[0, 0]
            target_swarm_id = tf.random.categorical(tf.math.log(target_swarm_probs), 1)[0, 0]
            
            # 閾値を適切に抽出（最初の要素から値を取得）
            branch_threshold = float(thresholds[0].numpy()[0, 0])
            integration_threshold = float(thresholds[1].numpy()[0, 0])
            
            # 学習ログを記録
            if log_dir and hasattr(self, 'logger'):
                self._log_learning_metrics(episode, {
                    'branch_threshold': branch_threshold,
                    'integration_threshold': integration_threshold,
                    'action_type': int(action_type),
                    'target_swarm_id': int(target_swarm_id),
                    'value': float(value)
                }, log_dir)
            
            return {
                'action_type': int(action_type),
                'target_swarm_id': int(target_swarm_id)
            }, {
                'thresholds': [branch_threshold, integration_threshold],
                'action_type_probs': action_type_probs.numpy().tolist(),
                'target_swarm_probs': target_swarm_probs.numpy().tolist(),
                'value': float(value)
            }
        else:
            # モデルがない場合はデフォルト動作
            return {
                'action_type': 0,  # none
                'target_swarm_id': 0
            }, {}

    def _log_learning_metrics(self, episode: int, metrics: Dict[str, float], log_dir: str):
        """学習メトリクスをログに記録"""
        try:
            from utils.logger import create_experiment_logger
            logger = create_experiment_logger(log_dir, "system_learning")
            logger.log_learning_progress(episode, "system", metrics)
            logger.close()
        except ImportError:
            # ログ機能が利用できない場合は無視
            pass

    def train(self, *args, **kwargs):
        """システムエージェントの学習"""
        if not self.isLearning:
            return
        # 学習ロジックは必要に応じて実装
        pass

    def get_swarm_agents(self) -> Dict[int, Any]:
        """登録されている群エージェントの辞書を取得"""
        return self.swarm_agents.copy()

    def get_learning_history(self) -> Dict[str, Any]:
        """学習履歴を取得"""
        return self.learning_history.copy()

    def get_current_swarm_count(self) -> int:
        """現在の群数を取得"""
        return self.current_swarm_count 