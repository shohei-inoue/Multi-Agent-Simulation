"""
SystemAgent用のActor-Criticモデル
アクション出力: action_type（branch/integrate/none）とtarget_swarm_id
学習パラメータ: branch_threshold, integration_threshold
"""

import tensorflow as tf
import numpy as np
import keras


class SystemActorCritic(keras.Model):
    """
    SystemAgent用のActor-Criticモデル
    アクション出力: action_type, target_swarm_id
    学習パラメータ: branch_threshold, integration_threshold
    """
    
    def __init__(self, input_dim: int, max_swarms: int = 10):
        """
        SystemActorCriticモデルを初期化
        Args:
            input_dim (int): 入力状態の次元数
            max_swarms (int): 最大群数
        """
        super().__init__()
        
        # 共有レイヤー
        self.shared_dense1 = keras.layers.Dense(128, activation='relu')
        self.shared_dense2 = keras.layers.Dense(64, activation='relu')
        
        # 学習用パラメータ出力（branch_threshold, integration_threshold）
        self.threshold_output = keras.layers.Dense(2)  # branch_threshold, integration_threshold
        self.threshold_log_std = tf.Variable(initial_value=tf.zeros(2), trainable=True)
        
        # アクション出力
        # action_type: branch/integrate/none（3クラス分類）
        self.action_type_logits = keras.layers.Dense(3)  # 3つのアクションタイプ
        
        # target_swarm_id: 統合先の群ID（max_swarms個の選択肢）
        self.target_swarm_logits = keras.layers.Dense(max_swarms)
        
        # Critic（価値関数）出力
        self.critic_output = keras.layers.Dense(1)
        
        self.input_dim = input_dim
        self.max_swarms = max_swarms
    
    def call(self, state):
        """
        フォワードパス
        Args:
            state (tf.Tensor): 入力状態
        Returns:
            threshold_mu (tf.Tensor): 閾値パラメータの平均
            threshold_std (tf.Tensor): 閾値パラメータの標準偏差
            action_type_probs (tf.Tensor): アクションタイプの確率分布
            target_swarm_probs (tf.Tensor): ターゲット群の確率分布
            value (tf.Tensor): 状態価値
        """
        x = self.shared_dense1(state)
        x = self.shared_dense2(x)
        
        # 学習用パラメータ（branch_threshold, integration_threshold）
        raw_thresholds = self.threshold_output(x)
        threshold_mu = tf.nn.sigmoid(raw_thresholds)  # 0-1の範囲
        threshold_std = tf.exp(self.threshold_log_std)
        
        # アクションタイプの確率分布
        action_type_logits = self.action_type_logits(x)
        action_type_probs = tf.nn.softmax(action_type_logits)
        
        # ターゲット群の確率分布
        target_swarm_logits = self.target_swarm_logits(x)
        target_swarm_probs = tf.nn.softmax(target_swarm_logits)
        
        # 価値関数
        value = self.critic_output(x)
        
        return threshold_mu, threshold_std, action_type_probs, target_swarm_probs, value
    
    @staticmethod
    def sample_action(action_type_probs, target_swarm_probs):
        """
        アクション（action_type, target_swarm_id）をサンプリング
        Args:
            action_type_probs (tf.Tensor): アクションタイプの確率分布
            target_swarm_probs (tf.Tensor): ターゲット群の確率分布
        Returns:
            action_type (tf.Tensor): サンプリングされたアクションタイプ
            target_swarm_id (tf.Tensor): サンプリングされたターゲット群ID
            action_type_log_prob (tf.Tensor): アクションタイプの対数確率
            target_swarm_log_prob (tf.Tensor): ターゲット群の対数確率
        """
        # アクションタイプをサンプリング
        action_type = tf.random.categorical(tf.math.log(action_type_probs + 1e-8), 1)
        action_type_log_prob = tf.reduce_sum(
            tf.math.log(action_type_probs + 1e-8) * tf.one_hot(action_type, 3), axis=-1
        )
        
        # ターゲット群IDをサンプリング
        target_swarm_id = tf.random.categorical(tf.math.log(target_swarm_probs + 1e-8), 1)
        target_swarm_log_prob = tf.reduce_sum(
            tf.math.log(target_swarm_probs + 1e-8) * tf.one_hot(target_swarm_id, target_swarm_probs.shape[-1]), axis=-1
        )
        
        return action_type, target_swarm_id, action_type_log_prob, target_swarm_log_prob
    
    @staticmethod
    def sample_thresholds(threshold_mu, threshold_std):
        """
        閾値パラメータ（branch_threshold, integration_threshold）をサンプリング
        Args:
            threshold_mu (tf.Tensor): 閾値の平均
            threshold_std (tf.Tensor): 閾値の標準偏差
        Returns:
            thresholds (tf.Tensor): サンプリングされた閾値
            epsilon (tf.Tensor): サンプリングに使用されたノイズ
        """
        epsilon = tf.random.normal(shape=threshold_mu.shape)
        thresholds = threshold_mu + threshold_std * epsilon
        
        # 0-1の範囲にクリップ
        thresholds_clipped = tf.clip_by_value(thresholds, 0.0, 1.0)
        
        return thresholds_clipped, epsilon
    
    @staticmethod
    def compute_log_prob(mu, std, action):
        """
        アクションの対数確率を計算
        Args:
            mu (tf.Tensor): 平均
            std (tf.Tensor): 標準偏差
            action (tf.Tensor): 実行されたアクション
        Returns:
            log_prob (tf.Tensor): アクションの対数確率
        """
        var = tf.square(std)
        log_std = tf.math.log(std + 1e-8)
        log_prob = -0.5 * tf.reduce_sum(
            ((action - mu) ** 2) / (var + 1e-8) + 2.0 * log_std + tf.math.log(2.0 * np.pi),
            axis=-1
        )
        return log_prob
    
    @staticmethod
    def compute_action_log_prob(action_type_probs, target_swarm_probs, action_type, target_swarm_id):
        """
        アクション（action_type, target_swarm_id）の対数確率を計算
        Args:
            action_type_probs (tf.Tensor): アクションタイプの確率分布
            target_swarm_probs (tf.Tensor): ターゲット群の確率分布
            action_type (tf.Tensor): 実行されたアクションタイプ
            target_swarm_id (tf.Tensor): 実行されたターゲット群ID
        Returns:
            log_prob (tf.Tensor): アクションの対数確率
        """
        # アクションタイプの対数確率
        action_type_log_prob = tf.reduce_sum(
            tf.math.log(action_type_probs + 1e-8) * tf.one_hot(action_type, 3), axis=-1
        )
        
        # ターゲット群の対数確率
        target_swarm_log_prob = tf.reduce_sum(
            tf.math.log(target_swarm_probs + 1e-8) * tf.one_hot(target_swarm_id, target_swarm_probs.shape[-1]), axis=-1
        )
        
        # 総対数確率
        log_prob = action_type_log_prob + target_swarm_log_prob
        
        return log_prob
    
    def compute_loss(self, states, threshold_actions, action_types, target_swarm_ids, returns, advantages):
        """
        Actor-Critic損失を計算
        Args:
            states (tf.Tensor): 入力状態
            threshold_actions (tf.Tensor): 閾値パラメータアクション
            action_types (tf.Tensor): 実行されたアクションタイプ
            target_swarm_ids (tf.Tensor): 実行されたターゲット群ID
            returns (tf.Tensor): リターン
            advantages (tf.Tensor): アドバンテージ
        Returns:
            total_loss (tf.Tensor): 総損失
            actor_loss (tf.Tensor): Actor損失
            critic_loss (tf.Tensor): Critic損失
        """
        threshold_mu, threshold_std, action_type_probs, target_swarm_probs, values = self(states)
        
        # 閾値パラメータの対数確率
        threshold_log_probs = self.compute_log_prob(threshold_mu, threshold_std, threshold_actions)
        
        # アクションの対数確率
        action_log_probs = self.compute_action_log_prob(action_type_probs, target_swarm_probs, action_types, target_swarm_ids)
        
        # Actor損失（閾値パラメータとアクションの両方）
        actor_loss = -tf.reduce_mean(threshold_log_probs * advantages + action_log_probs * advantages)
        
        # Critic損失（価値関数）
        critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values, axis=-1)))
        
        # エントロピーボーナス（探索促進）
        threshold_entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * tf.square(threshold_std)) + 0.5)
        action_type_entropy = -tf.reduce_mean(tf.reduce_sum(action_type_probs * tf.math.log(action_type_probs + 1e-8), axis=-1))
        target_swarm_entropy = -tf.reduce_mean(tf.reduce_sum(target_swarm_probs * tf.math.log(target_swarm_probs + 1e-8), axis=-1))
        entropy_bonus = 0.01 * (threshold_entropy + action_type_entropy + target_swarm_entropy)
        
        # 総損失
        total_loss = actor_loss + 0.5 * critic_loss - entropy_bonus
        
        return total_loss, actor_loss, critic_loss
    
    @tf.function
    def train_step(self, optimizer, states, threshold_actions, action_types, target_swarm_ids, returns, advantages):
        """
        1ステップの学習を実行
        Args:
            optimizer (tf.keras.optimizers.Optimizer): オプティマイザー
            states (tf.Tensor): 入力状態
            threshold_actions (tf.Tensor): 閾値パラメータアクション
            action_types (tf.Tensor): 実行されたアクションタイプ
            target_swarm_ids (tf.Tensor): 実行されたターゲット群ID
            returns (tf.Tensor): リターン
            advantages (tf.Tensor): アドバンテージ
        Returns:
            loss (tf.Tensor): 総損失
            actor_loss (tf.Tensor): Actor損失
            critic_loss (tf.Tensor): Critic損失
        """
        with tf.GradientTape() as tape:
            loss, actor_loss, critic_loss = self.compute_loss(states, threshold_actions, action_types, target_swarm_ids, returns, advantages)
        
        grads = tape.gradient(loss, self.trainable_variables)
        if grads is not None:
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss, actor_loss, critic_loss
    
    def get_branch_threshold(self):
        """分岐閾値を取得"""
        # 現在の閾値パラメータを取得（簡易版）
        return 0.5  # 実際の実装では学習済みパラメータから取得
    
    def get_integration_threshold(self):
        """統合閾値を取得"""
        # 現在の閾値パラメータを取得（簡易版）
        return 0.3  # 実際の実装では学習済みパラメータから取得
    
    def set_branch_threshold(self, value):
        """分岐閾値を設定"""
        # 実際の実装では学習済みパラメータを設定
        pass
    
    def set_integration_threshold(self, value):
        """統合閾値を設定"""
        # 実際の実装では学習済みパラメータを設定
        pass
    
    def get_config(self):
        """モデル設定を取得"""
        return {
            "input_dim": self.input_dim,
            "max_swarms": self.max_swarms
        }
    
    @classmethod
    def from_config(cls, config):
        """設定からモデルを作成"""
        return cls(**config) 