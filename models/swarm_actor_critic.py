"""
SwarmAgent用のActor-Criticモデル
アクション出力はtheta（移動方向）のみ
学習ではth, k_e, k_cパラメータを最適化
"""

import tensorflow as tf
import numpy as np
import keras

SCALE_MAX = 50.0


class SwarmActorCritic(keras.Model):
    """
    SwarmAgent用のActor-Criticモデル
    アクション出力: theta（移動方向）
    学習パラメータ: th, k_e, k_c（アルゴリズム用）
    """
    
    def __init__(self, input_dim: int):
        """
        SwarmActorCriticモデルを初期化
        Args:
            input_dim (int): 入力状態の次元数
        """
        super().__init__()
        
        # 共有レイヤー
        self.shared_dense1 = keras.layers.Dense(128, activation='relu')
        self.shared_dense2 = keras.layers.Dense(64, activation='relu')
        
        # 学習用パラメータ出力（th, k_e, k_c）
        self.learning_params_output = keras.layers.Dense(3)  # th, k_e, k_c
        self.learning_params_log_std = tf.Variable(initial_value=tf.zeros(3), trainable=True)
        
        # アクション出力（theta）
        self.theta_output = keras.layers.Dense(1)  # theta
        self.theta_log_std = tf.Variable(initial_value=tf.zeros(1), trainable=True)
        
        # Critic（価値関数）出力
        self.critic_output = keras.layers.Dense(1)
        
        self.input_dim = input_dim
    
    def call(self, state):
        """
        フォワードパス
        Args:
            state (tf.Tensor): 入力状態
        Returns:
            learning_mu (tf.Tensor): 学習パラメータの平均（th, k_e, k_c）
            learning_std (tf.Tensor): 学習パラメータの標準偏差
            theta_mu (tf.Tensor): thetaの平均
            theta_std (tf.Tensor): thetaの標準偏差
            value (tf.Tensor): 状態価値
        """
        x = self.shared_dense1(state)
        x = self.shared_dense2(x)
        
        # 学習用パラメータ（th, k_e, k_c）
        raw_learning_params = self.learning_params_output(x)
        th = tf.nn.sigmoid(raw_learning_params[:, 0:1])  # 0-1
        k_e_c = tf.nn.sigmoid(raw_learning_params[:, 1:3]) * SCALE_MAX  # 0-SCALE_MAX
        learning_mu = tf.concat([th, k_e_c], axis=-1)  # shape = (batch, 3)
        learning_std = tf.exp(self.learning_params_log_std)
        
        # アクション出力（theta）
        raw_theta = self.theta_output(x)
        theta_mu = tf.nn.sigmoid(raw_theta) * 2 * np.pi  # 0-2π
        theta_std = tf.exp(self.theta_log_std)
        
        # 価値関数
        value = self.critic_output(x)
        
        return learning_mu, learning_std, theta_mu, theta_std, value
    
    @staticmethod
    def sample_action(theta_mu, theta_std):
        """
        アクション（theta）をサンプリング
        Args:
            theta_mu (tf.Tensor): thetaの平均
            theta_std (tf.Tensor): thetaの標準偏差
        Returns:
            theta (tf.Tensor): サンプリングされたtheta（0-2πにクリップ）
            epsilon (tf.Tensor): サンプリングに使用されたノイズ
        """
        epsilon = tf.random.normal(shape=theta_mu.shape)
        theta = theta_mu + theta_std * epsilon
        
        # 0-2πの範囲にクリップ
        theta_clipped = tf.clip_by_value(theta, 0.0, 2 * np.pi)
        
        return theta_clipped, epsilon
    
    @staticmethod
    def sample_learning_params(learning_mu, learning_std):
        """
        学習パラメータ（th, k_e, k_c）をサンプリング
        Args:
            learning_mu (tf.Tensor): 学習パラメータの平均
            learning_std (tf.Tensor): 学習パラメータの標準偏差
        Returns:
            learning_params (tf.Tensor): サンプリングされた学習パラメータ
            epsilon (tf.Tensor): サンプリングに使用されたノイズ
        """
        epsilon = tf.random.normal(shape=learning_mu.shape)
        learning_params = learning_mu + learning_std * epsilon
        
        # パラメータ範囲にクリップ
        th_clipped = tf.clip_by_value(learning_params[:, 0:1], 0.0, 1.0)
        k_e_c_clipped = tf.clip_by_value(learning_params[:, 1:3], 0.0, SCALE_MAX)
        learning_params_clipped = tf.concat([th_clipped, k_e_c_clipped], axis=-1)
        
        return learning_params_clipped, epsilon
    
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
    
    def compute_loss(self, states, learning_actions, theta_actions, returns, advantages):
        """
        Actor-Critic損失を計算
        Args:
            states (tf.Tensor): 入力状態
            learning_actions (tf.Tensor): 学習パラメータアクション（th, k_e, k_c）
            theta_actions (tf.Tensor): thetaアクション
            returns (tf.Tensor): リターン
            advantages (tf.Tensor): アドバンテージ
        Returns:
            total_loss (tf.Tensor): 総損失
            actor_loss (tf.Tensor): Actor損失
            critic_loss (tf.Tensor): Critic損失
        """
        learning_mu, learning_std, theta_mu, theta_std, values = self(states)
        
        # 学習パラメータの対数確率
        learning_log_probs = self.compute_log_prob(learning_mu, learning_std, learning_actions)
        
        # thetaの対数確率
        theta_log_probs = self.compute_log_prob(theta_mu, theta_std, theta_actions)
        
        # Actor損失（学習パラメータとthetaの両方）
        actor_loss = -tf.reduce_mean(learning_log_probs * advantages + theta_log_probs * advantages)
        
        # Critic損失（価値関数）
        critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values, axis=-1)))
        
        # エントロピーボーナス（探索促進）
        learning_entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * tf.square(learning_std)) + 0.5)
        theta_entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * tf.square(theta_std)) + 0.5)
        entropy_bonus = 0.01 * (learning_entropy + theta_entropy)
        
        # 総損失
        total_loss = actor_loss + 0.5 * critic_loss - entropy_bonus
        
        return total_loss, actor_loss, critic_loss
    
    @tf.function
    def train_step(self, optimizer, states, learning_actions, theta_actions, returns, advantages):
        """
        1ステップの学習を実行
        Args:
            optimizer (tf.keras.optimizers.Optimizer): オプティマイザー
            states (tf.Tensor): 入力状態
            learning_actions (tf.Tensor): 学習パラメータアクション
            theta_actions (tf.Tensor): thetaアクション
            returns (tf.Tensor): リターン
            advantages (tf.Tensor): アドバンテージ
        Returns:
            loss (tf.Tensor): 総損失
            actor_loss (tf.Tensor): Actor損失
            critic_loss (tf.Tensor): Critic損失
        """
        with tf.GradientTape() as tape:
            loss, actor_loss, critic_loss = self.compute_loss(states, learning_actions, theta_actions, returns, advantages)
        
        grads = tape.gradient(loss, self.trainable_variables)
        if grads is not None:
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss, actor_loss, critic_loss
    
    def get_config(self):
        """モデル設定を取得"""
        return {"input_dim": self.input_dim}
    
    @classmethod
    def from_config(cls, config):
        """設定からモデルを作成"""
        return cls(**config) 