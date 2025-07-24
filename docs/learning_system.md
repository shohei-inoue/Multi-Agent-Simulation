# 学習システム (Learning System)

## 概要

学習システムは、群ロボットシステムにおける強化学習を統合的に管理するシステムです。SystemAgent と SwarmAgent の協調学習、分岐・統合時の学習情報の継承・統合、VFH-Fuzzy アルゴリズムのパラメータ最適化を実現します。

## 設計思想

### 1. 階層的学習

- SystemAgent: 高レベル制御の学習
- SwarmAgent: 低レベル行動の学習
- 協調的な学習の実現

### 2. 学習情報の継承

- 分岐時の学習パラメータの引き継ぎ
- 統合時の学習情報の統合
- 知識の効率的な移転

### 3. 適応的パラメータ調整

- VFH-Fuzzy アルゴリズムの動的最適化
- 衝突回避の学習
- 探査効率の向上

## アーキテクチャ

### 1. 学習パラメータ管理

```python
@dataclass
class LearningParameter:
    type: Literal["a2c"]
    model: Literal["actor-critic"]
    optimizer: Literal["adam"]
    gamma: float
    learningLate: float
    nStep: int
    inherit_learning_info: bool = True
    merge_learning_info: bool = True
```

### 2. エージェント別学習設定

#### SystemAgent 学習設定

```python
# SystemAgentParam内の学習設定
learningParameter: LearningParameter = LearningParameter(
    type="a2c",
    model="actor-critic",
    optimizer="adam",
    gamma=0.99,
    learningLate=0.001,
    nStep=10,
    inherit_learning_info=True,
    merge_learning_info=True
)
```

#### SwarmAgent 学習設定

```python
# SwarmAgentParam内の学習設定
learningParameter: LearningParameter = LearningParameter(
    type="a2c",
    model="actor-critic",
    optimizer="adam",
    gamma=0.99,
    learningLate=0.001,
    nStep=5,
    inherit_learning_info=True,
    merge_learning_info=True
)
```

## 学習アルゴリズム

### 1. Actor-Critic (A2C)

```python
class ActorCritic:
    """Actor-Critic model for reinforcement learning"""

    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.actor = self._build_actor(state_size, action_size)
        self.critic = self._build_critic(state_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state):
        """Get action from current state"""
        state = np.array([state])
        action_probs = self.actor.predict(state)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs

    def update(self, states, actions, rewards, next_states, dones):
        """Update actor and critic networks"""
        # Actor-Critic更新ロジック
        pass
```

### 2. VFH-Fuzzy パラメータ学習

```python
class VFHFuzzyLearning:
    """Learning system for VFH-Fuzzy parameters"""

    def update_params(self, th, k_e, k_c):
        """Update VFH-Fuzzy parameters based on learning"""
        self.th = th      # 閾値パラメータ
        self.k_e = k_e    # 探査抑制パラメータ
        self.k_c = k_c    # 衝突抑制パラメータ

    def get_learning_reward(self, collision_flag, exploration_improvement):
        """Calculate learning reward"""
        reward = 0.0

        # 衝突ペナルティ
        if collision_flag:
            reward -= 5.0

        # 探査向上報酬
        reward += exploration_improvement * 10.0

        return reward
```

## 学習フロー

### 1. SwarmAgent 学習フロー

```python
def swarm_agent_learning_flow(self, state, action, reward, next_state):
    """SwarmAgent learning flow"""

    # 1. 状態の観測
    observation = self.get_observation(state)

    # 2. 行動の選択
    theta, valid_directions = self.algorithm.policy(observation, self.learning_params)

    # 3. 環境での実行
    next_state, reward, done = self.env.step({"theta": theta})

    # 4. 学習パラメータの更新
    self.update_learning_params(reward)

    # 5. VFH-Fuzzyパラメータの調整
    self.algorithm.update_params(self.th, self.k_e, self.k_c)
```

### 2. SystemAgent 学習フロー

```python
def system_agent_learning_flow(self, swarm_state):
    """SystemAgent learning flow"""

    # 1. システム状態の観測
    system_observation = self.get_system_observation(swarm_state)

    # 2. 分岐・統合の判断
    branch_action = self.check_branch(system_observation)
    integration_action = self.check_integration(system_observation)

    # 3. 行動の実行
    if branch_action:
        self.execute_branch(branch_action)
    elif integration_action:
        self.execute_integration(integration_action)

    # 4. システム報酬の計算
    system_reward = self.calculate_system_reward(swarm_state)

    # 5. 学習パラメータの更新
    self.update_learning_params(system_reward)
```

## 学習情報の継承・統合

### 1. 分岐時の学習継承

```python
def inherit_learning_info(self, source_swarm, new_swarm):
    """Inherit learning information during branching"""

    # 学習パラメータの継承
    if hasattr(source_swarm, 'learning_params'):
        new_swarm.learning_params = source_swarm.learning_params.copy()

    # VFH-Fuzzyパラメータの継承
    if hasattr(source_swarm, 'algorithm'):
        new_swarm.algorithm.th = source_swarm.algorithm.th
        new_swarm.algorithm.k_e = source_swarm.algorithm.k_e
        new_swarm.algorithm.k_c = source_swarm.algorithm.k_c

    # 経験バッファの共有
    if hasattr(source_swarm, 'experience_buffer'):
        shared_experience = source_swarm.experience_buffer.sample(
            size=min(100, len(source_swarm.experience_buffer))
        )
        new_swarm.experience_buffer.extend(shared_experience)
```

### 2. 統合時の学習統合

```python
def merge_learning_info(self, source_swarm, target_swarm):
    """Merge learning information during integration"""

    # 重み付き平均による統合
    source_weight = 0.3
    target_weight = 0.7

    # 学習パラメータの統合
    for param_name in ['th', 'k_e', 'k_c']:
        if hasattr(source_swarm.algorithm, param_name) and hasattr(target_swarm.algorithm, param_name):
            source_value = getattr(source_swarm.algorithm, param_name)
            target_value = getattr(target_swarm.algorithm, param_name)

            merged_value = (source_value * source_weight + target_value * target_weight)
            setattr(target_swarm.algorithm, param_name, merged_value)

    # 経験バッファの統合
    if hasattr(source_swarm, 'experience_buffer') and hasattr(target_swarm, 'experience_buffer'):
        shared_experience = source_swarm.experience_buffer.sample(
            size=min(50, len(source_swarm.experience_buffer))
        )
        target_swarm.experience_buffer.extend(shared_experience)
```

## 報酬設計

### 1. SwarmAgent 報酬

```python
def calculate_swarm_reward(self, collision_flag, exploration_improvement, movement_distance):
    """Calculate reward for SwarmAgent"""
    reward = 0.0

    # 衝突ペナルティ
    if collision_flag:
        reward -= 5.0

    # 探査向上報酬
    reward += exploration_improvement * 10.0

    # 移動距離報酬
    reward += movement_distance * 0.1

    return reward
```

### 2. SystemAgent 報酬

```python
def calculate_system_reward(self, swarm_state):
    """Calculate reward for SystemAgent"""
    reward = 0.0

    # 探査効率報酬
    exploration_efficiency = swarm_state.get('exploration_efficiency', 0.0)
    reward += exploration_efficiency * 10.0

    # 群数バランス報酬
    swarm_count_balance = self.calculate_swarm_count_balance()
    reward += swarm_count_balance * 2.0

    # 移動性スコア報酬
    mobility_score = swarm_state.get('mobility_score', 0.0)
    reward += mobility_score * 5.0

    # 学習移転成功報酬
    if self.learning_transfer_success:
        reward += 3.0

    # システム安定性報酬
    system_stability = self.calculate_system_stability()
    reward += system_stability * 2.0

    return reward
```

## 学習パラメータの最適化

### 1. VFH-Fuzzy パラメータ

#### 閾値パラメータ (th)

```python
def update_threshold_parameter(self, reward, current_th):
    """Update threshold parameter based on reward"""
    if reward > 0:
        # 良い報酬の場合、閾値を上げてより厳格にする
        new_th = current_th + 0.01
    else:
        # 悪い報酬の場合、閾値を下げてより寛容にする
        new_th = current_th - 0.01

    return np.clip(new_th, 0.1, 0.9)
```

#### 探査抑制パラメータ (k_e)

```python
def update_exploration_parameter(self, reward, current_k_e):
    """Update exploration parameter based on reward"""
    if reward > 0:
        # 良い報酬の場合、探査を促進
        new_k_e = current_k_e - 0.1
    else:
        # 悪い報酬の場合、探査を抑制
        new_k_e = current_k_e + 0.1

    return np.clip(new_k_e, 0.1, 5.0)
```

#### 衝突抑制パラメータ (k_c)

```python
def update_collision_parameter(self, collision_flag, current_k_c):
    """Update collision parameter based on collision"""
    if collision_flag:
        # 衝突が発生した場合、衝突抑制を強化
        new_k_c = current_k_c + 0.2
    else:
        # 衝突がない場合、衝突抑制を緩和
        new_k_c = current_k_c - 0.05

    return np.clip(new_k_c, 0.1, 5.0)
```

### 2. 学習率の動的調整

```python
def adjust_learning_rate(self, episode, base_learning_rate=0.001):
    """Dynamically adjust learning rate"""
    # エピソード数に応じて学習率を減衰
    decay_rate = 0.995
    adjusted_lr = base_learning_rate * (decay_rate ** episode)

    return max(adjusted_lr, 0.0001)  # 最小学習率を保証
```

## 学習の監視と評価

### 1. 学習メトリクス

```python
class LearningMetrics:
    """Learning metrics for monitoring"""

    def __init__(self):
        self.episode_rewards = []
        self.exploration_rates = []
        self.collision_rates = []
        self.parameter_changes = []

    def update_metrics(self, episode, reward, exploration_rate, collision_rate, params):
        """Update learning metrics"""
        self.episode_rewards.append(reward)
        self.exploration_rates.append(exploration_rate)
        self.collision_rates.append(collision_rate)
        self.parameter_changes.append(params)

    def get_learning_progress(self):
        """Get learning progress summary"""
        return {
            'avg_reward': np.mean(self.episode_rewards[-100:]),  # 最近100エピソード
            'avg_exploration_rate': np.mean(self.exploration_rates[-100:]),
            'avg_collision_rate': np.mean(self.collision_rates[-100:]),
            'parameter_trend': self.analyze_parameter_trend()
        }
```

### 2. 学習の可視化

```python
def visualize_learning_progress(self, metrics):
    """Visualize learning progress"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 報酬の推移
    axes[0, 0].plot(metrics.episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')

    # 探査率の推移
    axes[0, 1].plot(metrics.exploration_rates)
    axes[0, 1].set_title('Exploration Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Exploration Rate')

    # 衝突率の推移
    axes[1, 0].plot(metrics.collision_rates)
    axes[1, 0].set_title('Collision Rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Collision Rate')

    # パラメータの推移
    params = np.array(metrics.parameter_changes)
    axes[1, 1].plot(params[:, 0], label='th')
    axes[1, 1].plot(params[:, 1], label='k_e')
    axes[1, 1].plot(params[:, 2], label='k_c')
    axes[1, 1].set_title('Parameter Changes')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
```

## 設定パラメータ

### 1. 学習パラメータ

```python
# 本格的な学習設定
episodeNum: int = 100
maxStepsPerEpisode: int = 50

# SwarmAgent学習設定
swarm_learning_params = LearningParameter(
    type="a2c",
    model="actor-critic",
    optimizer="adam",
    gamma=0.99,
    learningLate=0.001,
    nStep=5,
    inherit_learning_info=True,
    merge_learning_info=True
)

# SystemAgent学習設定
system_learning_params = LearningParameter(
    type="a2c",
    model="actor-critic",
    optimizer="adam",
    gamma=0.99,
    learningLate=0.001,
    nStep=10,
    inherit_learning_info=True,
    merge_learning_info=True
)
```

### 2. VFH-Fuzzy 初期パラメータ

```python
# VFH-Fuzzy初期パラメータ
initial_th = 0.3    # 初期閾値
initial_k_e = 1.0   # 初期探査抑制パラメータ
initial_k_c = 1.0   # 初期衝突抑制パラメータ
```

## 使用例

### 1. 基本的な学習設定

```python
# 学習パラメータの設定
learning_params = LearningParameter(
    type="a2c",
    model="actor-critic",
    optimizer="adam",
    gamma=0.99,
    learningLate=0.001,
    nStep=5
)

# SwarmAgentの作成
swarm_agent = SwarmAgent(
    swarm_id=1,
    learning_params=learning_params,
    algorithm="vfh_fuzzy"
)
```

### 2. 学習の実行

```python
# 学習ループ
for episode in range(100):
    state = env.reset()

    for step in range(50):
        # 行動の選択
        action = swarm_agent.get_action(state)

        # 環境での実行
        next_state, reward, done = env.step(action)

        # 学習の更新
        swarm_agent.update_learning_params(reward)

        state = next_state
        if done:
            break

    # エピソード終了時の処理
    swarm_agent.end_episode()
```

## パフォーマンス最適化

### 1. 学習効率の向上

- 経験リプレイバッファの活用
- 優先度付きサンプリング
- バッチ学習の最適化

### 2. メモリ効率

- 不要な経験データの削除
- 学習パラメータの効率的な保存
- メモリリークの防止

### 3. 計算効率

- GPU 活用の最適化
- 並列学習の実装
- 学習更新の非同期化

## 今後の拡張

### 1. 新しい学習アルゴリズム

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

### 2. メタ学習の導入

- 環境適応の高速化
- 転移学習の強化
- 少数サンプル学習

### 3. 分散学習の実装

- マルチエージェント学習
- フェデレーテッド学習
- 分散最適化
