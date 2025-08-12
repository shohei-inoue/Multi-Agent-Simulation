# 学習システムとアルゴリズム構成 (Learning System and Algorithm Architecture)

## 概要

本システムは、群ロボットシステムにおける強化学習と VFH-Fuzzy アルゴリズムを統合的に管理するシステムです。SystemAgent と SwarmAgent の協調学習、分岐・統合時の学習情報の継承・統合、VFH-Fuzzy アルゴリズムのパラメータ最適化を実現します。

## システム構成

### 1. エージェント階層

```
┌─────────────────┐
│   SystemAgent   │  ← 高レベル制御（分岐・統合判定）
│   (学習あり/なし) │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │ SwarmAgent│  ← 低レベル行動（移動方向決定）
    │(学習あり/なし)│
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ VFH-Fuzzy │  ← 行動決定アルゴリズム
    │ Algorithm │
    └───────────┘
```

### 2. Config 別の設定

| Config | SystemAgent | SwarmAgent | 分岐・統合 | 学習モード         |
| ------ | ----------- | ---------- | ---------- | ------------------ |
| **A**  | 学習なし    | 学習なし   | 無効       | 固定パラメータ     |
| **B**  | 学習なし    | 学習あり   | 無効       | 学習済みパラメータ |
| **C**  | 学習あり    | 学習なし   | 有効       | 固定パラメータ     |
| **D**  | 学習あり    | 学習あり   | 有効       | 学習済みパラメータ |

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
```

### 2. 学習パラメータ管理

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

### 3. エージェント別学習設定

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

## VFH-Fuzzy アルゴリズム

### 1. 基本パラメータ

#### アルゴリズム定数

- `BIN_SIZE_DEG`: ビンのサイズ（度）- デフォルト: 10 度
- `BIN_NUM`: ビン数 - デフォルト: 36 個
- `ANGLES`: 角度配列 - 0 から 2π まで均等分布

#### 学習可能パラメータ

- `th`: 抑制のしきい値（しきい値以下の drivability は抑制）
- `k_e`: 探査逆方向の抑制強度
- `k_c`: 衝突方向の抑制強度
- `α`: ソフト抑制の鋭さ（固定: α=10.0）

### 2. ヒストグラム生成

#### 2.1 走行可能性ヒストグラム（Drivability）

```python
def get_obstacle_density_histogram(self):
    """
    障害物密度に基づく走行可能性分布を生成
    """
    histogram = np.zeros(self.BIN_NUM)

    # 前回衝突した方向を避ける
    if self.last_collision_theta is not None:
        collision_bin = int(self.last_collision_theta * self.BIN_NUM / (2 * np.pi)) % self.BIN_NUM
        for i in range(self.BIN_NUM):
            distance = min(abs(i - collision_bin), abs(i - collision_bin + self.BIN_NUM), abs(i - collision_bin - self.BIN_NUM))
            if distance < 4:  # 前回衝突した方向の周辺4bin（40度範囲）を低評価
                histogram[i] = 0.05
            else:
                histogram[i] = 1.0
        return histogram

    # フォロワーの衝突データのみを使用
    histogram = np.ones(self.BIN_NUM, dtype=np.float32)

    for azimuth, distance in self.follower_collision_data:
        if distance < 1e-6:
            continue

        azimuth_deg = azimuth % 360.0
        bin_index = int(azimuth_deg) // self.BIN_SIZE_DEG

        obstacle_weight = 1.0 / (distance + 1e-3)
        histogram[bin_index] -= obstacle_weight

    histogram = np.maximum(histogram, 0.01)
    histogram += 1e-6
    histogram /= np.sum(histogram)

    return histogram
```

#### 2.2 探査向上性ヒストグラム（Exploration Improvement）

```python
def get_exploration_improvement_histogram(self) -> np.ndarray:
    """
    探査向上性に基づく探索欲求分布を生成
    """
    histogram = np.ones(self.BIN_NUM)

    def apply_direction_weight_gauss(base_azimuth: float, k: float, sharpness: float = 10.0):
        """
        base_azimuth方向を中心にガウス分布で【抑制】をかける
        """
        for i in range(self.BIN_NUM):
            angle = 2 * np.pi * i / self.BIN_NUM
            angle_diff = abs(angle - base_azimuth)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            decay = 1 - (1 - k) * np.exp(-sharpness * angle_diff ** 2)
            histogram[i] *= decay

    def apply_direction_weight_von(base_azimuth: float, kappa: float):
        """
        von Mises分布による方向重みの適用
        """
        for i in range(self.BIN_NUM):
            angle = 2 * np.pi * i / self.BIN_NUM
            angle_diff = abs(angle - base_azimuth)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            decay = np.exp(kappa * np.cos(angle_diff)) / (2 * np.pi * i0(kappa))
            histogram[i] *= decay

    # 探査逆方向（過去方向）の抑制
    if hasattr(self, 'agent_azimuth') and self.agent_azimuth is not None:
        apply_direction_weight_gauss(self.agent_azimuth, self.k_e)

    # 衝突方向の抑制
    if hasattr(self, 'agent_azimuth') and self.agent_azimuth is not None:
        apply_direction_weight_von(self.agent_azimuth, self.k_c)

    histogram += 1e-6
    histogram /= np.sum(histogram)

    return histogram
```

### 3. ファジィ推論

#### 3.1 ソフト抑制付きファジィ推論

```python
def get_final_result_histogram(self, drivability, exploration_improvement) -> np.ndarray:
    """
    ファジィ推論に基づき、方向ごとのスコアを統合したヒストグラムを返す
    """
    fused_score = np.zeros(self.BIN_NUM, dtype=np.float32)

    # 元々の設計パラメータ
    alpha = 10.0  # ソフト抑制の鋭さ

    for bin in range(self.BIN_NUM):
        drive_val = drivability[bin]
        explore_val = exploration_improvement[bin]

        # ソフト抑制係数
        suppression = 1 / (1 + np.exp(-alpha * (drive_val - self.th)))

        # ファジィ積推論
        fused_score[bin] = suppression * drive_val * explore_val

    fused_score += 1e-6
    fused_score /= np.sum(fused_score)

    return fused_score
```

### 4. 方向選択

#### 4.1 重み付けサンプリング

```python
def select_final_direction_by_weighted_sampling(self, result: np.ndarray) -> float:
    """
    四分位数ベースの重み付けサンプリング
    """
    q1, q2, q3 = np.quantile(result, [0.25, 0.5, 0.75])

    bins_very_good = [i for i, score in enumerate(result) if score >= q3]
    bins_good = [i for i, score in enumerate(result) if q2 <= score < q3]
    bins_okay = [i for i, score in enumerate(result) if q1 <= score < q2]

    bins = bins_very_good + bins_good + bins_okay

    if not bins:
        # フォールバック: ランダム方向
        fallback_theta = np.random.uniform(0, 2 * np.pi)
        return fallback_theta

    # 元々の設計の重み付け
    score_q1 = 0.6 if bins_very_good else 0.0  # Very Good: 60%
    score_q2 = 0.25 if bins_good else 0.0      # Good: 25%
    score_q3 = 0.15 if bins_okay else 0.0      # Okay: 15%

    total_score = score_q1 + score_q2 + score_q3

    # 正規化
    score_q1 /= total_score if total_score > 0 else 1.0
    score_q2 /= total_score if total_score > 0 else 1.0
    score_q3 /= total_score if total_score > 0 else 1.0

    # 重みに従って構築
    weights = []
    if bins_very_good:
        weights += [score_q1 / len(bins_very_good)] * len(bins_very_good)
    if bins_good:
        weights += [score_q2 / len(bins_good)] * len(bins_good)
    if bins_okay:
        weights += [score_q3 / len(bins_okay)] * len(bins_okay)

    selected_bin = np.random.choice(bins, p=weights)
    selected_angle = 2 * np.pi * selected_bin / self.BIN_NUM

    return selected_angle
```

## 学習モードの違い

### 1. 学習なしモード（Config A & C）

```python
# SwarmAgent.get_action() - 学習なしモード
if not self.isLearning or self.model is None:
    # デフォルトのパラメータを使用してアルゴリズムを実行
    default_params = np.array([0.5, 10.0, 5.0])  # th, k_e, k_c
    theta, valid_directions = self.algorithm.policy(state, default_params)

    # 分岐・統合条件の判定
    follower_scores = state.get("follower_mobility_scores", [])
    follower_count = len(follower_scores)
    avg_mobility = np.mean(follower_scores) if follower_count > 0 else 0.0

    # SystemAgentから閾値を取得
    branch_threshold = 0.5  # デフォルト値
    integration_threshold = 0.3  # デフォルト値

    should_branch = (
        follower_count >= 3 and
        valid_directions and len(valid_directions) >= 2 and
        avg_mobility >= branch_threshold
    )
    should_integrate = avg_mobility < integration_threshold

    # 分岐・統合判定
    if self.system_agent and should_branch:
        self.system_agent.check_branch(system_state)
    if self.system_agent and should_integrate:
        self.system_agent.check_integration(system_state)

    return {"theta": theta}, {
        'theta': theta,
        'valid_directions': valid_directions
    }
```

### 2. 学習ありモード（Config B & D）

```python
# SwarmAgent.get_action() - 学習ありモード
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

return {"theta": theta_value}, {
    'theta': float(action_theta) if not isinstance(action_theta, tuple) else float(action_theta[0]),
    'learning_params': learning_params.numpy().tolist() if hasattr(learning_params, 'numpy') else list(learning_params),
    'valid_directions': valid_directions,
    'value': float(value)
}
```

## 分岐・統合システム

### 1. 分岐条件

```python
def check_branch(self, system_state: Dict[str, Any]) -> bool:
    """
    分岐条件をチェックし、条件を満たせば分岐を実行
    """
    # 分岐が無効になっている場合は分岐を実行しない
    if not self.branchCondition.branch_enabled:
        return False

    # クールダウンチェック
    current_time = time.time()
    if current_time - self.last_branch_time < self.branchCondition.swarm_creation_cooldown:
        return False

    # 分岐条件チェック
    follower_count = system_state.get("follower_count", 0)
    valid_directions = system_state.get("valid_directions", [])
    avg_mobility = system_state.get("avg_mobility", 0.0)

    should_branch = (
        follower_count >= 3 and
        len(valid_directions) >= 2 and
        avg_mobility >= self.branch_threshold
    )

    if should_branch:
        # 分岐実行
        self._execute_branch(system_state)
        self.last_branch_time = current_time
        return True

    return False
```

### 2. 統合条件

```python
def check_integration(self, system_state: Dict[str, Any]) -> bool:
    """
    統合条件をチェックし、条件を満たせば統合を実行
    """
    # 統合が無効になっている場合は統合を実行しない
    if not self.integrationCondition.integration_enabled:
        return False

    # クールダウンチェック
    current_time = time.time()
    if current_time - self.last_integration_time < self.integrationCondition.swarm_merge_cooldown:
        return False

    # 統合条件チェック
    avg_mobility = system_state.get("avg_mobility", 0.0)
    swarm_count = system_state.get("swarm_count", 1)

    base_condition = (
        swarm_count >= self.integrationCondition.min_swarms_for_integration and
        (avg_mobility < self.integration_threshold or swarm_count >= 5)
    )

    # 探査領域の重複チェック
    has_overlapping_swarms = False
    if base_condition and hasattr(self.env, 'check_exploration_area_overlap'):
        swarm_id = system_state.get("swarm_id")
        if swarm_id is not None:
            for target_swarm_id in self.swarm_agents.keys():
                if target_swarm_id != swarm_id:
                    if self.env.check_exploration_area_overlap(swarm_id, target_swarm_id):
                        has_overlapping_swarms = True
                        break

    # 統合の確率を制御（20%の確率で統合を実行）
    should_integrate = base_condition and has_overlapping_swarms and np.random.random() < 0.2

    if should_integrate:
        # 統合実行
        self._execute_integration(system_state)
        self.last_integration_time = current_time
        return True

    return False
```

## 学習情報の継承・統合

### 1. 分岐時の学習情報継承

```python
def _inherit_learning_info(self, source_swarm_id: int) -> Dict[str, Any]:
    """
    分岐元の学習情報を継承
    """
    if source_swarm_id in self.learning_history:
        source_info = self.learning_history[source_swarm_id].copy()

        # 学習パラメータの微調整
        if 'learning_params' in source_info:
            source_info['learning_params'] = self._adjust_learning_params(
                source_info['learning_params']
            )

        return source_info
    else:
        return self._get_default_learning_info()
```

### 2. 統合時の学習情報統合

```python
def _merge_learning_info(self, source_swarm_id: int, target_swarm_id: int):
    """
    統合時の学習情報を統合
    """
    if source_swarm_id in self.learning_history and target_swarm_id in self.learning_history:
        source_info = self.learning_history[source_swarm_id]
        target_info = self.learning_history[target_swarm_id]

        # 学習パラメータの平均化
        merged_params = self._average_learning_params(
            source_info.get('learning_params', []),
            target_info.get('learning_params', [])
        )

        # 統合された情報を保存
        target_info['learning_params'] = merged_params
        target_info['merged_from'] = source_swarm_id

        # ソースの学習情報を削除
        del self.learning_history[source_swarm_id]
```

## パフォーマンス最適化

### 1. メモリ効率

- 学習履歴の定期的なクリーンアップ
- 不要なデータの自動削除
- メモリ使用量の監視

### 2. 計算効率

- バッチ処理による学習の効率化
- 並列処理によるシミュレーション加速
- キャッシュ機能による重複計算の回避

### 3. 学習効率

- 適応的学習率調整
- 早期停止条件の実装
- 学習進捗の可視化

## 今後の改善方向

### 1. 動的パラメータ調整

- 環境状態に応じたリアルタイム調整
- 探査進捗に基づく動的閾値調整
- 群の状態に応じた制御パラメータ調整

### 2. 階層的学習

- 個体レベルと群レベルの同時最適化
- 群制御と個体行動の協調学習
- マルチスケール最適化

### 3. メタ学習

- 様々な環境への適応能力の向上
- 環境特性の自動認識
- 汎用的な最適化戦略の学習

### 4. 多群最適化

- 複数群の協調的最適化
- 群間の競合と協調のバランス
- 大規模群システムの最適化

### 5. 予測的群制御

- 将来の動きやすさ予測
- 予防的な群再編成
- 長期的な効率最適化
