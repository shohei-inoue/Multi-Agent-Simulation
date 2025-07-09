# 群ロボット全探索問題のパラメータ最適化

## 概要

群ロボットの全探索問題において、VFH-Fuzzy アルゴリズムのパラメータ（th, k_e, k_c）と群制御パラメータ（分岐・統合閾値）を最適化して探査率を最大化する戦略を実装しました。

## 最適化対象パラメータ

### 1. VFH-Fuzzy パラメータ

#### th（走行可能性閾値）

- **範囲**: 0.0 - 1.0
- **役割**: 走行可能性と探査向上性のバランスを制御
- **最適化目標**: 探査効率を最大化する閾値の設定

#### k_e（探査向上性重み）

- **範囲**: 0.0 - 50.0（拡張済み）
- **役割**: 未探査領域への誘導強度
- **最適化目標**: 効率的な探査パターンの生成

#### k_c（衝突回避重み）

- **範囲**: 0.0 - 50.0（拡張済み）
- **役割**: 障害物回避の強度
- **最適化目標**: 安全性と探査効率のバランス

### 2. 群制御パラメータ

#### branch_threshold（分岐閾値）

- **範囲**: 0.0 - 1.0
- **役割**: 群分岐の判断基準（動きやすさ指標の閾値）
- **最適化目標**: 最適な群サイズと分岐タイミングの決定

#### merge_threshold（統合閾値）

- **範囲**: 0.0 - 1.0
- **役割**: 群統合の判断基準
- **最適化目標**: 効率的な群統合による協調探査

## 最適化戦略

### 1. 探査率向上に特化した報酬設計

```python
# 群ロボット探査専用報酬
def calculate_swarm_exploration_reward(exploration_ratio, previous_ratio, reward_config,
                                     step_count, total_steps, explored_area, total_area):
    # 1. 探査率向上の報酬（最重要）
    if exploration_ratio > previous_ratio:
        improvement = exploration_ratio - previous_ratio
        base_reward = reward_config.exploration_base_reward
        improvement_reward = reward_config.exploration_improvement_weight * improvement
        nonlinear_bonus = improvement_reward * (1 + exploration_ratio)

    # 2. 時間効率報酬（早期完了ボーナス）
    if exploration_ratio > 0.8:
        time_efficiency = 1.0 - (step_count / total_steps)
        time_bonus = reward_config.time_efficiency_bonus * time_efficiency * exploration_ratio

    # 3. 目標達成報酬（全探索完了）
    if exploration_ratio >= 0.95:
        completion_bonus = reward_config.completion_bonus * exploration_ratio
```

### 2. 動きやすさ指標に基づく群制御

```python
def _determine_swarm_mode(self, state, sampled_params):
    """
    群分岐・統合のmodeを決定（動きやすさ指標ベース）
    """
    # フォロワーの動きやすさ指標を取得
    follower_mobility_scores = state.get('follower_mobility_scores', [])

    if len(follower_mobility_scores) >= 3:
        # 上位3台のフォロワーの平均動きやすさを計算
        avg_mobility = np.mean(follower_mobility_scores[:3])

        # 分岐閾値（学習可能）
        branch_threshold = sampled_params[3] if len(sampled_params) > 3 else 0.5

        if avg_mobility > branch_threshold:
            return 1  # 分岐モード
        else:
            return 2  # 統合モード

    return 0  # 通常モード
```

### 3. パラメータ範囲の拡張

```python
# パラメータ範囲を20.0 → 50.0に拡張
SCALE_MAX = 50.0

# より細かい制御が可能
th     = tf.nn.sigmoid(raw_mu[:, 0:1])               # 0-1
k_e_c  = tf.nn.sigmoid(raw_mu[:, 1:]) * SCALE_MAX    # 0-50
branch_threshold = tf.nn.sigmoid(raw_mu[:, 3:4])      # 0-1
merge_threshold = tf.nn.sigmoid(raw_mu[:, 4:5])       # 0-1
```

### 4. エントロピーボーナスの調整

```python
# 探査最適化のためのエントロピーボーナス増加
entropy_bonus = 0.02 * entropy  # 0.01 → 0.02
```

## 最適化設定例

### 1. 積極的探査設定

```python
from params.reward import SwarmExplorationRewardConfig

config = SwarmExplorationRewardConfig(
    exploration_base_reward=15.0,        # 高い基本報酬
    exploration_improvement_weight=80.0,  # 非常に高い改善重み
    time_efficiency_bonus=30.0,          # 高い時間効率ボーナス
    completion_bonus=150.0,              # 高い完了ボーナス
    revisit_penalty_weight=8.0           # 強い再訪問ペナルティ
)

# 群制御パラメータ
sampled_params = [
    0.3,  # th: 低い閾値で積極的探査
    25.0, # k_e: 高い探査誘導
    15.0, # k_c: 中程度の衝突回避
    0.4,  # branch_threshold: 低い分岐閾値
    0.6   # merge_threshold: 高い統合閾値
]
```

### 2. 安定探査設定

```python
config = SwarmExplorationRewardConfig(
    exploration_base_reward=8.0,         # 中程度の基本報酬
    exploration_improvement_weight=40.0,  # 中程度の改善重み
    time_efficiency_bonus=15.0,          # 中程度の時間効率ボーナス
    completion_bonus=80.0,               # 中程度の完了ボーナス
    revisit_penalty_weight=3.0           # 緩い再訪問ペナルティ
)

# 群制御パラメータ
sampled_params = [
    0.5,  # th: 中程度の閾値
    15.0, # k_e: 中程度の探査誘導
    10.0, # k_c: 中程度の衝突回避
    0.5,  # branch_threshold: 中程度の分岐閾値
    0.5   # merge_threshold: 中程度の統合閾値
]
```

### 3. 高速探査設定

```python
config = SwarmExplorationRewardConfig(
    exploration_base_reward=20.0,        # 非常に高い基本報酬
    exploration_improvement_weight=100.0, # 非常に高い改善重み
    time_efficiency_bonus=50.0,          # 非常に高い時間効率ボーナス
    completion_bonus=200.0,              # 非常に高い完了ボーナス
    revisit_penalty_weight=12.0          # 非常に強い再訪問ペナルティ
)

# 群制御パラメータ
sampled_params = [
    0.2,  # th: 非常に低い閾値で高速探査
    35.0, # k_e: 非常に高い探査誘導
    20.0, # k_c: 高い衝突回避
    0.3,  # branch_threshold: 非常に低い分岐閾値
    0.7   # merge_threshold: 非常に高い統合閾値
]
```

## パラメータ最適化の効果

### 1. 探査効率の向上

- **k_e の最適化**: 未探査領域への効率的な誘導
- **th の最適化**: 走行可能性と探査向上性の最適バランス
- **k_c の最適化**: 安全性を保ちながら探査を継続

### 2. 群制御の最適化

- **branch_threshold の最適化**: 最適な群サイズと分岐タイミング
- **merge_threshold の最適化**: 効率的な群統合による協調探査
- **動きやすさ指標の活用**: 客観的な群制御判断

### 3. 時間効率の向上

- 早期完了ボーナスによる時間意識の向上
- 非線形報酬による探査率向上の加速
- 目標達成報酬による完了への誘導

### 4. 群ロボットの協調行動

- 分散度報酬による群の分散維持
- 効率的移動報酬による無駄な移動の削減
- 協調的な探査パターンの生成
- 動的群制御による適応的組織化

## 学習戦略

### 1. 段階的最適化

1. **初期段階**: 基本的な探査行動の学習
2. **中期段階**: 効率的な探査パターンの学習
3. **後期段階**: 時間効率と完了率の最適化
4. **最終段階**: 群制御パラメータの最適化

### 2. 適応的パラメータ調整

- 探査率に応じたパラメータ範囲の動的調整
- 環境の複雑さに応じた重みの調整
- 群ロボット数に応じた協調パラメータの調整
- 動きやすさ指標に基づく群制御の動的調整

### 3. 多目標最適化

- 探査率最大化
- 時間効率最大化
- 安全性確保
- 群の協調性維持
- 群制御の最適化

## 評価指標

### 1. 主要指標

- **最終探査率**: 目標達成時の探査率
- **平均探査率**: 全エピソードの平均探査率
- **完了時間**: 目標達成までのステップ数
- **探査効率**: 探査率/ステップ数の比率

### 2. 群制御指標

- **群分岐回数**: 分岐の頻度と効果
- **群統合回数**: 統合の頻度と効果
- **平均群サイズ**: 群の最適サイズ
- **群制御効率**: 群制御による探査効率向上

### 3. 副次指標

- **衝突回数**: 安全性の指標
- **再訪問率**: 探査効率の指標
- **群の分散度**: 協調性の指標
- **移動距離**: 効率性の指標
- **動きやすさ指標**: 環境適応性の指標

## 最適化アルゴリズム

### 1. A2C（Advantage Actor-Critic）

```python
# Actor-Criticモデルによるパラメータ最適化
class ModelActorCritic:
    def __init__(self, input_dim):
        self.actor = self._build_actor(input_dim)
        self.critic = self._build_critic(input_dim)

    def _build_actor(self, input_dim):
        # パラメータ生成ネットワーク
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='sigmoid')  # th, k_e, k_c, branch_th, merge_th
        ])
```

### 2. パラメータ生成

```python
def get_action(self, state, sampled_params, episode, log_dir):
    """
    学習可能パラメータによる行動決定
    """
    # パラメータの正規化
    th = sampled_params[0]                    # 0-1
    k_e = sampled_params[1] * SCALE_MAX       # 0-50
    k_c = sampled_params[2] * SCALE_MAX       # 0-50
    branch_threshold = sampled_params[3]      # 0-1
    merge_threshold = sampled_params[4]       # 0-1

    # アルゴリズムパラメータの更新
    self.algorithm.update_params(th, k_e, k_c, branch_threshold, merge_threshold)

    # 行動決定
    return self.algorithm.policy(state, sampled_params, episode, log_dir)
```

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
