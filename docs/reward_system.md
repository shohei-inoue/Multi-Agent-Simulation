# 探査報酬システム

## 概要

探査率の上昇具合に応じて動的に報酬を計算するシステムを実装しました。これにより、より効率的な探査行動の学習が可能になります。

## 基本機能

### 1. 上昇量に応じた報酬計算

```python
# 基本報酬 + 上昇量に応じた追加報酬
base_reward = 5.0
multiplier = 10.0
improvement = exploration_ratio - previous_ratio
additional_reward = multiplier * (improvement ** 0.5)  # 平方根で非線形化
total_reward = base_reward + additional_reward
```

### 2. 非線形報酬計算

- `nonlinearity = 0.5`: 平方根（推奨）
- `nonlinearity = 1.0`: 線形
- `nonlinearity = 2.0`: 二乗

### 3. 報酬の範囲制限

- 最大報酬: `max_reward = 50.0`
- 最小報酬: `min_reward = -10.0`

## 使用方法

### 基本使用

```python
from envs.reward import calculate_exploration_reward

# 環境のstepメソッド内で
exploration_reward = calculate_exploration_reward(
    self.exploration_ratio, 
    previous_ratio, 
    self.__reward_dict
)
reward += exploration_reward
```

### 高度な設定

```python
from params.reward import RewardConfig, ExplorationRewardConfig
from envs.reward import calculate_exploration_reward_advanced

# カスタム設定
config = RewardConfig(
    exploration=ExplorationRewardConfig(
        base_reward=3.0,
        improvement_multiplier=15.0,
        nonlinearity=0.7,
        max_reward=30.0,
        min_reward=-5.0
    ),
    revisit_penalty=-1.5
)

# 報酬計算
reward = calculate_exploration_reward_advanced(
    exploration_ratio, 
    previous_ratio, 
    config
)
```

### モメンタム機能

```python
from envs.reward import calculate_exploration_reward_with_momentum

# 連続改善ボーナス付き
consecutive_improvements = 0  # エピソード開始時に初期化

# 各ステップで
reward, consecutive_improvements = calculate_exploration_reward_with_momentum(
    exploration_ratio, 
    previous_ratio, 
    config,
    momentum_factor=0.8,
    consecutive_improvements=consecutive_improvements
)
```

## 設定例

### 1. 積極的探査設定

```python
config = RewardConfig(
    exploration=ExplorationRewardConfig(
        base_reward=8.0,
        improvement_multiplier=20.0,
        nonlinearity=0.3,  # より線形に近い
        max_reward=100.0
    ),
    revisit_penalty=-5.0  # 再訪問を強く抑制
)
```

### 2. 安定探査設定

```python
config = RewardConfig(
    exploration=ExplorationRewardConfig(
        base_reward=3.0,
        improvement_multiplier=8.0,
        nonlinearity=0.8,  # より非線形
        max_reward=25.0
    ),
    revisit_penalty=-1.0  # 再訪問を緩く抑制
)
```

### 3. 高速探査設定

```python
config = RewardConfig(
    exploration=ExplorationRewardConfig(
        base_reward=10.0,
        improvement_multiplier=30.0,
        nonlinearity=0.2,  # ほぼ線形
        max_reward=200.0
    ),
    revisit_penalty=-10.0  # 再訪問を強く抑制
)
```

## 効果

1. **効率的な探査**: 大きな改善に対してより大きな報酬を与えることで、効率的な探査を促進
2. **学習の安定性**: 非線形性により、小さな改善でも適切な報酬を与える
3. **モメンタム効果**: 連続改善により、一貫した探査行動を促進
4. **カスタマイズ性**: パラメータ設定により、様々な探査戦略に対応

## 注意点

- 報酬の範囲制限により、学習の安定性を確保
- 非線形性パラメータは慎重に調整する必要がある
- モメンタム機能は連続改善回数の管理が必要 