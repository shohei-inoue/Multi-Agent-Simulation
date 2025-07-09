# 探査報酬システム

## 概要

群ロボット探査シミュレーションにおける報酬システムは、探査率の上昇具合に応じて動的に報酬を計算し、効率的な探査行動の学習を促進します。複数の報酬計算方式を提供し、様々な探査戦略に対応できます。

## 基本報酬設定

### 1. デフォルト報酬辞書

```python
def create_reward():
    return {
        'default': -1,                    # デフォルトの時間ペナルティ
        'exploration_gain': +5,           # 新しい未踏エリアに到達したときの報酬
        'exploration_gain_multiplier': 10.0,  # 探査率上昇量に応じた倍率
        'revisit_penalty': -2,            # 探索済み領域を再訪したときのペナルティ
        'collision_penalty': -10,         # エージェントが衝突したときのペナルティ
        'clear_target_rate': +50,         # 探査率が目的の値以上になった場合
        'none_finish_penalty': -50,       # 探査が最終ステップまでに終わらなかった場合
    }
```

## 報酬計算方式

### 1. 基本探査報酬計算

```python
def calculate_exploration_reward(exploration_ratio, previous_ratio, reward_dict):
    if exploration_ratio > previous_ratio:
        # 探査率が上昇した場合
        improvement = exploration_ratio - previous_ratio

        # 基本報酬 + 上昇量に応じた追加報酬
        base_reward = reward_dict.get('exploration_gain', 5.0)
        multiplier = reward_dict.get('exploration_gain_multiplier', 10.0)

        # 上昇量が大きいほど報酬も大きくなる（非線形）
        additional_reward = multiplier * (improvement ** 0.5)  # 平方根で非線形化

        total_reward = base_reward + additional_reward
        return total_reward
    else:
        # 探査率が上昇しなかった場合
        return reward_dict.get('revisit_penalty', -2.0)
```

**特徴:**

- 探査率の上昇に対して非線形報酬を付与
- 平方根による非線形化で学習の安定性を確保
- 再訪問に対するペナルティ

### 2. 群ロボット専用報酬計算

```python
def calculate_swarm_exploration_reward(exploration_ratio, previous_ratio, reward_config,
                                     step_count, total_steps, explored_area, total_area):
    total_reward = 0.0

    # 1. 探査率向上の報酬（最重要）
    if exploration_ratio > previous_ratio:
        improvement = exploration_ratio - previous_ratio
        base_reward = reward_config.exploration_base_reward
        improvement_reward = reward_config.exploration_improvement_weight * improvement

        # 非線形報酬（探査率が高いほど報酬を増加）
        nonlinear_bonus = improvement_reward * (1 + exploration_ratio)
        exploration_reward = base_reward + nonlinear_bonus
        total_reward += exploration_reward
    else:
        # 探査率が上昇しなかった場合のペナルティ
        total_reward += reward_config.revisit_penalty_weight * (previous_ratio - exploration_ratio)

    # 2. 時間効率報酬（早期完了ボーナス）
    if exploration_ratio > 0.8:  # 80%以上探査済みの場合
        time_efficiency = 1.0 - (step_count / total_steps)
        time_bonus = reward_config.time_efficiency_bonus * time_efficiency * exploration_ratio
        total_reward += time_bonus

    # 3. 目標達成報酬（全探索完了）
    if exploration_ratio >= 0.95:  # 95%以上探査済みの場合
        completion_bonus = reward_config.completion_bonus * exploration_ratio
        total_reward += completion_bonus

    return total_reward
```

**特徴:**

- 探査率向上を最優先とした設計
- 時間効率を考慮した早期完了ボーナス
- 目標達成時の特別報酬

### 3. 高度な報酬計算（設定ファイル対応）

```python
def calculate_exploration_reward_advanced(exploration_ratio, previous_ratio, reward_config):
    if exploration_ratio > previous_ratio:
        improvement = exploration_ratio - previous_ratio

        # 設定から値を取得
        base_reward = reward_config.exploration.base_reward
        multiplier = reward_config.exploration.improvement_multiplier
        nonlinearity = reward_config.exploration.nonlinearity
        max_reward = reward_config.exploration.max_reward
        min_reward = reward_config.exploration.min_reward

        # 非線形報酬計算
        if nonlinearity == 0:
            # 線形
            additional_reward = multiplier * improvement
        else:
            # 非線形（平方根、二乗など）
            additional_reward = multiplier * (improvement ** nonlinearity)

        total_reward = base_reward + additional_reward

        # 報酬の範囲制限
        total_reward = max(min_reward, min(max_reward, total_reward))

        return total_reward
    else:
        return reward_config.revisit_penalty
```

**特徴:**

- 設定ファイルによる柔軟なパラメータ調整
- 報酬の範囲制限による学習安定性
- 非線形性の動的調整

### 4. モメンタム考慮報酬計算

```python
def calculate_exploration_reward_with_momentum(exploration_ratio, previous_ratio, reward_config,
                                             momentum_factor=0.8, consecutive_improvements=0):
    if exploration_ratio > previous_ratio:
        improvement = exploration_ratio - previous_ratio

        # 基本報酬計算
        base_reward = reward_config.exploration.base_reward
        multiplier = reward_config.exploration.improvement_multiplier
        nonlinearity = reward_config.exploration.nonlinearity

        # 非線形報酬計算
        if nonlinearity == 0:
            additional_reward = multiplier * improvement
        else:
            additional_reward = multiplier * (improvement ** nonlinearity)

        # モメンタムボーナス（連続改善による追加報酬）
        momentum_bonus = consecutive_improvements * momentum_factor * base_reward

        total_reward = base_reward + additional_reward + momentum_bonus

        # 報酬の範囲制限
        total_reward = max(reward_config.exploration.min_reward,
                          min(reward_config.exploration.max_reward, total_reward))

        # 連続改善回数を更新
        new_consecutive_improvements = consecutive_improvements + 1

        return total_reward, new_consecutive_improvements
    else:
        # 探査率が上昇しなかった場合、連続改善回数をリセット
        return reward_config.revisit_penalty, 0
```

**特徴:**

- 連続改善によるモメンタム効果
- 一貫した探査行動の促進
- 改善が止まった場合のリセット機能

## 環境での使用例

### 環境の step メソッド内での実装

```python
def step(self, action):
    # ... 行動実行 ...

    # 探査率の上昇具合に応じた報酬計算
    from .reward import calculate_exploration_reward
    exploration_reward = calculate_exploration_reward(
        self.exploration_ratio,
        previous_ratio,
        self.__reward_dict
    )
    reward += exploration_reward

    # 衝突ペナルティ
    if self.current_leader.collision_flag:
        reward += self.__reward_dict.get('collision_penalty', -10.0)

    # 目標達成報酬
    if self.explored_area >= self.total_area * self.__finish_rate:
        reward += self.__reward_dict.get('clear_target_rate', +50.0)

    return self.state, reward, done, truncated, {}
```

## 報酬設計の効果

### 1. 効率的な探査促進

- 大きな改善に対してより大きな報酬を与えることで、効率的な探査を促進
- 非線形性により、小さな改善でも適切な報酬を与える

### 2. 学習の安定性

- 報酬の範囲制限により、学習の安定性を確保
- ペナルティによる不適切な行動の抑制

### 3. 戦略的探査

- 時間効率を考慮した早期完了ボーナス
- 目標達成時の特別報酬によるモチベーション向上

### 4. モメンタム効果

- 連続改善により、一貫した探査行動を促進
- 一時的な停滞からの回復を支援

## 設定の推奨値

### 1. 積極的探査設定

```python
reward_dict = {
    'exploration_gain': 8.0,
    'exploration_gain_multiplier': 20.0,
    'revisit_penalty': -5.0,
    'collision_penalty': -15.0
}
```

### 2. 安定探査設定

```python
reward_dict = {
    'exploration_gain': 3.0,
    'exploration_gain_multiplier': 8.0,
    'revisit_penalty': -1.0,
    'collision_penalty': -8.0
}
```

### 3. 高速探査設定

```python
reward_dict = {
    'exploration_gain': 10.0,
    'exploration_gain_multiplier': 30.0,
    'revisit_penalty': -10.0,
    'collision_penalty': -20.0
}
```

## 注意点

- 報酬の範囲制限により、学習の安定性を確保
- 非線形性パラメータは慎重に調整する必要がある
- モメンタム機能は連続改善回数の管理が必要
- 群ロボット環境では、個別のロボットの報酬と群全体の報酬のバランスが重要
