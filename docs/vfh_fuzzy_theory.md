# VFH-Fuzzy 理論

## 概要

VFH-Fuzzy（Vector Field Histogram Fuzzy）は、群ロボットの未知環境における効率的な探査と障害物回避を実現するためのナビゲーションアルゴリズムです。従来の VFH（Vector Field Histogram）アルゴリズムにファジィ推論を組み合わせることで、より柔軟で適応的な行動決定を可能にします。

> **数式仕様**: 本アルゴリズムの詳細な数学的定義については、[VFH-Fuzzy推論の数式仕様](vfh_fuzzy_mathematical_specification.md)を参照してください。

## 1. VFH（Vector Field Histogram）の基礎理論

### 1.1 VFH アルゴリズムの原理

VFH アルゴリズムは、ロボット周囲の障害物情報を極座標ヒストグラムとして表現し、安全な移動方向を決定する手法です。

#### 1.1.1 極座標ヒストグラムの生成

```
極座標ヒストグラム = {h(k) | k = 0, 1, 2, ..., n-1}
```

ここで：

- `h(k)`: k 番目の角度ビンの値
- `n`: ビン数（本システムでは 36）
- `k`: ビンインデックス（0 ≤ k < n）

#### 1.1.2 角度ビンの計算

```python
# ビンサイズの計算
BIN_SIZE_DEG = 360 / BIN_NUM  # 10度（36ビンの場合）

# 角度からビンインデックスへの変換
def angle_to_bin(angle_deg):
    bin_index = int(angle_deg / BIN_SIZE_DEG) % BIN_NUM
    return bin_index

# ビンインデックスから角度への変換
def bin_to_angle(bin_index):
    angle_deg = bin_index * BIN_SIZE_DEG
    return angle_deg
```

### 1.2 障害物密度の計算

#### 1.2.1 距離ベースの障害物密度

```python
def calculate_obstacle_density(azimuth, distance):
    """
    方位azimuth、距離distanceにおける障害物密度を計算

    Args:
        azimuth: 方位角（ラジアン）
        distance: 障害物までの距離

    Returns:
        density: 障害物密度（0-1）
    """
    if distance < 1e-6:
        return 1.0  # 衝突

    # 距離に基づく密度計算（距離が近いほど密度が高い）
    density = 1.0 / (1.0 + distance)
    return density
```

#### 1.2.2 ヒストグラムの正規化

```python
def normalize_histogram(histogram):
    """
    ヒストグラムを正規化

    Args:
        histogram: 生のヒストグラム

    Returns:
        normalized_histogram: 正規化されたヒストグラム
    """
    # 最小値を保証
    histogram = np.maximum(histogram, 0.01)

    # ゼロ除算対策
    histogram += 1e-6

    # 正規化
    normalized_histogram = histogram / np.sum(histogram)

    return normalized_histogram
```

## 2. ファジィ推論の理論

### 2.1 ファジィ集合論の基礎

#### 2.1.1 ファジィ集合の定義

ファジィ集合 A は、要素 x の所属度 μ_A(x)で定義されます：

```
A = {(x, μ_A(x)) | x ∈ X}
```

ここで：

- `X`: 論議領域
- `μ_A(x)`: 要素 x の所属度（0 ≤ μ_A(x) ≤ 1）

#### 2.1.2 ファジィ集合の演算

**和集合（OR 演算）**:

```
μ_{A∪B}(x) = max(μ_A(x), μ_B(x))
```

**積集合（AND 演算）**:

```
μ_{A∩B}(x) = min(μ_A(x), μ_B(x))
```

**補集合（NOT 演算）**:

```
μ_{A'}(x) = 1 - μ_A(x)
```

### 2.2 ファジィ推論システム

#### 2.2.1 ファジィルール

VFH-Fuzzy では以下のファジィルールを使用します：

```
IF 走行可能性 IS 高 AND 探査向上性 IS 高 THEN 行動価値 IS 非常に良い
IF 走行可能性 IS 高 AND 探査向上性 IS 低 THEN 行動価値 IS 良い
IF 走行可能性 IS 低 AND 探査向上性 IS 高 THEN 行動価値 IS あまり良くない
IF 走行可能性 IS 低 AND 探査向上性 IS 低 THEN 行動価値 IS 悪い
```

#### 2.2.2 ファジィ推論の実装

```python
def fuzzy_inference(drivability, exploration_improvement, th=0.5, alpha=10.0):
    """
    ファジィ推論による行動価値の計算

    Args:
        drivability: 走行可能性（0-1）
        exploration_improvement: 探査向上性（0-1）
        th: 抑制閾値
        alpha: ソフト抑制の鋭さ

    Returns:
        action_value: 行動価値（0-1）
    """
    # ソフト抑制係数の計算
    suppression = 1 / (1 + np.exp(-alpha * (drivability - th)))

    # ファジィ積推論（AND演算）
    action_value = suppression * drivability * exploration_improvement

    return action_value
```

## 3. VFH-Fuzzy アルゴリズムの理論

### 3.1 アルゴリズムの全体構造

VFH-Fuzzy アルゴリズムは以下の 3 つの主要ステップで構成されます：

1. **走行可能性ヒストグラムの生成**
2. **探査向上性ヒストグラムの生成**
3. **ファジィ推論による統合**

### 3.2 走行可能性ヒストグラム（Drivability Histogram）

#### 3.2.1 理論的基礎

走行可能性は、各方向における障害物の密度に基づいて計算されます：

```
D(k) = 1 - ρ(k)
```

ここで：

- `D(k)`: k 番目ビンの走行可能性
- `ρ(k)`: k 番目ビンの障害物密度

#### 3.2.2 衝突回避機能

前回衝突した方向を避ける機能を実装：

```python
def collision_avoidance_histogram(last_collision_theta, bin_num):
    """
    衝突回避ヒストグラムの生成

    Args:
        last_collision_theta: 前回衝突した方向（ラジアン）
        bin_num: ビン数

    Returns:
        histogram: 衝突回避ヒストグラム
    """
    histogram = np.ones(bin_num)

    if last_collision_theta is not None:
        # 衝突ビンの計算
        collision_bin = int(last_collision_theta * bin_num / (2 * np.pi)) % bin_num

        # 衝突方向周辺の抑制
        for i in range(bin_num):
            distance = min(
                abs(i - collision_bin),
                abs(i - collision_bin + bin_num),
                abs(i - collision_bin - bin_num)
            )

            if distance < 4:  # 40度範囲の抑制
                histogram[i] = 0.05  # 低評価
            else:
                histogram[i] = 1.0   # 通常評価

    return histogram
```

### 3.3 探査向上性ヒストグラム（Exploration Improvement Histogram）

#### 3.3.1 理論的基礎

探査向上性は、以下の要素を考慮して計算されます：

1. **過去方向の抑制**: 既に探査済みの方向を避ける
2. **衝突方向の抑制**: 衝突が発生した方向を避ける
3. **探査欲求の促進**: 未探査領域への誘導

#### 3.3.2 ガウス分布による方向重み

```python
def gaussian_direction_weight(base_azimuth, k, sharpness=10.0):
    """
    ガウス分布による方向重みの適用

    Args:
        base_azimuth: 基準方位（ラジアン）
        k: 抑制強度（0-1）
        sharpness: 分布の鋭さ

    Returns:
        weight_function: 重み関数
    """
    def weight_function(angle):
        angle_diff = abs(angle - base_azimuth)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        # ガウス分布による減衰
        decay = 1 - (1 - k) * np.exp(-sharpness * angle_diff ** 2)
        return decay

    return weight_function
```

#### 3.3.3 von Mises 分布による方向重み

```python
def von_mises_direction_weight(base_azimuth, kappa):
    """
    von Mises分布による方向重みの適用

    Args:
        base_azimuth: 基準方位（ラジアン）
        kappa: 集中度パラメータ

    Returns:
        weight_function: 重み関数
    """
    def weight_function(angle):
        angle_diff = abs(angle - base_azimuth)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        # von Mises分布
        decay = np.exp(kappa * np.cos(angle_diff)) / (2 * np.pi * i0(kappa))
        return decay

    return weight_function
```

### 3.4 ファジィ推論による統合

#### 3.4.1 ソフト抑制付きファジィ推論

```python
def soft_suppression_fuzzy_inference(drivability, exploration_improvement, th, alpha=10.0):
    """
    ソフト抑制付きファジィ推論

    Args:
        drivability: 走行可能性ヒストグラム
        exploration_improvement: 探査向上性ヒストグラム
        th: 抑制閾値
        alpha: ソフト抑制の鋭さ

    Returns:
        result_histogram: 統合結果ヒストグラム
    """
    result_histogram = np.zeros_like(drivability)

    for i in range(len(drivability)):
        drive_val = drivability[i]
        explore_val = exploration_improvement[i]

        # ソフト抑制係数の計算
        suppression = 1 / (1 + np.exp(-alpha * (drive_val - th)))

        # ファジィ積推論
        result_histogram[i] = suppression * drive_val * explore_val

    return result_histogram
```

#### 3.4.2 数学的説明

ソフト抑制関数は以下の数式で定義されます：

```
suppression(d) = 1 / (1 + exp(-α * (d - th)))
```

ここで：

- `d`: 走行可能性値
- `th`: 抑制閾値
- `α`: ソフト抑制の鋭さ

この関数は：

- `d < th` のとき、抑制が強く働く（値が小さくなる）
- `d > th` のとき、抑制が弱く働く（値が大きくなる）
- `α` が大きいほど、閾値付近での変化が急激になる

## 4. 方向選択アルゴリズム

### 4.1 重み付けサンプリング

#### 4.1.1 四分位数ベースの分類

```python
def quantile_based_classification(result_histogram):
    """
    四分位数ベースのビン分類

    Args:
        result_histogram: 結果ヒストグラム

    Returns:
        bins_very_good: 非常に良いビン
        bins_good: 良いビン
        bins_okay: まあまあのビン
    """
    q1, q2, q3 = np.quantile(result_histogram, [0.25, 0.5, 0.75])

    bins_very_good = [i for i, score in enumerate(result_histogram) if score >= q3]
    bins_good = [i for i, score in enumerate(result_histogram) if q2 <= score < q3]
    bins_okay = [i for i, score in enumerate(result_histogram) if q1 <= score < q2]

    return bins_very_good, bins_good, bins_okay
```

#### 4.1.2 重み付け確率の計算

```python
def calculate_weighted_probabilities(bins_very_good, bins_good, bins_okay):
    """
    重み付け確率の計算

    Args:
        bins_very_good: 非常に良いビン
        bins_good: 良いビン
        bins_okay: まあまあのビン

    Returns:
        weights: 重み配列
        bins: ビン配列
    """
    # 重みの設定
    score_q1 = 0.6 if bins_very_good else 0.0  # Very Good: 60%
    score_q2 = 0.25 if bins_good else 0.0      # Good: 25%
    score_q3 = 0.15 if bins_okay else 0.0      # Okay: 15%

    total_score = score_q1 + score_q2 + score_q3

    # 正規化
    if total_score > 0:
        score_q1 /= total_score
        score_q2 /= total_score
        score_q3 /= total_score

    # 重み配列の構築
    weights = []
    bins = []

    if bins_very_good:
        weights += [score_q1 / len(bins_very_good)] * len(bins_very_good)
        bins += bins_very_good

    if bins_good:
        weights += [score_q2 / len(bins_good)] * len(bins_good)
        bins += bins_good

    if bins_okay:
        weights += [score_q3 / len(bins_okay)] * len(bins_okay)
        bins += bins_okay

    return weights, bins
```

### 4.2 候補方向選択

#### 4.2.1 隣接ビンのグループ化

```python
def group_adjacent_bins(valid_bins, bin_num):
    """
    隣接ビンのグループ化

    Args:
        valid_bins: 有効なビンのリスト
        bin_num: ビン数

    Returns:
        groups: グループ化されたビンのリスト
    """
    if len(valid_bins) >= bin_num * 0.8:  # 80%以上のビンが有効な場合
        # 個別のビンをグループとして扱う
        groups = [[bin] for bin in valid_bins]
    else:
        # 隣接ビンをグループ化
        groups = []
        current_group = []

        for idx in range(len(valid_bins)):
            if not current_group:
                current_group.append(valid_bins[idx])
            else:
                prev = current_group[-1]
                if (valid_bins[idx] == (prev + 1) % bin_num):
                    current_group.append(valid_bins[idx])
                else:
                    groups.append(current_group)
                    current_group = [valid_bins[idx]]

        if current_group:
            # 先頭と末尾がつながっている場合
            if groups and (current_group[0] == 0 and groups[0][-1] == bin_num - 1):
                groups[0] = current_group + groups[0]
            else:
                groups.append(current_group)

    return groups
```

#### 4.2.2 グループ中心角度の計算

```python
def calculate_group_center_angle(group, bin_num):
    """
    グループの中心角度を計算

    Args:
        group: ビングループ
        bin_num: ビン数

    Returns:
        center_angle: 中心角度（ラジアン）
    """
    if len(group) == 1:
        # 個別ビンの場合は直接角度を計算
        center_angle = 2 * np.pi * group[0] / bin_num
    else:
        # グループの場合は平均角度を計算
        angles = [2 * np.pi * i / bin_num for i in group]
        center_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    return center_angle
```

## 5. 学習パラメータの理論

### 5.1 学習可能パラメータ

VFH-Fuzzy アルゴリズムでは以下のパラメータが学習対象となります：

#### 5.1.1 抑制閾値（th）

```python
# 抑制閾値の意味
th: 走行可能性の閾値（0-1）
- th < 0.3: 非常に寛容（多くの方向を選択可能）
- 0.3 ≤ th < 0.7: 適度な厳格さ
- th ≥ 0.7: 非常に厳格（少数の方向のみ選択）
```

#### 5.1.2 探査抑制強度（k_e）

```python
# 探査抑制強度の意味
k_e: 探査逆方向の抑制強度（0-1）
- k_e ≈ 0: 過去方向への強い抑制
- k_e ≈ 0.5: 適度な抑制
- k_e ≈ 1: 抑制なし（過去方向も選択可能）
```

#### 5.1.3 衝突抑制強度（k_c）

```python
# 衝突抑制強度の意味
k_c: 衝突方向の抑制強度（0-1）
- k_c ≈ 0: 衝突方向への強い抑制
- k_c ≈ 0.5: 適度な抑制
- k_c ≈ 1: 抑制なし（衝突方向も選択可能）
```

### 5.2 パラメータ学習の理論

#### 5.2.1 強化学習による最適化

```python
def parameter_learning_objective(reward, current_params, target_params):
    """
    パラメータ学習の目的関数

    Args:
        reward: 報酬
        current_params: 現在のパラメータ
        target_params: 目標パラメータ

    Returns:
        loss: 損失値
    """
    # 報酬に基づく損失
    reward_loss = -reward

    # パラメータ正則化
    regularization_loss = 0.1 * np.sum((current_params - target_params) ** 2)

    total_loss = reward_loss + regularization_loss
    return total_loss
```

#### 5.2.2 適応的パラメータ調整

```python
def adaptive_parameter_update(current_params, reward, learning_rate=0.001):
    """
    適応的パラメータ更新

    Args:
        current_params: 現在のパラメータ
        reward: 報酬
        learning_rate: 学習率

    Returns:
        updated_params: 更新されたパラメータ
    """
    # 報酬に基づく勾配計算
    if reward > 0:
        # 良い報酬の場合、現在のパラメータを強化
        gradient = learning_rate * reward
    else:
        # 悪い報酬の場合、パラメータを調整
        gradient = -learning_rate * abs(reward)

    # パラメータ更新
    updated_params = current_params + gradient

    # パラメータの範囲制限
    updated_params = np.clip(updated_params, 0.0, 1.0)

    return updated_params
```

## 6. 理論的優位性

### 6.1 従来の VFH との比較

#### 6.1.1 VFH の限界

1. **硬い閾値**: 固定閾値による硬い判断
2. **単一目的**: 障害物回避のみに焦点
3. **適応性不足**: 環境変化への対応が困難

#### 6.1.2 VFH-Fuzzy の改善点

1. **柔軟な判断**: ファジィ推論による曖昧性の処理
2. **多目的最適化**: 障害物回避と探査効率の両立
3. **適応的動作**: 学習によるパラメータ最適化

### 6.2 ファジィ推論の効果

#### 6.2.1 曖昧性の処理

```python
# 従来のVFH（硬い判断）
if drivability > threshold:
    direction_valid = True
else:
    direction_valid = False

# VFH-Fuzzy（柔軟な判断）
suppression = 1 / (1 + np.exp(-alpha * (drivability - th)))
action_value = suppression * drivability * exploration_improvement
```

#### 6.2.2 多目的最適化

```python
# 走行可能性と探査向上性の統合
def multi_objective_optimization(drivability, exploration_improvement):
    """
    多目的最適化による行動価値の計算
    """
    # ファジィ積推論による統合
    action_value = drivability * exploration_improvement

    return action_value
```

## 7. 理論的保証

### 7.1 収束性の保証

#### 7.1.1 ファジィ推論の収束性

ファジィ推論システムは以下の条件で収束することが保証されます：

1. **有界性**: すべてのファジィ集合が有界
2. **連続性**: ファジィ推論関数が連続
3. **単調性**: 入力の増加に対して出力が単調増加

#### 7.1.2 学習アルゴリズムの収束性

Actor-Critic アルゴリズムの収束性は以下の条件で保証されます：

1. **探索性**: 十分な探索が行われる
2. **学習率**: 適切な学習率の設定
3. **報酬設計**: 適切な報酬関数の設計

### 7.2 安定性の保証

#### 7.2.1 システム安定性

VFH-Fuzzy システムの安定性は以下の要素により保証されます：

1. **衝突回避**: 確実な衝突回避機能
2. **パラメータ制限**: パラメータの範囲制限
3. **正則化**: 過学習の防止

#### 7.2.2 ロバスト性

システムのロバスト性は以下の要素により実現されます：

1. **フォールバック**: 異常時の代替行動
2. **冗長性**: 複数の判断基準
3. **適応性**: 環境変化への対応

## 8. 今後の理論的発展

### 8.1 深層学習との統合

#### 8.1.1 ニューラルネットワークによるパラメータ学習

```python
class DeepVFHFuzzy:
    """
    深層学習を統合したVFH-Fuzzy
    """
    def __init__(self, state_size, action_size):
        self.actor = self._build_actor(state_size, action_size)
        self.critic = self._build_critic(state_size)

    def _build_actor(self, state_size, action_size):
        """
        Actorネットワークの構築
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])
        return model
```

### 8.2 メタ学習の導入

#### 8.2.1 環境適応の高速化

```python
class MetaVFHFuzzy:
    """
    メタ学習を導入したVFH-Fuzzy
    """
    def __init__(self):
        self.meta_learner = self._build_meta_learner()

    def adapt_to_environment(self, environment_samples):
        """
        環境への適応
        """
        # メタ学習による迅速な適応
        adapted_params = self.meta_learner.adapt(environment_samples)
        return adapted_params
```

### 8.3 理論的拡張

#### 8.3.1 多エージェント協調

```python
class MultiAgentVFHFuzzy:
    """
    多エージェント協調VFH-Fuzzy
    """
    def __init__(self, num_agents):
        self.agents = [VFHFuzzy() for _ in range(num_agents)]
        self.coordination_mechanism = self._build_coordination()

    def coordinated_action_selection(self, states):
        """
        協調的行動選択
        """
        # エージェント間の協調による行動選択
        coordinated_actions = self.coordination_mechanism.select_actions(states)
        return coordinated_actions
```

## 9. 結論

VFH-Fuzzy アルゴリズムは、従来の VFH アルゴリズムの限界を克服し、より柔軟で適応的な群ロボットナビゲーションを実現する理論的基盤を提供します。

### 9.1 主要な理論的貢献

1. **ファジィ推論の導入**: 曖昧性の処理と柔軟な判断
2. **多目的最適化**: 障害物回避と探査効率の両立
3. **学習による適応**: 環境変化への動的対応
4. **理論的保証**: 収束性と安定性の保証

### 9.2 実用的な利点

1. **実装の簡潔性**: 理解しやすいアルゴリズム構造
2. **計算効率**: 効率的な計算アルゴリズム
3. **拡張性**: 新機能の追加が容易
4. **堅牢性**: 異常時への対応能力

### 9.3 今後の研究方向

1. **深層学習統合**: より高度な学習機能の実現
2. **メタ学習**: 環境適応の高速化
3. **多エージェント協調**: 大規模システムへの適用
4. **理論的発展**: より厳密な数学的基盤の構築

VFH-Fuzzy アルゴリズムは、群ロボットシステムにおける実用的で理論的に裏付けられたナビゲーション手法として、今後も発展が期待される重要な技術です。
