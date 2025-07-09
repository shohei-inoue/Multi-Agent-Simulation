# Algorithm: VFH-Fuzzy 行動決定法

## 概要

VFH-Fuzzy（Vector Field Histogram Fuzzy）は、障害物回避と探査向上性を組み合わせた群ロボット用ナビゲーションアルゴリズムです。ファジィ推論を用いて走行可能性と探査向上性を統合し、効率的な探査行動を実現します。

## 1. 基本パラメータ

### アルゴリズム定数

- `BIN_SIZE_DEG`: ビンのサイズ（度）- デフォルト: 5 度
- `BIN_NUM`: ビン数 - デフォルト: 72 個
- `ANGLES`: 角度配列 - 0 から 2π まで均等分布

### 学習可能パラメータ

- `th`: 抑制のしきい値（しきい値以下の drivability は抑制）
- `k_e`: 探査逆方向の抑制強度
- `k_c`: 衝突方向の抑制強度
- `α`: ソフト抑制の鋭さ（固定: α=10.0）

## 2. ヒストグラム生成

### 2.1 走行可能性ヒストグラム（Drivability）

```python
def get_obstacle_density_histogram(self):
    """
    障害物密度に基づく走行可能性分布を生成
    """
    histogram = np.zeros(self.BIN_NUM)

    # 各ビンについて障害物密度を計算
    for i, angle in enumerate(self.ANGLES):
        # 障害物までの距離を計算
        distance = self._calculate_obstacle_distance(angle)

        # 距離に基づいて走行可能性を計算
        if distance > 0:
            histogram[i] = 1.0 / (1.0 + distance)
        else:
            histogram[i] = 0.0  # 障害物に衝突

    return histogram
```

### 2.2 探査向上性ヒストグラム（Exploration Improvement）

```python
def get_exploration_improvement_histogram(self) -> np.ndarray:
    """
    探査向上性に基づく探索欲求分布を生成
    """
    histogram = np.ones(self.BIN_NUM)

    # 過去方向・衝突方向の抑制
    def apply_direction_weight(base_azimuth: float, k: float, sharpness: float = 10.0):
        """
        方向重みの適用
        """
        for i, angle in enumerate(self.ANGLES):
            angle_diff = abs(angle - base_azimuth)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            # 抑制関数
            decay = 1 - (1 - k) * np.exp(-sharpness * angle_diff ** 2)
            histogram[i] *= decay

    # 探査逆方向（過去方向）の抑制
    if hasattr(self, 'agent_azimuth') and self.agent_azimuth is not None:
        apply_direction_weight(self.agent_azimuth, self.k_e, 10.0)

    # 衝突方向の抑制
    for collision_point in self.follower_collision_data:
        if len(collision_point) >= 2:
            collision_angle = np.arctan2(collision_point[1], collision_point[0])
            apply_direction_weight(collision_angle, self.k_c, 20.0)

    return histogram
```

## 3. ファジィ推論

### 3.1 ファジィルール

```
走行可能性    | 探査向上性 | 結果
---------------------------------------
| 低        | 低        | 悪い          |
| 低        | 高        | あまり良くない  |
| 高        | 低        | 良い          |
| 高        | 高        | 非常に良い     |
---------------------------------------
```

### 3.2 ソフト抑制付きファジィ推論

```python
def get_final_result_histogram(self, drivability, exploration_improvement) -> np.ndarray:
    """
    ファジィ推論による最終結果ヒストグラムを生成
    """
    result = np.zeros(self.BIN_NUM)

    for i in range(self.BIN_NUM):
        D_i = drivability[i]  # 走行可能性
        E_i = exploration_improvement[i]  # 探査向上性

        # ソフト抑制項
        suppression_i = 1 / (1 + np.exp(-10.0 * (D_i - self.th)))

        # 積和平均（product inference + defuzzify）
        result[i] = suppression_i * D_i * E_i

    return result
```

**数式:**

- 抑制項: `suppression_i = 1 / (1 + exp(-α * (D_i - th)))`
- 最終スコア: `F_i = suppression_i * D_i * E_i`

## 4. 方向選択

### 4.1 重み付きサンプリング

```python
def select_final_direction_by_weighted_sampling(self, result: np.ndarray) -> float:
    """
    重み付きサンプリングによる最終方向選択
    """
    # 四分位点を計算
    q1, q2, q3 = np.quantile(result, [0.25, 0.5, 0.75])

    # ビンクラス分類
    very_good_bins = []
    good_bins = []
    okay_bins = []

    for i, score in enumerate(result):
        if score >= q3:
            very_good_bins.append(i)
        elif score >= q2:
            good_bins.append(i)
        elif score >= q1:
            okay_bins.append(i)

    # 重み付き選択
    weights = [0.6, 0.25, 0.15]  # Very Good, Good, Okay
    bin_groups = [very_good_bins, good_bins, okay_bins]

    # クラス選択
    selected_class = np.random.choice(len(bin_groups), p=weights)
    selected_bins = bin_groups[selected_class]

    if selected_bins:
        # 選択されたクラス内で等確率選択
        selected_bin = np.random.choice(selected_bins)
        self.selected_bin = selected_bin
        return self.ANGLES[selected_bin]
    else:
        # フォールバック: 最高スコアの方向
        selected_bin = np.argmax(result)
        self.selected_bin = selected_bin
        return self.ANGLES[selected_bin]
```

### 4.2 クラス分類と重み

| ビンクラス | 条件                       | 重み |
| ---------- | -------------------------- | ---- |
| Very Good  | `F_i ≥ Q3` (上位 25%)      | 0.6  |
| Good       | `Q2 ≤ F_i < Q3` (上位 50%) | 0.25 |
| Okay       | `Q1 ≤ F_i < Q2` (上位 75%) | 0.15 |

## 5. 群制御モード決定

```python
def _determine_swarm_mode(self, state, sampled_params):
    """
    群分岐・統合のモードを決定
    """
    # フォロワーの動きやすさ指標を取得
    follower_mobility_scores = state.get('follower_mobility_scores', [])

    if len(follower_mobility_scores) >= 3:
        avg_mobility = np.mean(follower_mobility_scores[:3])

        # 分岐閾値（学習可能）
        branch_threshold = sampled_params[3] if len(sampled_params) > 3 else 0.5

        if avg_mobility > branch_threshold:
            return 1  # 分岐モード
        else:
            return 2  # 統合モード

    return 0  # 通常モード
```

## 6. 出力

### 6.1 行動辞書

```python
action = {
    "theta": self.theta,  # 移動方向（ラジアン）
    "mode": self.mode     # 群制御モード（0: 通常, 1: 分岐, 2: 統合）
}
```

### 6.2 可視化情報

- `drivability_histogram`: 走行可能性分布
- `exploration_improvement_histogram`: 探査向上性分布
- `result_histogram`: 最終結果分布
- `selected_bin`: 選択されたビン番号

## 7. 特徴

### 7.1 学習可能パラメータ

- 抑制閾値 `th`: 走行可能性の閾値
- 探査抑制強度 `k_e`: 過去方向への抑制
- 衝突抑制強度 `k_c`: 衝突方向への抑制
- 分岐閾値: 群分岐の判断基準

### 7.2 適応性

- 障害物密度に基づく動的な走行可能性計算
- 探査履歴に基づく方向性の学習
- 群の状態に応じた制御モードの動的切り替え

### 7.3 安全性

- 衝突方向への強力な抑制
- ソフト抑制による滑らかな行動決定
- 重み付きサンプリングによる多様性確保

## 8. 使用例

```python
# アルゴリズムの初期化
algorithm = AlgorithmVfhFuzzy(env)

# パラメータの更新（学習時）
algorithm.update_params(th=0.5, k_e=10.0, k_c=5.0)

# 行動決定
action_tensor, action_dict = algorithm.policy(
    state=current_state,
    sampled_params=[0.5, 10.0, 5.0, 0.3, 0.7],  # th, k_e, k_c, branch_th, merge_th
    episode=episode,
    log_dir=log_dir
)

# 行動の実行
theta = action_dict["theta"]  # 移動方向
mode = action_dict["mode"]    # 群制御モード
```
