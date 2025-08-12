# Algorithm: VFH-Fuzzy 行動決定法

## 概要

VFH-Fuzzy（Vector Field Histogram Fuzzy）は、障害物回避と探査向上性を組み合わせた群ロボット用ナビゲーションアルゴリズムです。ファジィ推論を用いて走行可能性と探査向上性を統合し、効率的な探査行動を実現します。

## 1. 基本パラメータ

### アルゴリズム定数

- `BIN_SIZE_DEG`: ビンのサイズ（度）- デフォルト: 10 度
- `BIN_NUM`: ビン数 - デフォルト: 36 個
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

### 2.2 探査向上性ヒストグラム（Exploration Improvement）

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

## 3. ファジィ推論

### 3.1 ソフト抑制付きファジィ推論

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

## 4. 方向選択

### 4.1 重み付けサンプリング

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

### 4.2 候補方向選択

```python
def select_direction_with_candidates(self, state, sampled_params):
    """
    theta（選択方向）とvalid_directions（有効方向リスト）を返す。
    隣接binを1経路グループとしてまとめ、各グループの中心角度とスコアを返す。
    """
    self.get_state_info(state, sampled_params)
    self.drivability_histogram = self.get_obstacle_density_histogram()
    self.exploration_improvement_histogram = self.get_exploration_improvement_histogram()
    self.result_histogram = self.get_final_result_histogram(self.drivability_histogram, self.exploration_improvement_histogram)
    mean_score = np.mean(self.result_histogram)
    valid_bins = [i for i, score in enumerate(self.result_histogram) if score >= mean_score and score > 0.0]

    # 全方向が有効な場合（36個すべて）は個別のbinを選択
    if len(valid_bins) >= self.BIN_NUM * 0.8:  # 80%以上のbinが有効な場合（29個以上）
        # 個別のbinをグループとして扱う
        groups = [[bin] for bin in valid_bins]
        print(f"[GROUPING] All directions valid ({len(valid_bins)}/36 bins), using individual bins")
    else:
        # 隣接binをグループ化
        groups = []
        current_group = []
        for idx in range(len(valid_bins)):
            if not current_group:
                current_group.append(valid_bins[idx])
            else:
                prev = current_group[-1]
                if (valid_bins[idx] == (prev + 1) % self.BIN_NUM):
                    current_group.append(valid_bins[idx])
                else:
                    groups.append(current_group)
                    current_group = [valid_bins[idx]]
        if current_group:
            # 先頭と末尾がつながっている場合
            if groups and (current_group[0] == 0 and groups[0][-1] == self.BIN_NUM - 1):
                groups[0] = current_group + groups[0]
            else:
                groups.append(current_group)

    # 各グループの中心角度・代表スコア
    valid_directions = []
    for group in groups:
        if len(group) == 1:
            # 個別binの場合は直接角度を計算
            center_angle = 2 * np.pi * group[0] / self.BIN_NUM
        else:
            # グループの場合は平均角度を計算
            angles = [2 * np.pi * i / self.BIN_NUM for i in group]
            center_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

        group_score = np.mean([self.result_histogram[i] for i in group])
        valid_directions.append({"angle": center_angle, "score": group_score})

    # thetaはグループのスコアで重み付けサンプリング
    scores = np.array([d["score"] for d in valid_directions])
    if len(scores) > 0:
        probs = scores / scores.sum()
        selected_idx = np.random.choice(len(valid_directions), p=probs)
        theta = valid_directions[selected_idx]["angle"]
    else:
        theta = np.random.uniform(0, 2 * np.pi)

    # 選択した方向を記録（衝突回避用）
    self.last_theta = theta

    return theta, valid_directions
```

## 5. 衝突回避機能

### 5.1 衝突方向の記録

```python
def get_state_info(self, state, sampled_params):
    """
    state内からこのpolicyで必要な情報を取得
    """
    self.agent_azimuth = state.get("agent_azimuth", 0.0)
    self.agent_collision_flag = state.get("agent_collision_flag", False)
    self.agent_coordinate_x = state.get("agent_coordinate_x", 0.0)
    self.agent_coordinate_y = state.get("agent_coordinate_y", 0.0)

    # 衝突フラグが立っている場合、前回の方向を衝突方向として記録
    if self.agent_collision_flag:
        if self.last_theta is not None:
            self.last_collision_theta = self.last_theta
            print(f"[COLLISION] Recording collision direction: {np.rad2deg(self.last_collision_theta):.1f}°")
        else:
            # last_thetaがまだない場合は、現在の方位を衝突方向として記録
            if self.agent_azimuth is not None:
                self.last_collision_theta = self.agent_azimuth
                print(f"[COLLISION] Recording collision direction (from azimuth): {np.rad2deg(self.last_collision_theta):.1f}°")
    else:
        # 衝突フラグが立っていない場合は毎ステップリセット
        self.last_collision_theta = None
```

### 5.2 衝突方向の回避

```python
# get_obstacle_density_histogram内の衝突回避処理
if self.last_collision_theta is not None:
    collision_bin = int(self.last_collision_theta * self.BIN_NUM / (2 * np.pi)) % self.BIN_NUM
    avoided_bins = 0
    for i in range(self.BIN_NUM):
        distance = min(abs(i - collision_bin), abs(i - collision_bin + self.BIN_NUM), abs(i - collision_bin - self.BIN_NUM))
        if distance < 4:  # 前回衝突した方向の周辺4bin（40度範囲）を低評価
            histogram[i] = 0.05
            avoided_bins += 1
        else:
            histogram[i] = 1.0
    print(f"[AVOIDANCE] Avoiding collision direction: {np.rad2deg(self.last_collision_theta):.1f}° (avoided {avoided_bins} bins)")
    return histogram
```

## 6. 学習モードとの統合

### 6.1 学習なしモード

```python
# デフォルトのパラメータを使用してアルゴリズムを実行
default_params = np.array([0.5, 10.0, 5.0])  # th, k_e, k_c
theta, valid_directions = self.algorithm.policy(state, default_params)
```

### 6.2 学習ありモード

```python
# モデルから学習パラメータを取得
learning_mu, learning_std, theta_mu, theta_std, value = self.model(state_vec)
learning_params = self.model.sample_learning_params(learning_mu, learning_std)

# アルゴリズムに学習パラメータを渡してポリシーを実行
theta, valid_directions = self.algorithm.policy(state, learning_params)
```

## 7. パフォーマンス最適化

### 7.1 ビン数の最適化

- **36 ビン**: 10 度間隔で方向分解能と計算効率のバランス
- **個別 bin 選択**: 全方向が有効な場合の効率的な処理
- **グループ化**: 隣接 bin の効率的な統合

### 7.2 計算効率

- **ヒストグラムキャッシュ**: 重複計算の回避
- **ベクトル化処理**: NumPy 配列の効率的な利用
- **メモリ効率**: 不要なデータの自動削除

## 8. 今後の改善方向

### 8.1 動的パラメータ調整

- 環境状態に応じたリアルタイム調整
- 探査進捗に基づく動的閾値調整
- 群の状態に応じた制御パラメータ調整

### 8.2 高度な衝突回避

- 予測的衝突回避
- 動的障害物への対応
- 群間協調による回避

### 8.3 探査効率の向上

- 未探査領域への誘導強化
- 探査パターンの最適化
- 群間協調による効率化
