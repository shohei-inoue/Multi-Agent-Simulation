# VFH-Fuzzy アルゴリズム改善ドキュメント

## 概要

群ロボットの全探索問題において、VFH-Fuzzy アルゴリズムの探査効率と協調性を向上させるための改善を実装しました。

## 主要な改善点

### 1. 未探査領域への誘導強化

- **目的**: 探査されていない領域への積極的な誘導
- **実装**: 探査マップ情報を活用した方向選択
- **効果**: 探査率の向上と重複探査の削減

### 2. ファジィ推論の重み調整

- **目的**: 探査向上性の重要性を増加
- **実装**: `exploration_weight = 1.5` で探査向上性を重視
- **効果**: より積極的な探査行動の促進

### 3. 方向選択アルゴリズムの改善

- **目的**: 探査効率を重視した重み付け
- **実装**: 四分位数ベースの方向選択と重み調整
- **効果**: より効率的な方向選択

### 4. 環境情報の活用

- **目的**: 探査マップ情報による最適化
- **実装**: 未探査領域への誘導機能
- **効果**: 探査効率の大幅な向上

### 5. ランダム性増加機能

- **目的**: 局所解からの脱出と探査の多様性確保
- **実装**: パラメータベースのランダム性調整
- **効果**: より広範囲な探査行動

## 環境情報に依存しない最適化（最新版）

### パラメータベースの探査戦略

環境情報（探査マップ）を取得できない状況に対応するため、以下の戦略を実装：

#### 1. k_e パラメータによる探査行動の調整

- **探査強度**: `exploration_intensity = k_e / 50.0` で 0-1 に正規化
- **ランダム性増加**: `randomness_factor = 1.0 + exploration_intensity * 0.3`
- **効果**: k_e が大きいほど探査行動が積極的になる

#### 2. 方向性の強化

- **高探査モード**: `exploration_intensity > 0.5` で発動
- **離散方向強化**: 現在の方位から 90 度以上離れた方向を強化
- **効果**: より広範囲な探査行動の促進

#### 3. 群ロボットの分散促進

- **フォロワー回避**: フォロワーの衝突データを活用
- **分散抑制**: フォロワーがいる方向（45 度以内）を抑制
- **効果**: 群ロボット間の効率的な分散

### 実装された機能

#### `_apply_parameter_based_exploration(histogram)`

```python
def _apply_parameter_based_exploration(self, histogram):
    """
    パラメータベースの探査戦略（環境情報に依存しない）
    - k_eに応じた探査行動の調整
    - ランダム性と方向性のバランス
    """
    # k_eに応じた探査行動の調整
    exploration_intensity = self.k_e / 50.0  # 0-1に正規化

    # 1. ランダム性の増加（探査促進）
    randomness_factor = 1.0 + exploration_intensity * 0.3
    for i in range(self.BIN_NUM):
        histogram[i] *= (1.0 + np.random.uniform(-0.1, 0.1) * randomness_factor)

    # 2. 方向性の強化（k_eが大きいほど特定方向を重視）
    if exploration_intensity > 0.5:
        # 高探査モード：より積極的な方向選択
        for i in range(self.BIN_NUM):
            # 現在の方位から離れた方向を強化
            if self.agent_azimuth is not None:
                angle_diff = abs(2 * np.pi * i / self.BIN_NUM - self.agent_azimuth)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                if angle_diff > np.pi / 2:  # 90度以上離れた方向
                    histogram[i] *= (1.0 + exploration_intensity * 0.5)

    # 3. 群ロボットの分散促進
    # フォロワーの衝突データを活用して分散を促進
    if len(self.follower_collision_data) > 0:
        for azimuth, distance in self.follower_collision_data:
            if distance < 1e-6:
                continue

            # フォロワーがいる方向を避ける（分散促進）
            for i in range(self.BIN_NUM):
                angle = 2 * np.pi * i / self.BIN_NUM
                angle_diff = min(abs(angle - azimuth), 2 * np.pi - abs(angle - azimuth))
                if angle_diff < np.pi / 4:  # 45度以内の方向を抑制
                    histogram[i] *= (1.0 - exploration_intensity * 0.3)
```

## パラメータ最適化戦略

### 推奨パラメータ範囲

- **th**: 0.5 - 2.0（走行可能性の閾値）
- **k_e**: 10.0 - 50.0（探査向上性の重み）
- **k_c**: 5.0 - 20.0（衝突回避の重み）

### 学習戦略

1. **初期探索**: 広いパラメータ範囲での探索
2. **局所最適化**: 有望な領域での詳細な調整
3. **動的調整**: 探査率に応じたパラメータの動的変更

## 期待される効果

### 探査効率の向上

- より効率的な方向選択
- 重複探査の削減
- 探査率の大幅な向上

### 協調性の向上

- 群ロボット間の効率的な分散
- 衝突回避の強化
- 協調的な探査行動

### 汎用性の向上

- 環境情報に依存しない設計
- 様々な環境での適用可能性
- パラメータ調整による柔軟性

## 今後の改善方向

1. **適応的パラメータ調整**: 探査状況に応じた動的パラメータ調整
2. **マルチエージェント学習**: 群ロボット間の協調学習
3. **環境適応**: 異なる環境での自動適応機能
4. **リアルタイム最適化**: 実行時のパラメータ最適化
