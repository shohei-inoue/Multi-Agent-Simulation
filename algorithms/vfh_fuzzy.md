# AlgorithmVfhFuzzy: ファジィ推論に基づく方向選択アルゴリズム

## 概要

`AlgorithmVfhFuzzy` は、群ロボットのリーダーエージェントが移動方向を決定するためのファジィ推論ベースのアルゴリズムです。  
フォロワーロボットの衝突情報とエージェント自身の行動履歴を組み合わせることで、安全性と探査性を両立する行動を選択します。

本アルゴリズムは、以下の 2 要素を統合して方向を決定します：

- **走行可能性**（Obstacle Drivability）
- **探査向上性**（Exploration Improvement）

---

## パラメータ

| パラメータ | 意味                         | デフォルト |
| ---------- | ---------------------------- | ---------- |
| `th`       | 走行可能性の閾値             | 1.0        |
| `k_e`      | 探索方向抑制係数（前回方向） | 1.0        |
| `k_c`      | 探索方向抑制係数（衝突方向） | 1.0        |

---

## アルゴリズムの構成

### 1. 状態入力

```python
state = {
    "follower_collision_data": List[Tuple[distance, azimuth]],
    "agent_azimuth": float,
    "agent_collision_flag": bool,
    "agent_coordinate_x": int,
    "agent_coordinate_y": int
}
```

### 2. 走行可能性ヒストグラム ( D_i )

- 各フォロワーからの衝突情報 ((d_j, \theta_j)) に基づき、ヒストグラムビンへ加算：

```
D_i = \sum_{j} \frac{1}{d_j + \varepsilon} \cdot \delta\left(i = \left\lfloor \frac{\theta_j \cdot 180 / \pi}{\Delta \theta} \right\rfloor \right)
```

- 距離が遠い衝突は影響が小さく、近い衝突ほどスコアに大きく影響

### 3. 探査向上性ヒストグラム ( E_i )

- 前回移動方向の抑制

```
E_i \leftarrow E_i \cdot \left(1 - (1 - k_e) \cdot \exp\left(-\alpha (\Delta \theta_{i,\text{prev}})^2\right)\right)
```

- 衝突方向の抑制

```
E_i \leftarrow E_i \cdot \left(1 - (1 - k_c) \cdot \exp\left(-\beta (\Delta \theta_{i,\text{col}})^2\right)\right)
```

    •	( \alpha = 10.0, \beta = 20.0 )
    •	( \Delta \theta ) は角度差（0〜π）

- 正規化

```
E_i \leftarrow \frac{E_i + \varepsilon}{\sum_j (E_j + \varepsilon)}
```

### 4. 統合スコアヒストグラム ( R_i )

ファジィ推論として積集合を計算：

```
2R_i =
\begin{cases}
D_i \cdot E_i & \text{if } D_i \geq \text{th} \
0 & \text{otherwise}
\end{cases}
```

```
R_i \leftarrow \frac{R_i + \varepsilon}{\sum_j (R_j + \varepsilon)}
```

### 5. 重み付き確率サンプリングによる方向選択

スコアヒストグラム `R_i` を四分位点でカテゴリ分けし、以下の確率でサンプリング：

| スコアカテゴリ         | 条件            | 選択確率 |
| ---------------------- | --------------- | -------- |
| 非常に良い (Q3〜)      | `R_i ≥ Q3`      | 60%      |
| 良い (Q2〜Q3)          | `Q2 ≤ R_i < Q3` | 30%      |
| あまり良くない(Q1〜Q2) | `Q1 ≤ R_i < Q2` | 10%      |

最終的な角度は以下で決定：

```
θ = 2π × selected_bin / BIN_NUM
```

### 6. 出力形式

```python
action = {
    "theta": float,  # 選択された移動方向（ラジアン）
    "mode": int      # 行動モード（今後の拡張用）
}

action_tensor: tf.Tensor  # 学習対象パラメータ（th, k_e, k_c）
```

### 7. features

#### 特徴と利点

- `sampled_params` を渡すことで、強化学習によるパラメータ最適化が可能
- **安全性と探索性の両立**
  - 衝突回避と新規領域探索のバランスを制御可能
- **環境依存性の低さ**
  - `explored_map` に依存しない設計であり、柔軟な状態定義が可能

#### 展望
- `mode` による分岐行動（停止、旋回など）の導入
- `state` に探索済み確率やコストマップを含めて探索性強化
- `follower_collision_data` の時間履歴を用いた動的予測