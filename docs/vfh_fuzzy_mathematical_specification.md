# VFH-Fuzzy 推論の数式仕様

## 概要

本文書では、Multi-Agent-Simulation で実装されている VFH-Fuzzy（Vector Field Histogram with Fuzzy Inference）アルゴリズムの数学的仕様を定義する。

### アルゴリズム全体フロー

```
入力状態 → [状態情報取得] → [走行可能性計算] → [探査向上性計算] → [ファジィ推論] → [方向選択] → 出力角度
    ↓              ↓                ↓                ↓             ↓           ↓
衝突データ     衝突回避処理      前回方向抑制      ソフト抑制      四分位数分類    確率的選択
フォロワー     フォロワー        衝突方向抑制      ファジィ積      重み付け       グループ化
方位角        衝突データ        Von Mises分布     推論統合        サンプリング   中心角計算
```

## 1. 基本パラメータ

### 1.1 定数

- **ビン数**: $N = 36$ （角度分解能：10 度）
- **角度範囲**: $\theta \in [0, 2\pi)$
- **ビンサイズ**: $\Delta\theta = \frac{2\pi}{N} = \frac{\pi}{18} \approx 0.175$ [rad]
- **角度配列**: $\Theta = \{\frac{2\pi k}{N} : k = 0, 1, ..., N-1\}$

### 1.2 学習パラメータ

- **閾値パラメータ**: $th \in [0, 1]$ （走行可能性の閾値）
- **探査抑制強度**: $k_e \in [0, 50]$ （前回方向の抑制強度）
- **衝突回避強度**: $k_c \in [0, 50]$ （衝突方向の抑制強度）

## 2. 走行可能性ヒストグラム（Drivability Histogram）

### 2.1 基本定義

走行可能性ヒストグラム $D = [d_0, d_1, ..., d_{N-1}]$ は、各方向の障害物密度に基づく走行の安全性を表す。

### 2.2 衝突回避処理

前回衝突方向 $\theta_{collision}$ が存在する場合：

$$
d_i = \begin{cases}
0.05 & \text{if } \min(|i - b_{collision}|, |i - b_{collision} + N|, |i - b_{collision} - N|) < 4 \\
1.0 & \text{otherwise}
\end{cases}
$$

ここで、$b_{collision} = \lfloor \frac{\theta_{collision} \cdot N}{2\pi} \rfloor \bmod N$ は衝突方向のビンインデックス。

### 2.3 フォロワー衝突データ処理

衝突回避処理がない場合、フォロワーの衝突データ $\{(\alpha_j, r_j)\}_{j=1}^{M}$ から計算：

1. **初期化**: $d_i = 1.0, \forall i \in \{0, 1, ..., N-1\}$

2. **障害物重み計算**:
   $$w_j = \frac{1}{r_j + \epsilon}, \quad \epsilon = 10^{-3}$$

3. **ビンへの重み適用**:
   $$b_j = \lfloor \frac{\alpha_j \bmod 360°}{360°/N} \rfloor$$
   $$d_{b_j} = d_{b_j} - w_j$$

4. **クリッピングと正規化**:
   $$d_i = \max(d_i, 0.01)$$
   $$d_i = \frac{d_i + \epsilon_{norm}}{\sum_{k=0}^{N-1}(d_k + \epsilon_{norm})}, \quad \epsilon_{norm} = 10^{-6}$$

## 3. 探査向上性ヒストグラム（Exploration Improvement Histogram）

### 3.1 基本定義

探査向上性ヒストグラム $E = [e_0, e_1, ..., e_{N-1}]$ は、探査効率を向上させる方向を表す。

### 3.2 初期化

$$e_i = 1.0, \quad \forall i \in \{0, 1, ..., N-1\}$$

### 3.3 前回方向の抑制（Von Mises 分布）

エージェントの現在方位角 $\psi$ が存在する場合、逆方向 $\psi_{reverse} = (\psi + \pi) \bmod 2\pi$ を抑制：

$$e_i = e_i \times \left(1 - \frac{\exp(\kappa \cos(\theta_i - \psi_{reverse}))}{2\pi I_0(\kappa) \cdot \frac{\exp(\kappa)}{2\pi I_0(\kappa)}}\right)$$

ここで：

- $\theta_i = \frac{2\pi i}{N}$ はビン $i$ の中心角度
- $\kappa = k_e$ は Von Mises 分布の集中度パラメータ
- $I_0(\kappa)$ は第一種修正ベッセル関数

### 3.4 衝突方向の抑制

衝突フラグが立っている場合、現在方位角 $\psi$ を抑制：

$$e_i = e_i \times \left(1 - \frac{\exp(\kappa \cos(\theta_i - \psi))}{2\pi I_0(\kappa) \cdot \frac{\exp(\kappa)}{2\pi I_0(\kappa)}}\right)$$

ここで $\kappa = k_c$。

### 3.5 正規化

$$e_i = \frac{e_i + \epsilon_{norm}}{\sum_{k=0}^{N-1}(e_k + \epsilon_{norm})}, \quad \epsilon_{norm} = 10^{-6}$$

## 4. ファジィ推論による統合

### 4.1 ソフト抑制関数

各ビン $i$ に対して、走行可能性に基づくソフト抑制係数を計算：

$$s_i = \frac{1}{1 + \exp(-\alpha(d_i - th))}$$

ここで $\alpha = 10.0$ は抑制の鋭さパラメータ。

### 4.2 ファジィ積推論

最終結果ヒストグラム $R = [r_0, r_1, ..., r_{N-1}]$ は以下で計算：

$$r_i = s_i \cdot d_i \cdot e_i$$

### 4.3 正規化

$$r_i = \frac{r_i + \epsilon_{norm}}{\sum_{k=0}^{N-1}(r_k + \epsilon_{norm})}, \quad \epsilon_{norm} = 10^{-6}$$

## 5. 方向選択アルゴリズム

### 5.1 四分位数ベース分類

結果ヒストグラム $R$ の四分位数を計算：

- $Q_1$: 第 1 四分位数（25%点）
- $Q_2$: 第 2 四分位数（50%点、中央値）
- $Q_3$: 第 3 四分位数（75%点）

### 5.2 ビン分類

$$
\begin{align}
B_{very\_good} &= \{i : r_i \geq Q_3 \text{ and } r_i > 0\} \\
B_{good} &= \{i : Q_2 \leq r_i < Q_3 \text{ and } r_i > 0\} \\
B_{okay} &= \{i : Q_1 \leq r_i < Q_2 \text{ and } r_i > 0\}
\end{align}
$$

### 5.3 重み付け確率

$$
\begin{align}
w_{very\_good} &= 0.6 \text{ if } |B_{very\_good}| > 0 \text{ else } 0 \\
w_{good} &= 0.25 \text{ if } |B_{good}| > 0 \text{ else } 0 \\
w_{okay} &= 0.15 \text{ if } |B_{okay}| > 0 \text{ else } 0
\end{align}
$$

正規化重み：
$$w_{total} = w_{very\_good} + w_{good} + w_{okay}$$
$$\tilde{w}_{category} = \frac{w_{category}}{w_{total}}$$

### 5.4 最終方向選択

各カテゴリ内での均等確率：

- $B_{very\_good}$ 内の各ビン: $\frac{\tilde{w}_{very\_good}}{|B_{very\_good}|}$
- $B_{good}$ 内の各ビン: $\frac{\tilde{w}_{good}}{|B_{good}|}$
- $B_{okay}$ 内の各ビン: $\frac{\tilde{w}_{okay}}{|B_{okay}|}$

選択されたビン $i_{selected}$ から最終角度：
$$\theta_{final} = \frac{2\pi \cdot i_{selected}}{N}$$

## 6. 有効方向グループ化

### 6.1 有効ビン判定

平均スコア $\bar{r} = \frac{1}{N}\sum_{i=0}^{N-1} r_i$ に対して：
$$B_{valid} = \{i : r_i \geq \bar{r} \text{ and } r_i > 0\}$$

### 6.2 グループ化条件

- **個別処理条件**: $|B_{valid}| \geq 0.8N$ （80%以上が有効）
- **隣接グループ化**: 上記以外の場合、隣接するビンをグループ化

### 6.3 グループ中心角度計算

グループ $G = \{i_1, i_2, ..., i_k\}$ に対して：

**単一ビンの場合** ($|G| = 1$):
$$\theta_G = \frac{2\pi \cdot i_1}{N}$$

**複数ビンの場合** ($|G| > 1$):
$$\theta_G = \arctan2\left(\frac{1}{|G|}\sum_{i \in G}\sin\left(\frac{2\pi i}{N}\right), \frac{1}{|G|}\sum_{i \in G}\cos\left(\frac{2\pi i}{N}\right)\right)$$

### 6.4 グループスコア

$$score_G = \frac{1}{|G|}\sum_{i \in G} r_i$$

## 7. フォールバック処理

### 7.1 有効ビンなしの場合

1. **境界回避**: 環境マップが利用可能な場合、境界から最も遠い方向を選択
2. **逆方向**: 現在方位角の逆方向 $(\psi + \pi) \bmod 2\pi$
3. **ランダム**: $\theta \sim \mathcal{U}(0, 2\pi)$

## 8. 数学的性質

### 8.1 確率的性質

- 全ての確率分布は正規化されている：$\sum p_i = 1$
- 重み付けサンプリングにより確率的方向選択

### 8.2 連続性

- Von Mises 分布による滑らかな抑制
- シグモイド関数によるソフト閾値処理

### 8.3 適応性

- 学習パラメータ $(th, k_e, k_c)$ による行動調整
- 環境状態に応じた動的重み付け

## 9. 実装上の注意点

### 9.1 数値安定性

- ゼロ除算回避: $\epsilon = 10^{-6}$
- 最小値制限: $\min(d_i) = 0.01$

### 9.2 角度処理

- 全ての角度は $[0, 2\pi)$ 範囲で正規化
- 循環境界条件の適切な処理

### 9.3 計算効率

- ヒストグラム操作はベクトル化
- 三角関数の事前計算

---

## 10. パラメータ感度分析

### 10.1 閾値パラメータ $th$ の影響

$$\frac{\partial s_i}{\partial th} = -\frac{\alpha \exp(-\alpha(d_i - th))}{(1 + \exp(-\alpha(d_i - th)))^2}$$

- $th$ が小さいほど多くの方向が選択可能
- $th$ が大きいほど安全な方向のみ選択

### 10.2 抑制強度 $k_e, k_c$ の影響

Von Mises 分布の集中度パラメータとして：

- $k = 0$: 抑制なし（一様分布）
- $k \to \infty$: 完全抑制（該当方向のみゼロ）

## 11. 実装との対応

### 11.1 主要関数との対応

| 数式記号         | 実装関数                                        | 説明                   |
| ---------------- | ----------------------------------------------- | ---------------------- |
| $D$              | `get_obstacle_density_histogram()`              | 走行可能性ヒストグラム |
| $E$              | `get_exploration_improvement_histogram()`       | 探査向上性ヒストグラム |
| $R$              | `get_final_result_histogram()`                  | ファジィ推論結果       |
| $\theta_{final}$ | `select_final_direction_by_weighted_sampling()` | 最終方向選択           |

### 11.2 パラメータ対応

| 数式記号 | 実装変数       | 値域      |
| -------- | -------------- | --------- |
| $th$     | `self.th`      | $[0, 1]$  |
| $k_e$    | `self.k_e`     | $[0, 50]$ |
| $k_c$    | `self.k_c`     | $[0, 50]$ |
| $N$      | `self.BIN_NUM` | $36$      |
| $\alpha$ | `alpha`        | $10.0$    |

---

**更新履歴**

- 2025-08-09: 初版作成（現在の実装に基づく数式化）
