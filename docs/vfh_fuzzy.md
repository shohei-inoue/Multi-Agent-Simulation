# Algorithm: VFH-Fuzzy 行動決定法

## 1. 入力とパラメータ

- `drivability_histogram`：障害物密度に基づく走行可能性分布  
- `exploration_improvement_histogram`：探査向上性に基づく探索欲求分布  
- `th`: 抑制のしきい値（しきい値以下のdrivabilityは抑制）  
- `k_e`: 探査逆方向の抑制強度  
- `k_c`: 衝突方向の抑制強度  
- `α`: ソフト抑制の鋭さ（固定: α=10.0）

## 2. 探査向上性の抑制関数

### a. 過去方向・衝突方向の抑制

- 各ビン \( i \) の角度差 \( \Delta \theta_i \) に対して：

\[
\text{decay}_i = 1 - (1 - k) \cdot \exp(-s \cdot \Delta \theta_i^2)
\]

- 探査逆方向（過去方向）→ `k = k_e`, `s = 10.0`  
- 衝突方向 → `k = k_c`, `s = 20.0`

- 総スコア：

\[
E_i = 1 \cdot \prod_{\text{penalty}} \text{decay}_i
\]

---

## 3. ソフト抑制付きファジィ推論（最終スコア）

- 各ビン \( i \) における走行可能性 \( D_i \)、探索向上性 \( E_i \)

- 抑制項：

\[
\text{suppression}_i = \frac{1}{1 + \exp(-\alpha \cdot (D_i - th))}
\]

- 最終スコア：

\[
F_i = \text{suppression}_i \cdot D_i \cdot E_i
\]

---

## 4. 最終方向選択（重み付きサンプリング）

- スコアヒストグラム \( \{F_i\} \) の四分位点：

\[
Q_1 = 25\%,\quad Q_2 = 50\%,\quad Q_3 = 75\%
\]

- ビンクラス分類：

| ビン | 条件                          | 重み |
|------|-------------------------------|------|
| Very Good | \( F_i \ge Q_3 \)           | 0.6  |
| Good      | \( Q_2 \le F_i < Q_3 \)     | 0.25 |
| Okay      | \( Q_1 \le F_i < Q_2 \)     | 0.15 |

- クラスごとに等確率で選択し、最終方向 \( \theta \) を決定：

\[
\theta = \frac{2\pi \cdot \text{selected\_bin}}{\text{BIN\_NUM}}
\]

---

## 5. 出力

- 行動：  
  - `theta`：移動方向（ラジアン）  
  - `mode`：現在は固定で 0

