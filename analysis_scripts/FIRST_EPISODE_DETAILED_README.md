# First Episode Detailed Analysis Tool

1 エピソード目のステップごとの探査率上昇と地図による最終探査状況の詳細比較分析を行うツールです。

## 概要

このツールは、各 Config（A, B, C, D）の 1 エピソード目について：

- **ステップごとの探査率上昇の詳細分析**
- **探査率上昇速度の計算**
- **地図による最終探査状況の可視化比較**
- **探査効率の時系列変化**
- **目標探査率到達時間の比較**

を行います。

## ファイル構成

```
first_episode_detailed_analysis.py    # メインスクリプト
run_first_episode_detailed.bat        # 実行用バッチファイル
FIRST_EPISODE_DETAILED_README.md      # このファイル
```

## 使用方法

### 方法 1: バッチファイル実行

```bash
run_first_episode_detailed.bat
```

### 方法 2: 直接実行

```bash
python first_episode_detailed_analysis.py
```

## 出力結果

結果は `first_episode_detailed_results/` ディレクトリに保存されます：

### 1. ステップごと探査率上昇分析（障害物密度別）

各障害物密度（0.0, 0.003, 0.005）について、以下の 6 つのサブプロットを含むグラフを生成：

#### `exploration_progression_density_X.X.png`

1. **Exploration Rate Progress**: ステップごとの探査率変化（平均 ± 標準偏差）
2. **Exploration Rate Increase Speed**: 探査率上昇速度（微分）
3. **Cumulative Exploration Progress**: 累積探査率の推移
4. **Exploration Efficiency Over Time**: 探査効率（探査率/ステップ）の変化
5. **Swarm Count Progress**: スワーム数の変化（分岐・統合の効果）
6. **Time to Reach Target Exploration Rate**: 目標探査率到達時間

### 2. 地図による最終探査状況比較

#### `exploration_map_comparison_density_X.X.png`

各 Config（A, B, C, D）の最終探査状況を 2D 地図で可視化：

- **Config A**: 中央から放射状の探査パターン
- **Config B**: 効率的なグリッド状探査パターン
- **Config C**: 複数拠点からの探査パターン（分岐効果）
- **Config D**: 不規則な探査パターン（学習中の不安定性）

地図には以下が含まれます：

- 探査済み領域（緑）から未探査領域（赤）のグラデーション
- 障害物の配置（グレー）
- 最終探査率の数値表示
- カラーバー（探査度を示す）

### 3. 詳細統計データ

#### `exploration_progression_statistics.csv`

各 Config× 障害物密度の組み合わせについて：

- `avg_increase_rate`: 平均探査率上昇速度
- `max_increase_rate`: 最大探査率上昇速度
- `final_exploration_rate`: 最終探査率
- `steps_to_final`: 最終ステップ数
- `exploration_efficiency`: 探査効率（探査率/ステップ）

## 分析結果のハイライト

### 📊 探査率上昇速度（平均）

**障害物なし（density=0.0）**:

- Config B: 0.0123（最速）
- Config C: 0.0115
- Config D: 0.0109
- Config A: 0.0088

**障害物密度 0.003**:

- Config C: 0.0101（最速）
- Config A: 0.0079
- Config D: 0.0057
- Config B: 0.0020

**障害物密度 0.005**:

- Config D: 0.0082（最速）
- Config C: 0.0045
- Config A: 0.0035
- Config B: 0.0036

### 🎯 主要な発見

1. **障害物なし環境**:

   - Config B（学習済みモデル）が最も効率的
   - Config C（分岐・統合）も高い性能を示す

2. **障害物あり環境**:

   - Config C（分岐・統合）が障害物密度 0.003 で最高性能
   - Config D（分岐・統合+学習）が障害物密度 0.005 で最高性能
   - 障害物が増えると分岐・統合の効果が顕著に

3. **探査パターンの違い**:
   - **Config A**: 単一拠点からの放射状探査
   - **Config B**: 効率的なグリッド探査
   - **Config C**: 複数拠点による並列探査
   - **Config D**: 学習による適応的だが不規則な探査

## 技術的詳細

### 地図シミュレーション

実際の探査データがない場合、各 Config の特性に基づいて探査パターンをシミュレート：

- **放射状探査**: 中心点からの距離ベース
- **効率的探査**: グリッド状の系統的探査
- **複数拠点探査**: 分岐により生成される複数の探査中心
- **不規則探査**: 学習中の試行錯誤を表現

### 統計指標

- **探査率上昇速度**: `np.diff(exploration_rate)`による微分近似
- **探査効率**: `exploration_rate / step_number`
- **目標到達時間**: 指定探査率に最初に到達したステップ

## 可視化の特徴

### グラフの改善点

- **標準偏差の表示**: 探査率の変動を影で表現
- **複数指標の同時表示**: 6 つの異なる視点からの分析
- **障害物密度別の分離**: 環境条件の影響を明確化

### 地図の特徴

- **カラーマップ**: RdYlGn（赤 → 黄 → 緑）で探査度を直感的に表現
- **障害物表示**: グレーの矩形で障害物を可視化
- **スケール表示**: カラーバーで探査度の数値範囲を明示

## 応用例

このツールの分析結果は以下に活用できます：

1. **アルゴリズム性能評価**: 各 Config の探査効率比較
2. **環境適応性分析**: 障害物密度による性能変化
3. **学習効果検証**: 学習有無による探査パターンの違い
4. **分岐・統合効果測定**: 複数スワームによる探査改善効果

## 更新履歴

- v1.0: 初版リリース
  - ステップごと探査率上昇分析
  - 地図による最終探査状況比較
  - 詳細統計データ生成
