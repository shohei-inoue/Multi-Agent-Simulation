# First Episode Analysis Tool

各 Config の 1 エピソード目のみを抽出して詳細比較分析を行うツールです。

## 概要

このツールは、各設定（Config A, B, C, D）の 1 エピソード目のデータのみを対象として：

- 最終探査率の比較
- ステップ数の比較
- 障害物密度の影響分析
- ステップごとの詳細な進捗分析
- 統計的有意差検定（ANOVA）
- 包括的なレポート生成

を行います。

## ファイル構成

```
first_episode_analysis.py          # メインスクリプト
run_first_episode_analysis.bat     # 実行用バッチファイル
FIRST_EPISODE_ANALYSIS_README.md   # このファイル
```

## 使用方法

### 方法 1: バッチファイル実行

```bash
run_first_episode_analysis.bat
```

### 方法 2: 直接実行

```bash
python first_episode_analysis.py
```

## 入力データ

- **データソース**: `verify_configs/verification_results/` ディレクトリ
- **対象**: 各 Config ディレクトリ内の JSON ファイル
- **抽出対象**: 各 JSON ファイルの最初のエピソード（episode: 1）のデータ

## 出力結果

### 生成されるファイル

結果は `first_episode_analysis_results/` ディレクトリに保存されます：

#### 1. グラフファイル

- `first_episode_comparison.png` - 基本比較グラフ（6 つのサブプロット）

  - Config 別最終探査率
  - Config 別ステップ数
  - 障害物密度別探査率
  - Config× 障害物密度ヒートマップ
  - 探査率分布（箱ひげ図）
  - 探査率 vs ステップ数散布図

- `first_episode_step_analysis.png` - ステップ詳細分析（6 つのサブプロット）
  - ステップごとの探査率変化
  - ステップごとの報酬変化
  - スワーム数の変化
  - 衝突発生率
  - 探査効率（探査率/ステップ）
  - 累積報酬

#### 2. 統計データ

- `first_episode_statistics.csv` - Config 別基本統計量
- `first_episode_anova.json` - ANOVA 分析結果

#### 3. レポート

- `first_episode_report.md` - サマリーレポート（Markdown 形式）

### グラフの説明

#### 基本比較グラフ

1. **Final Exploration Rate by Config**: Config 別の最終探査率平均値
2. **Steps Taken by Config**: Config 別の実行ステップ数平均値
3. **Final Exploration Rate by Obstacle Density**: 障害物密度別の探査率
4. **Exploration Rate Heatmap**: Config× 障害物密度の探査率ヒートマップ
5. **Exploration Rate Distribution**: Config 別探査率分布（箱ひげ図）
6. **Exploration Rate vs Steps**: 探査率とステップ数の関係（散布図）

#### ステップ詳細分析

1. **Exploration Rate Progress**: 1 エピソード中の探査率変化
2. **Reward Progress**: 1 エピソード中の報酬変化
3. **Swarm Count Progress**: スワーム数の変化（分岐・統合の影響）
4. **Collision Rate Progress**: 衝突発生率の変化
5. **Exploration Efficiency**: 探査効率（探査率/ステップ）の変化
6. **Cumulative Reward**: 累積報酬の変化

## 分析のポイント

### 1 エピソード目に焦点を当てる理由

- **学習の影響を排除**: 1 エピソード目は学習が進んでいない状態での純粋な初期性能
- **アルゴリズムの基本性能**: 各 Config の基本的な探査アルゴリズムの性能差
- **分岐・統合の初期効果**: 学習に依存しない分岐・統合機能の効果測定
- **公平な比較**: 全ての Config が同じ初期条件でスタートした状態での比較

### 期待される分析結果

- **Config A**: VFH-Fuzzy のみ（分岐・統合なし、学習なし）
- **Config B**: 学習済みモデル使用（分岐・統合なし）
- **Config C**: 分岐・統合あり（学習なし）
- **Config D**: 分岐・統合あり + 学習あり

## 統計分析

### ANOVA 分析

Config 間の探査率に統計的有意差があるかを検定します：

- **F 統計量**: グループ間分散/グループ内分散
- **p 値**: 有意差の確率（p < 0.05 で有意）
- **結果解釈**: Config の違いが探査性能に実際に影響するかを判定

## トラブルシューティング

### よくある問題

1. **データが見つからない**

   ```
   ❌ データディレクトリが存在しません
   ```

   - `verify_configs/verification_results/` ディレクトリが存在するか確認
   - 各 Config ディレクトリに JSON ファイルがあるか確認

2. **グラフが生成されない**

   ```
   ❌ データが空のため、グラフ生成をスキップします
   ```

   - JSON ファイルの形式が正しいか確認
   - 1 エピソード目のデータが含まれているか確認

3. **ライブラリエラー**
   ```bash
   pip install pandas matplotlib seaborn numpy scipy
   ```

### データ形式要件

JSON ファイルは以下の構造である必要があります：

```json
[
  {
    "episode": 1,
    "final_exploration_rate": 0.123,
    "steps_taken": 45,
    "total_reward": 12.34,
    "step_details": [
      {
        "step": 1,
        "exploration_rate": 0.001,
        "reward": 0.1,
        "swarm_count": 1
      }
    ]
  }
]
```

## 更新履歴

- v1.0: 初版リリース
  - 基本比較グラフ
  - ステップ詳細分析
  - 統計分析
  - レポート生成
