# 探査速度分析スクリプト

Config A、B、C、D の探査率向上スピードを詳細に分析・比較するツールです。

## 📋 概要

このスクリプトは、各 Config 設定での探査効率を以下の観点から分析します：

- **探査速度**: 単位時間あたりの探査率向上
- **到達時間**: 50%、80%の探査率到達にかかる時間
- **最終性能**: 最終的な探査率
- **加速度**: 探査速度の変化傾向
- **統計的有意差**: Config 間の性能差の統計検定

## 🎯 分析対象

### Config 設定

| Config | SystemAgent 学習 | SwarmAgent 学習 | 分岐・統合 | 期待される特徴            |
| ------ | ---------------- | --------------- | ---------- | ------------------------- |
| **A**  | なし             | なし            | なし       | 確率的行動、基準性能      |
| **B**  | なし             | あり            | なし       | 学習による効率化          |
| **C**  | あり             | なし            | あり       | 分岐・統合による範囲拡大  |
| **D**  | あり             | あり            | あり       | 最高性能（学習+分岐統合） |

### 環境条件

- **障害物密度**: 0.0、0.003、0.005
- **マップサイズ**: 200×100
- **ロボット数**: 20 台
- **最大ステップ数**: 200

## 🚀 実行方法

### 方法 1: バッチファイル使用（推奨）

```bash
run_exploration_speed_analysis.bat
```

### 方法 2: Python スクリプト直接実行

```bash
python exploration_speed_analysis.py
```

### 方法 3: カスタムオプション付き実行

```bash
python exploration_speed_analysis.py --results-dir verify_configs/verification_results --output-dir my_analysis
```

## 📊 分析メトリクス

### 1. 探査速度メトリクス

- **平均探査速度** (%/step): 全期間の平均的な探査率向上速度
- **最大探査速度** (%/step): 最も効率的だった期間の探査速度
- **探査加速度** (%/step²): 探査速度の変化傾向（正なら加速、負なら減速）

### 2. 到達時間メトリクス

- **50%到達時間** (steps): 探査率 50%に到達するまでのステップ数
- **80%到達時間** (steps): 探査率 80%に到達するまでのステップ数

### 3. 最終性能メトリクス

- **最終探査率** (%): エピソード終了時の探査率

## 📈 生成される分析結果

### グラフファイル

1. **`exploration_speed_comparison.png`**:

   - 6 つのメトリクスでの Config 比較
   - 障害物密度別の性能差
   - エラーバー付き統計表示

2. **`exploration_heatmaps.png`**:

   - 探査速度と最終探査率のヒートマップ
   - Config× 障害物密度の性能マトリクス

3. **`exploration_time_series.png`**:
   - 各 Config の探査進捗の時系列変化
   - 平均値と標準偏差の表示

### データファイル

- **`exploration_speed_data.csv`**: 全分析結果の生データ
- **`exploration_speed_report.md`**: 分析結果の詳細レポート

## 🔍 分析の解釈

### 探査速度の意味

- **高い平均速度**: 一貫して効率的な探査
- **高い最大速度**: 瞬間的な高効率期間の存在
- **正の加速度**: 時間とともに効率が向上（学習効果）

### Config 間の比較ポイント

1. **学習効果**: Config B vs A、Config D vs C
2. **分岐・統合効果**: Config C vs A、Config D vs B
3. **相乗効果**: Config D vs その他
4. **環境適応性**: 障害物密度による性能変化

### 統計的検定

- **ANOVA**: Config 間の全体的な有意差
- **t 検定**: Config 間のペアワイズ比較
- **p < 0.05**: 統計的有意差あり

## ⚠️ 注意事項

### データ要件

- `verify_configs/verification_results/` に各 Config の実行結果が必要
- JSON ファイルに `step_data` と `exploration_rate` の記録が必要

### 実行環境

- Python 3.7 以上
- 必要ライブラリ: pandas, matplotlib, seaborn, scipy, numpy

### トラブルシューティング

#### データが見つからない場合

```
⚠️ ディレクトリが見つかりません: verify_configs/verification_results/Config_A_obstacle_0.0
```

→ 該当 Config の検証を先に実行してください

#### グラフが表示されない場合

- Matplotlib のバックエンド設定を確認
- 日本語フォントのインストール状況を確認

#### メモリ不足エラー

- 大量のデータの場合、分析対象を絞って実行

## 📝 カスタマイズ

### 分析対象の変更

```python
# exploration_speed_analysis.py の設定変更
self.configs = ['A', 'B']  # 特定のConfigのみ
self.obstacle_densities = [0.0]  # 特定の障害物密度のみ
```

### メトリクスの追加

```python
def calculate_exploration_speed_metrics(self, steps, exploration_rates):
    # 新しいメトリクスを追加
    return {
        # 既存メトリクス...
        'custom_metric': custom_calculation(steps, exploration_rates)
    }
```

## 🎯 活用例

1. **アルゴリズム性能評価**: どの Config が最も効率的か
2. **学習効果の定量化**: 学習ありとなしの性能差
3. **環境適応性評価**: 障害物密度による性能変化
4. **最適設定の決定**: 用途に応じた最適 Config 選択

## 📞 サポート

問題や質問がある場合は、以下を確認してください：

1. 実行前に各 Config の検証結果が存在するか
2. 必要な Python ライブラリがインストールされているか
3. JSON ファイルに必要なデータ構造が含まれているか
