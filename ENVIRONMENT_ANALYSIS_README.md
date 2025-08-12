# Environment Exploration Comparison Analysis

環境ごとの探査率の伸び方を比較する分析スクリプトです。

## 📋 概要

このスクリプトは、各 Config（A、B、C、D）と障害物密度（0.0、0.003、0.005）の組み合わせで、探査率がどのように向上していくかを詳細に分析し、可視化します。

## 🚀 使用方法

### 1. 直接実行

```bash
python environment_exploration_comparison.py
```

### 2. バッチファイル実行（Windows）

```bash
run_environment_analysis.bat
```

## 📊 分析内容

### 1. 環境別の探査率進行

- 各障害物密度（0.0、0.003、0.005）ごとに、Config A、B、C、D の探査率進行を比較
- ステップごとの探査率の変化を可視化

### 2. エピソード別の探査率進行

- 各エピソードでの最終探査率を比較
- 学習効果の確認

### 3. 障害物密度別の比較

- 密度による探査率への影響分析
- 密度 ×Config の組み合わせ効果

### 4. Config 別の最終探査率比較

- 各 Config の性能比較
- 学習効果と分岐・統合の影響分析

## 📁 出力ファイル

分析結果は `analysis_results/` ディレクトリに保存されます：

### グラフファイル

- `environment_exploration_progress.png` - 環境別の探査率進行
- `episode_exploration_progress.png` - エピソード別の探査率進行
- `density_comparison_analysis.png` - 障害物密度別の比較分析
- `config_comparison_analysis.png` - Config 別の比較分析

### テキストファイル

- `exploration_analysis_report.txt` - 詳細な分析レポート

## 🔍 分析の特徴

### 1. 多角的な比較

- **時間軸**: ステップごと、エピソードごと
- **設定軸**: Config A、B、C、D
- **環境軸**: 障害物密度 0.0、0.003、0.005

### 2. 詳細な統計

- 平均値、標準偏差、最小値、最大値
- 学習効果の定量化
- 最良パフォーマンスの特定

### 3. 視覚化

- 折れ線グラフ、ヒートマップ、箱ひげ図
- 高解像度（300 DPI）での保存
- 英語表記での統一

## 📈 期待される分析結果

### Config A（VFH-Fuzzy、学習なし、分岐・統合なし）

- 安定した探査率の向上
- 環境による影響の少なさ

### Config B（学習済みモデル、分岐・統合なし）

- 高い初期探査率
- 学習済みパラメータの効果

### Config C（学習あり、分岐・統合あり）

- 学習による探査率の向上
- 分岐・統合による探索効率の改善

### Config D（学習あり、分岐・統合あり）

- 学習と分岐・統合の組み合わせ効果
- 最も適応的な行動パターン

## ⚠️ 注意事項

1. **データの存在確認**: `verify_configs/verification_results/` ディレクトリに分析対象の JSON ファイルが存在することを確認してください

2. **メモリ使用量**: 大量のデータを処理する場合、十分なメモリを確保してください

3. **実行時間**: データ量に応じて分析に時間がかかる場合があります

## 🛠️ カスタマイズ

### 分析対象の変更

```python
# 結果ディレクトリの変更
analyzer = EnvironmentExplorationAnalyzer(results_dir="your_results_directory")
```

### 出力ディレクトリの変更

```python
# 出力ディレクトリの変更
self.create_exploration_progress_plots(output_dir="your_output_directory")
```

### グラフのカスタマイズ

各プロット関数（`_plot_*`）を編集することで、グラフのスタイルや内容をカスタマイズできます。

## 📞 サポート

分析スクリプトに関する質問や問題がございましたら、お気軽にお問い合わせください。
