# Analysis Scripts Directory

このディレクトリには、Multi-Agent Simulation プロジェクトの分析に使用する各種スクリプトが含まれています。

## 📊 主要分析スクリプト

### エピソード分析

- `first_episode_analysis.py` - 第 1 エピソードの基本分析
- `first_episode_detailed_analysis.py` - 第 1 エピソードの詳細分析
- `exploration_speed_analysis.py` - 探査速度の分析
- `exploration_rate_progress_single.py` - 単一エピソードの探査率進捗分析

### 設定比較分析

- `config_comparison_analysis.py` - 設定間の比較分析
- `environment_exploration_comparison.py` - 環境別探査比較
- `best_performance_comparison.py` - 最高性能比較
- `analyze_verification_results.py` - 検証結果の分析

### 分岐・統合分析

- `branch_integration_speed_test.py` - 分岐・統合速度テスト
- `test_branch_integration.py` - 分岐・統合テスト

### 高度な分析

- `advanced_analysis.py` - 高度な分析機能
- `update_episode_num.py` - エピソード番号更新

### ユーティリティ

- `generate_map_image.py` - マップ画像生成
- `convert_to_pdf.py` - PDF 変換
- `start_tensorboard.py` - TensorBoard 起動

## 🚀 実行スクリプト

### Windows 用バッチファイル

- `run_analysis.bat` - 基本分析実行
- `run_first_episode_analysis.bat` - 第 1 エピソード分析実行
- `run_first_episode_detailed.bat` - 第 1 エピソード詳細分析実行
- `run_exploration_speed_analysis.bat` - 探査速度分析実行
- `run_exploration_rate_progress_single.bat` - 探査率進捗分析実行
- `run_environment_analysis.bat` - 環境分析実行
- `run_config_comparison.bat` - 設定比較分析実行
- `run_best_performance_comparison.bat` - 最高性能比較実行
- `convert_docs_to_pdf.bat` - ドキュメント PDF 変換実行

## 📚 ドキュメント

- `ANALYSIS_README.md` - 分析全般の説明
- `FIRST_EPISODE_ANALYSIS_README.md` - 第 1 エピソード分析の説明
- `FIRST_EPISODE_DETAILED_README.md` - 第 1 エピソード詳細分析の説明
- `ENVIRONMENT_ANALYSIS_README.md` - 環境分析の説明
- `EXPLORATION_SPEED_ANALYSIS_README.md` - 探査速度分析の説明

## 🔧 使用方法

1. 必要な依存関係をインストール: `pip install -r ../requirements.txt`
2. 分析したいスクリプトを実行: `python3 script_name.py`
3. Windows の場合は対応するバッチファイルを実行

## 📁 出力先

分析結果は以下のディレクトリに保存されます：

- `../analysis_results/` - 基本分析結果
- `../first_episode_analysis_results/` - 第 1 エピソード分析結果
- `../config_comparison_results/` - 設定比較結果
- `../exploration_rate_progress_single/` - 探査率進捗結果

## ⚠️ 注意事項

- スクリプト実行前に、必要なデータファイルが存在することを確認してください
- 一部のスクリプトは特定の環境設定が必要です
- 大量のデータを処理する場合は、十分なメモリを確保してください
