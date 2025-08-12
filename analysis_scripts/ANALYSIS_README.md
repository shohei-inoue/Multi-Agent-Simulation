# シミュレーション結果解析ツール

このディレクトリには、`verify_configs/verification_results`にあるシミュレーション結果を解析するためのツールが含まれています。

## 📁 ファイル構成

```
Multi-Agent-Simulation/
├── analyze_verification_results.py    # 基本的な解析スクリプト
├── advanced_analysis.py              # 高度な統計解析スクリプト
├── run_analysis.bat                  # Windows用実行バッチファイル
├── ANALYSIS_README.md                # このファイル
└── analysis_results/                 # 解析結果出力ディレクトリ（自動作成）
    ├── exploration_analysis.png      # 探査率分析グラフ
    ├── reward_analysis.png           # 報酬分析グラフ
    ├── time_series_analysis.png      # 時系列分析グラフ
    ├── clustering_analysis.png       # クラスタリング分析グラフ
    ├── verification_summary_report.md # 基本レポート
    ├── comprehensive_analysis_report.md # 包括的レポート
    ├── verification_results_summary.csv # 統計サマリー
    ├── statistical_tests.csv         # 統計検定結果
    └── clustering_results.csv        # クラスタリング結果
```

## 🚀 使用方法

### 方法 1: バッチファイルで実行（推奨）

```bash
# Windows環境で実行
run_analysis.bat
```

このバッチファイルは：

- 必要なライブラリの自動インストール
- 基本解析の実行
- 結果ディレクトリの自動オープン

を行います。

### 方法 2: 個別スクリプト実行

#### 基本解析

```bash
python analyze_verification_results.py
```

**出力結果:**

- 📊 探査率比較グラフ
- 📈 報酬分析グラフ
- 📝 サマリーレポート（Markdown）
- 📄 統計データ（CSV）

#### 高度な解析

```bash
python advanced_analysis.py
```

**出力結果:**

- 📊 時系列分析グラフ
- 🔍 PCA・クラスタリング分析
- 📈 統計的検定結果
- 📝 包括的レポート（Markdown）

## 📊 解析内容

### 基本解析 (`analyze_verification_results.py`)

1. **探査率分析**

   - Config 別平均探査率
   - 障害物密度別性能
   - Config× 密度のヒートマップ
   - 探査率分布（箱ひげ図）

2. **効率性分析**

   - 平均ステップ数比較
   - 探査率 vs ステップ数の関係
   - 報酬効率分析

3. **統計サマリー**
   - 各 Config の詳細統計
   - 性能ランキング
   - 改善提案

### 高度な解析 (`advanced_analysis.py`)

1. **統計的検定**

   - ANOVA 分析（Config 間の有意差検定）
   - ペアワイズ t 検定
   - 効果量計算

2. **時系列分析**

   - ステップごとの探査率変化
   - エピソード間の学習曲線
   - 収束性分析
   - 探査率の分散分析

3. **クラスタリング分析**
   - PCA（主成分分析）
   - K-means クラスタリング
   - 特徴量重要度分析

## 📋 必要なライブラリ

```bash
pip install pandas matplotlib seaborn numpy scipy scikit-learn
```

## 📈 解析結果の見方

### 探査率ランキング

各 Config の性能を探査率で比較：

- **高い探査率** = 効率的な探査
- **低い分散** = 安定した性能
- **統計的有意差** = 信頼できる性能差

### 障害物密度の影響

- **密度 0.0**: 障害物なし（理想環境）
- **密度 0.003**: 低密度障害物環境
- **密度 0.005**: 中密度障害物環境

### Config 比較

- **Config A**: SystemAgent 学習なし、SwarmAgent 学習なし、分岐・統合なし
- **Config B**: SystemAgent 学習なし、SwarmAgent 学習あり、分岐・統合なし
- **Config C**: SystemAgent 学習あり、SwarmAgent 学習なし、分岐・統合あり
- **Config D**: SystemAgent 学習あり、SwarmAgent 学習あり、分岐・統合あり

## 🔧 カスタマイズ

### 出力ディレクトリの変更

```python
analyzer = VerificationResultAnalyzer()
analyzer.run_analysis(output_dir="custom_results")
```

### 入力ディレクトリの変更

```python
analyzer = VerificationResultAnalyzer(results_dir="path/to/your/results")
analyzer.run_analysis()
```

### 特定の Config のみ解析

解析スクリプトを編集して、特定の Config をフィルタリング：

```python
# 例: Config AとBのみ解析
filtered_data = self.results_data.copy()
for config_name in list(filtered_data.keys()):
    if not any(x in config_name for x in ['Config_A', 'Config_B']):
        del filtered_data[config_name]
```

## 📝 レポートの活用

### 基本レポート (`verification_summary_report.md`)

- 性能ランキング
- 統計的サマリー
- 改善提案
- 推奨設定

### 包括的レポート (`comprehensive_analysis_report.md`)

- 詳細統計分析
- 統計的検定結果
- クラスタリング分析結果
- 時系列分析結果

## ⚠️ 注意事項

1. **データ形式**: JSON ファイルが正しい形式であることを確認
2. **メモリ使用量**: 大量のデータの場合、メモリ不足に注意
3. **実行時間**: 高度な解析は時間がかかる場合があります
4. **グラフ表示**: 日本語フォントが正しく表示されない場合は、フォント設定を調整

## 🆘 トラブルシューティング

### よくあるエラーと対処法

1. **ModuleNotFoundError**

   ```bash
   pip install pandas matplotlib seaborn numpy scipy scikit-learn
   ```

2. **日本語フォントが表示されない**

   - システムに適切な日本語フォントがインストールされているか確認
   - `plt.rcParams['font.family']`の設定を調整

3. **JSON ファイルが読み込めない**

   - ファイルパスが正しいか確認
   - JSON ファイルの形式が正しいか確認

4. **グラフが保存されない**
   - 出力ディレクトリの書き込み権限を確認
   - ディスク容量を確認

## 📞 サポート

問題が発生した場合は、以下の情報を含めて報告してください：

1. エラーメッセージの全文
2. 実行環境（OS、Python バージョン）
3. 入力データの構造
4. 実行したコマンド

---

**Happy Analyzing! 📊✨**
