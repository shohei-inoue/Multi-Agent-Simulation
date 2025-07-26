# 実験ディレクトリ

このディレクトリには、SystemAgent と SwarmAgent の学習有無による性能比較実験のコードが含まれています。

## 📋 実験概要

### 設計方針

**既存のシステム設計を尊重**: 元々実装されていた`isLearning`パラメータと`learningParameter`構造を活用し、既存の動作に影響を与えないよう配慮しています。

### 4 つの実験構成

1. **Config_A**: SystemAgent(学習なし, 分岐なし) + SwarmAgent(学習なし)

   - `system_agent_param.learningParameter = None`
   - `swarm_agent_param.isLearning = False`

2. **Config_B**: SystemAgent(学習なし, 分岐なし) + SwarmAgent(学習済み)

   - `system_agent_param.learningParameter = None`
   - `swarm_agent_param.isLearning = True`

3. **Config_C**: SystemAgent(学習なし, 分岐あり) + SwarmAgent(学習なし)

   - `system_agent_param.learningParameter = None`
   - `swarm_agent_param.isLearning = False`
   - `branch_condition.branch_enabled = True`

4. **Config_D**: SystemAgent(学習済み, 分岐あり) + SwarmAgent(学習済み)
   - `system_agent_param.learningParameter = LearningParameter()`
   - `swarm_agent_param.isLearning = True`

### 評価指標

- **探査速度**: 探査率 80%達成までのステップ数
- **探査進捗**: 単位時間あたりの探査率向上
- **ロバスト性**: 環境変化に対する性能の安定性
- **統計的信頼性**: 複数回実行による平均値、標準偏差、信頼区間

### 統計的信頼性

探査は確率的なプロセスであるため、以下の統計分析を実装：

- **複数回実行**: 各構成・環境で 5 回実行（デフォルト）
- **統計量計算**: 平均値、標準偏差、最小値、最大値
- **信頼区間**: 95%信頼区間の計算
- **エラーバー**: グラフに標準偏差を表示

### 環境設定

- **マップサイズ**: 100×200
- **ロボット数**: 20 台
- **障害物密度**: 0.0, 0.003, 0.005
- **ステップ数設定**:
  - 1 領域あたりの follower の各探査ステップ: 30
  - 1 エピソードあたりの最大ステップ数: 200

## 🚀 実行方法

### 1. 事前学習済みモデルの作成

```bash
# 両方のモデルを学習
python train_pretrained_models.py --train-both

# 個別に学習
python train_pretrained_models.py --train-system
python train_pretrained_models.py --train-swarm

# カスタム設定
python train_pretrained_models.py \
    --episodes 300 \
    --steps 60 \
    --output-dir my_pretrained_models
```

### 2. 比較実験の実行

```bash
# 基本的な実行
python run_comparison_experiment.py

# クイックモード（短時間でテスト）
python run_comparison_experiment.py --quick

# 事前学習済みモデルなしで実行
python run_comparison_experiment.py --no-pretrained
```

### オプション詳細

#### train_pretrained_models.py

- `--episodes`: 学習エピソード数 (default: 200)
- `--steps`: エピソードあたりの最大ステップ数 (default: 50)
- `--output-dir`: モデル保存ディレクトリ (default: pretrained_models)
- `--train-system`: SystemAgent モデルを学習
- `--train-swarm`: SwarmAgent モデルを学習
- `--train-both`: 両方のモデルを学習（デフォルト）

#### run_comparison_experiment.py

- `--episodes`: エピソード数 (default: 50)
- `--steps`: エピソードあたりの最大ステップ数 (default: 200)
- `--target-rate`: 目標探査率 (default: 0.8)
- `--output-dir`: 結果出力ディレクトリ (default: experiment_results)
- `--quick`: クイックモード（エピソード数とステップ数を減らす）
- `--verbose`: 詳細なログ出力
- `--no-pretrained`: 事前学習済みモデルを使用しない
- `--num-runs`: 統計的信頼性のための複数回実行数 (default: 5)

## 📊 出力結果

### 統計情報

各実験結果には以下の統計情報が含まれます：

- **基本統計量**: 平均値、標準偏差、最小値、最大値
- **信頼区間**: 95%信頼区間（下限、上限、誤差範囲）
- **実行回数**: 統計計算に使用した実行回数
- **エラーバー**: グラフに標準偏差を表示

### 結果ファイル

```
experiment_results/
├── performance_comparison_YYYYMMDD_HHMMSS.png  # 性能比較グラフ（統計情報付き）
├── robustness_comparison_YYYYMMDD_HHMMSS.png   # ロバスト性比較
├── speed_comparison_YYYYMMDD_HHMMSS.png        # 速度比較
├── obstacle_density_impact_YYYYMMDD_HHMMSS.png # 障害物密度影響
└── results_YYYYMMDD_HHMMSS.json               # 詳細結果データ（統計情報含む）
```

### 結果の解釈

#### 性能比較グラフ

- **目標達成率**: 各構成が 80%探査率を達成した割合
- **平均ステップ数**: 目標達成までの平均ステップ数
- **探査速度**: 単位ステップあたりの探査率向上

#### ロバスト性比較グラフ

- **性能標準偏差**: 環境変化に対する性能の安定性
- 値が小さいほど、環境変化に強い（ロバスト）

#### 速度比較グラフ

- **平均速度**: 全環境での平均ステップ数
- **エラーバー**: 環境間での速度のばらつき

#### 障害物密度影響グラフ

- **性能変化**: 障害物密度による性能の変化
- **劣化率**: 障害物密度増加による性能劣化の程度

## 🔧 実験のカスタマイズ

### 新しい環境設定の追加

`comparison_experiment.py`の`create_environment_configs()`メソッドを編集：

```python
def create_environment_configs(self) -> List[EnvironmentConfig]:
    configs = [
        # 既存の設定...
        EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.01, robot_count=20),  # 新しい設定
    ]
    return configs
```

### ステップ数設定の固定化

検証設定として以下の値を固定：

```python
def estimate_steps_for_target(self, env_config: EnvironmentConfig) -> int:
    # 固定設定: 1エピソードあたりの最大ステップ数を200に設定
    # 1領域あたりのfollowerの各探査ステップを30として計算

    # 最小・最大値を設定
    estimated_steps = max(30, min(estimated_steps, 200))
```

### 新しい評価指標の追加

`ComparisonExperiment`クラスに新しい分析メソッドを追加：

```python
def _analyze_new_metric(self) -> Dict[str, Any]:
    """新しい評価指標の分析"""
    # 実装...
    return new_metric_data
```

## 📈 期待される結果

### 仮説

1. **Config_D** (両方学習済み) が最も高性能
2. **Config_B** (SwarmAgent のみ学習済み) が中程度の性能
3. **Config_C** (分岐のみ) が限定的な改善
4. **Config_A** (学習なし) が最低性能

### 検証ポイント

- 学習による探査効率の向上
- 分岐・統合による適応性の向上
- 環境変化に対するロバスト性
- 障害物密度による性能劣化の程度
- 学習コストと性能向上のトレードオフ

## 🐛 トラブルシューティング

### よくある問題

1. **メモリ不足**: `--quick`オプションを使用
2. **実行時間が長い**: エピソード数やステップ数を減らす
3. **結果が保存されない**: 出力ディレクトリの権限を確認
4. **事前学習済みモデルが見つからない**: `train_pretrained_models.py`を実行

### デバッグ

```bash
# 詳細ログ付きで実行
python run_comparison_experiment.py --verbose

# 最小設定でテスト
python run_comparison_experiment.py --quick --episodes 5 --steps 10

# 事前学習済みモデルなしでテスト
python run_comparison_experiment.py --no-pretrained --quick
```

### 事前学習のトラブルシューティング

```bash
# 学習状況の確認
python train_pretrained_models.py --train-both --episodes 50 --steps 20

# 個別モデルの学習
python train_pretrained_models.py --train-system --episodes 100
python train_pretrained_models.py --train-swarm --episodes 100
```

## 🔄 既存システムとの互換性

### パラメータ構造の維持

- **SwarmAgent**: `isLearning`と`learningParameter`を使用
- **SystemAgent**: `learningParameter`を使用
- **分岐・統合**: `branch_condition`と`integration_condition`を使用

### 既存の動作への影響

- 実験用の設定変更は一時的
- 元のパラメータ構造を完全に維持
- 既存のシミュレーション実行に影響なし

## 📚 関連ドキュメント

- [アーキテクチャ設計](../docs/architecture.md)
- [学習システム](../docs/learning_system.md)
- [群最適化](../docs/swarm_optimization.md)
- [VFH-Fuzzy アルゴリズム](../docs/vfh_fuzzy.md)
