# マルチエージェントシミュレーション検証システム

## 概要

このディレクトリには、マルチエージェントシミュレーションの検証用スクリプトが含まれています。各 Config（A、B、C、D）について、異なる障害物密度（0.0、0.003、0.005）での検証を実行できます。

## ファイル構成

### Config A（学習なし、分岐・統合なし）

- `verify_config_A_0.0.py` - 障害物密度: 0.0
- `verify_config_A_0.003.py` - 障害物密度: 0.003
- `verify_config_A_0.005.py` - 障害物密度: 0.005

### Config B（学習あり、分岐・統合なし）

- `verify_config_B_0.0.py` - 障害物密度: 0.0
- `verify_config_B_0.003.py` - 障害物密度: 0.003
- `verify_config_B_0.005.py` - 障害物密度: 0.005

### Config C（SystemAgent 学習あり、SwarmAgent 学習なし、分岐・統合あり）

- `verify_config_C_0.0.py` - 障害物密度: 0.0
- `verify_config_C_0.003.py` - 障害物密度: 0.003
- `verify_config_C_0.005.py` - 障害物密度: 0.005

### Config D（学習あり、分岐・統合あり）

- `verify_config_D_0.0.py` - 障害物密度: 0.0
- `verify_config_D_0.003.py` - 障害物密度: 0.003
- `verify_config_D_0.005.py` - 障害物密度: 0.005

## 検証パラメータ

### 基本設定

- **エピソード数**: 100
- **ステップ数/エピソード**: 200
- **ロボット数**: 20
- **マップサイズ**: 200×100

### 環境設定

- **探査開始位置**: (10.0, 10.0)
- **探査境界**: 内側 0.0, 外側 20.0
- **目標探査率**: 80%

### ログ設定

- **GIF 生成**: 有効
- **位置データ保存**: 有効
- **衝突データ保存**: 有効
- **サンプリングレート**: 1.0

## 検証プロセス

### 1. 環境設定

```python
def setup_verification_environment():
    """検証用環境設定"""
    sim_param = SimulationParam()
    sim_param.episodeNum = 100
    sim_param.maxStepsPerEpisode = 200
    # ... その他の設定
```

### 2. エージェント設定

各 Config に応じて異なるエージェント設定を適用：

- **Config A**: 学習なし、分岐・統合なし
- **Config B**: 学習あり、分岐・統合なし
- **Config C**: SystemAgent 学習あり、SwarmAgent 学習なし、分岐・統合あり
- **Config D**: 学習あり、分岐・統合あり

### 3. シミュレーション実行

```python
def run_verification():
    """検証実行"""
    # 1. 環境設定
    # 2. エージェント設定
    # 3. 環境作成
    # 4. エージェント作成
    # 5. エピソード実行
    # 6. 結果集計
    # 7. 結果保存
```

## 出力結果

### 保存場所

- **結果ファイル**: `verification_results/Config_[A-D]_obstacle_[密度]/verification_result.json`
- **GIF ファイル**: `verification_results/Config_[A-D]_obstacle_[密度]/episode_XXXX.gif`

### 結果データ構造

```json
{
  "config": "Config_A",
  "environment": {
    "map_size": "200x100",
    "obstacle_density": 0.0,
    "robot_count": 20
  },
  "episodes": [
    {
      "episode": 1,
      "steps_to_target": null,
      "final_exploration_rate": 0.142,
      "steps_taken": 200,
      "step_details": [...]
    }
  ],
  "summary": {
    "total_episodes": 100,
    "target_reached_episodes": 0,
    "average_exploration_rate": 0.129,
    "average_steps_taken": 200.0,
    "std_exploration_rate": 0.051,
    "std_steps_taken": 0.0
  }
}
```

## 実行方法

### 個別実行

```bash
# Config A, 障害物密度 0.0
python verify_configs/verify_config_A_0.0.py

# Config B, 障害物密度 0.003
python verify_configs/verify_config_B_0.003.py

# Config C, 障害物密度 0.005
python verify_configs/verify_config_C_0.005.py
```

### 並列実行（推奨）

複数のターミナルで同時実行することで、効率的に検証を進めることができます。

## 技術的詳細

### 修正済み問題

1. **GIF 生成エラー**: matplotlib バックエンドを`Agg`に設定
2. **JSON シリアライゼーションエラー**: `convert_numpy_types`関数を追加
3. **パス設定**: プロジェクトルートディレクトリを正しく追加

### 使用ライブラリ

- **matplotlib**: 可視化（バックエンド: Agg）
- **imageio**: GIF 生成
- **PIL**: 画像処理
- **numpy**: 数値計算
- **pandas**: データ処理

## 検証結果の解釈

### 主要指標

- **平均探査率**: 全エピソードでの平均探査率
- **目標達成率**: 80%探査率に到達したエピソードの割合
- **平均ステップ数**: 目標達成までの平均ステップ数
- **標準偏差**: 結果のばらつき

### 比較分析

- **Config 間比較**: 学習・分岐・統合の効果を評価
- **障害物密度比較**: 環境の複雑さによる影響を評価
- **統計的有意性**: 標準偏差による結果の信頼性を評価

## 注意事項

1. **実行時間**: 1 つの Config で約 30-60 分
2. **メモリ使用量**: 大量の GIF ファイル生成に注意
3. **ディスク容量**: 結果ファイルと GIF ファイルの保存容量を確認
4. **並列実行**: 複数の Config を同時実行する場合は、システムリソースを監視

## トラブルシューティング

### よくある問題

1. **ModuleNotFoundError**: プロジェクトルートディレクトリがパスに含まれていない
2. **GIF 生成エラー**: matplotlib バックエンドの問題（修正済み）
3. **JSON 保存エラー**: numpy 型のシリアライゼーション問題（修正済み）

### 解決方法

1. パス設定の確認
2. 必要なライブラリのインストール
3. システムリソースの確認
