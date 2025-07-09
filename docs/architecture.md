# プロジェクトアーキテクチャ

## 概要

red-group-behavior プロジェクトは、群ロボットの未知環境に対するカバレッジ問題を解くための最適化されたアーキテクチャを採用しています。

## アーキテクチャの原則

### 1. 関心の分離 (Separation of Concerns)

- 各コンポーネントは単一の責任を持つ
- 明確なインターフェースによる疎結合
- モジュール間の依存関係を最小化

### 2. 依存性注入 (Dependency Injection)

- ファクトリパターンによるコンポーネント生成
- 設定による動作の制御
- テスト容易性の向上

### 3. 設定の一元化 (Centralized Configuration)

- 全設定を一箇所で管理
- 環境に応じた自動設定
- 設定の検証と型安全性

### 4. 統一されたログ管理 (Unified Logging)

- 構造化されたログ出力
- コンポーネント別のログ管理
- メトリクスの自動収集

## ディレクトリ構造

```
red-group-behavior/
├── core/                          # コアモジュール
│   ├── __init__.py
│   ├── config.py                  # 設定管理
│   ├── interfaces.py              # 共通インターフェース
│   ├── factories.py               # ファクトリパターン
│   ├── logging.py                 # ログ管理
│   └── application.py             # メインアプリケーション
├── agents/                        # エージェント関連
│   ├── base_agent.py             # エージェント基底クラス
│   ├── agent_a2c.py              # A2Cエージェント
│   ├── agent_config.py           # エージェント設定
│   └── agent_factory.py          # エージェントファクトリ
├── algorithms/                    # アルゴリズム関連
│   ├── base_algorithm.py         # アルゴリズム基底クラス
│   ├── vfh_fuzzy.py              # VFH-Fuzzyアルゴリズム
│   └── algorithm_factory.py      # アルゴリズムファクトリ
├── envs/                          # 環境関連
│   ├── env.py                     # 探索環境
│   ├── env_map.py                 # マップ生成
│   ├── action_space.py            # アクション空間
│   ├── observation_space.py       # 状態空間
│   └── reward.py                  # 報酬設計
├── models/                        # モデル関連
│   ├── actor_critic.py            # Actor-Criticモデル
│   └── model.py                   # モデルファクトリ
├── params/                        # パラメータ関連
│   ├── simulation.py              # シミュレーション設定
│   ├── environment.py             # 環境設定
│   ├── agent.py                   # エージェント設定
│   ├── robot.py                   # ロボット設定
│   ├── explore.py                 # 探査設定
│   └── robot_logging.py           # ロボットデータ保存設定
├── robots/                        # ロボット関連
│   └── red.py                     # REDクラス
├── scores/                        # スコア関連
│   └── score.py                   # スコア計算・保存
├── utils/                         # ユーティリティ
│   ├── utils.py                   # 補助関数
│   ├── logger.py                  # ログ設定
│   └── metrics.py                 # メトリクス保存
├── docs/                          # ドキュメント
├── logs/                          # ログ出力
├── main.py                        # エントリーポイント
├── requirements.txt               # 依存パッケージ
└── README.md                      # プロジェクト説明
```

## コアコンポーネント

### 1. 設定管理 (core/config.py)

```python
class ConfigManager:
    """Central configuration manager"""

    def __init__(self):
        self.simulation = SimulationConfig()
        self.system = SystemConfig()
        self._params = {}
```

**機能:**

- シミュレーション設定の管理
- システム設定の自動検出
- ログディレクトリの自動作成
- 設定の検証と型安全性

### 2. インターフェース (core/interfaces.py)

```python
class Renderable(ABC):
    """Interface for objects that can be rendered"""

class Loggable(ABC):
    """Interface for objects that can be logged"""

class Configurable(ABC):
    """Interface for objects that can be configured"""

class SwarmMember(ABC):
    """Interface for swarm members"""
```

**機能:**

- 共通インターフェースの定義
- コンポーネント間の一貫性確保
- 拡張性の向上
- テスト容易性の向上

### 3. ファクトリパターン (core/factories.py)

```python
class BaseFactory(ABC):
    """Base factory class"""

class AlgorithmFactory(BaseFactory):
    """Factory for algorithms"""

class AgentFactory(BaseFactory):
    """Factory for agents"""
```

**機能:**

- コンポーネントの動的生成
- 依存関係の管理
- 設定による動作制御
- 拡張性の向上

### 4. ログ管理 (core/logging.py)

```python
class Logger:
    """Centralized logger"""

class ComponentLogger:
    """Logger for specific components"""
```

**機能:**

- 構造化されたログ出力
- コンポーネント別のログ管理
- メトリクスの自動収集
- TensorBoard 統合

### 5. メインアプリケーション (core/application.py)

```python
class Application:
    """Main application class"""

    def setup(self, params: Param):
        """Setup the application with parameters"""

    def run_experiment(self, experiment_config: ExperimentConfig):
        """Run a complete experiment"""
```

**機能:**

- アプリケーション全体の管理
- 実験の実行と管理
- リソースの適切な管理
- エラーハンドリング

## データフロー

### 1. 初期化フロー

```
main.py → Application.setup() → Factories → Components
```

1. メインファイルでパラメータを初期化
2. アプリケーションクラスでセットアップ
3. ファクトリが各コンポーネントを生成
4. 依存関係を解決してアプリケーションを構築

### 2. 実験実行フロー

```
Application.run_experiment() → Episode Loop → Agent → Environment → Logging
```

1. 実験設定に基づいてエピソードを実行
2. エージェントが行動を決定
3. 環境で行動を実行
4. 結果をログに記録

### 3. ログ出力フロー

```
Components → ComponentLogger → Logger → Files/TensorBoard
```

1. 各コンポーネントがイベントを記録
2. コンポーネントロガーが構造化
3. メインロガーが集約
4. ファイルと TensorBoard に出力

## 設計パターン

### 1. ファクトリパターン

- コンポーネントの動的生成
- 設定による動作制御
- 依存関係の管理

### 2. ストラテジーパターン

- アルゴリズムの動的切り替え
- エージェントの種類による動作変更
- 報酬関数の動的選択

### 3. オブザーバーパターン

- ログシステムによるイベント監視
- メトリクスの自動収集
- 状態変化の追跡

### 4. コマンドパターン

- 行動のカプセル化
- 実行履歴の管理
- アンドゥ/リドゥ機能の準備

## 拡張性

### 1. 新しいアルゴリズムの追加

```python
# 1. アルゴリズムクラスを作成
class NewAlgorithm(BaseAlgorithm):
    def policy(self, state, log_dir=None):
        # 実装
        pass

# 2. ファクトリに登録
algorithm_factory.register("new_algorithm", NewAlgorithm)
```

### 2. 新しいエージェントの追加

```python
# 1. エージェントクラスを作成
class NewAgent(BaseAgent):
    def get_action(self, state):
        # 実装
        pass

# 2. ファクトリに登録
agent_factory.register("new_agent", NewAgent)
```

### 3. 新しい環境の追加

```python
# 1. 環境クラスを作成
class NewEnvironment(gym.Env):
    def step(self, action):
        # 実装
        pass

# 2. ファクトリに登録
environment_factory.register("new_environment", NewEnvironment)
```

## パフォーマンス最適化

### 1. 非同期処理

- フォロワーの並列実行
- ログ出力の非同期化
- メモリ効率の最適化

### 2. メモリ管理

- 適切なリソース解放
- フレームキャプチャの最適化
- 不要なデータの削除

### 3. GPU 活用

- TensorFlow の自動 GPU 検出
- バッチ処理の最適化
- メモリ使用量の監視

## テスト戦略

### 1. 単体テスト

- 各コンポーネントの独立テスト
- インターフェースの検証
- エッジケースの確認

### 2. 統合テスト

- コンポーネント間の連携テスト
- データフローの検証
- エラーハンドリングの確認

### 3. パフォーマンステスト

- 実行時間の測定
- メモリ使用量の監視
- スケーラビリティの確認

## 今後の拡張

### 1. 分散処理

- マルチプロセス対応
- クラウド実行環境
- スケーラビリティの向上

### 2. リアルタイム処理

- ストリーミングデータ対応
- オンライン学習
- 動的適応

### 3. 可視化強化

- インタラクティブダッシュボード
- 3D 可視化
- リアルタイムモニタリング
