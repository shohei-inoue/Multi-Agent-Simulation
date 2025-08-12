# プロジェクトアーキテクチャ

## 概要

Multi-Agent-Simulation プロジェクトは、群ロボットの未知環境に対するカバレッジ問題を解くための最適化されたアーキテクチャを採用しています。本格的な強化学習による動的な群の分岐・統合システムを実装し、効率的な探査を実現します。

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

### 5. 動的群管理 (Dynamic Swarm Management)

- SystemAgent による高レベル制御
- SwarmAgent による低レベル行動
- 学習による適応的分岐・統合

## システム構成

### 1. エージェント階層

```
┌─────────────────┐
│   SystemAgent   │  ← 高レベル制御（分岐・統合判定）
│   (学習あり/なし) │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │ SwarmAgent│  ← 低レベル行動（移動方向決定）
    │(学習あり/なし)│
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ VFH-Fuzzy │  ← 行動決定アルゴリズム
    │ Algorithm │
    └───────────┘
```

### 2. Config 別の設定

| Config | SystemAgent | SwarmAgent | 分岐・統合 | 学習モード         |
| ------ | ----------- | ---------- | ---------- | ------------------ |
| **A**  | 学習なし    | 学習なし   | 無効       | 固定パラメータ     |
| **B**  | 学習なし    | 学習あり   | 無効       | 学習済みパラメータ |
| **C**  | 学習あり    | 学習なし   | 有効       | 固定パラメータ     |
| **D**  | 学習あり    | 学習あり   | 有効       | 学習済みパラメータ |

## ディレクトリ構造

```
Multi-Agent-Simulation/
├── core/                          # コアモジュール
│   ├── __init__.py
│   ├── config.py                  # 設定管理
│   ├── interfaces.py              # 共通インターフェース
│   ├── factories.py               # ファクトリパターン
│   ├── logging.py                 # ログ管理
│   └── application.py             # メインアプリケーション
├── agents/                        # エージェント関連
│   ├── base_agent.py             # エージェント基底クラス
│   ├── system_agent.py           # SystemAgent（高レベル制御）
│   ├── swarm_agent.py            # SwarmAgent（低レベル行動）
│   ├── agent_config.py           # エージェント設定
│   └── agent_factory.py          # エージェントファクトリ
├── algorithms/                    # アルゴリズム関連
│   ├── base_algorithm.py         # アルゴリズム基底クラス
│   ├── vfh_fuzzy.py              # VFH-Fuzzyアルゴリズム
│   ├── branch_algorithm.py       # 分岐アルゴリズム
│   ├── integration_algorithm.py  # 統合アルゴリズム
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
│   ├── system_agent.py           # SystemAgent設定
│   ├── swarm_agent.py            # SwarmAgent設定
│   ├── robot.py                   # ロボット設定
│   ├── explore.py                 # 探査設定
│   ├── reward.py                  # 報酬設定
│   ├── learning.py                # 学習パラメータ
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

設定の一元管理と型安全性を提供します。

```python
class Config:
    """Centralized configuration management"""

    def __init__(self):
        self.simulation = SimulationConfig()
        self.environment = EnvironmentConfig()
        self.agents = AgentConfig()
        self.learning = LearningConfig()

    def validate(self):
        """Validate all configurations"""
        # 設定の整合性チェック
        pass
```

### 2. エージェントシステム

#### SystemAgent

高レベル制御を担当し、群の分岐・統合を管理します。

```python
class SystemAgent(BaseAgent):
    """System-level agent for swarm management"""

    def __init__(self, env, algorithm, model, action_space, param):
        super().__init__(env, algorithm, model, action_space=action_space)
        self.param = param
        self.isLearning = param.learningParameter is not None
        self.learningParameter = param.learningParameter
        self.branchCondition = param.branch_condition
        self.integrationCondition = param.integration_condition

        # 群管理
        self.swarm_agents = {}  # swarm_id -> swarm_agent
        self.next_swarm_id = 0

        # 学習情報の管理
        self.learning_history = {}  # swarm_id -> 学習情報のマッピング

        # 分岐・統合のクールダウン
        self.last_branch_time = 0
        self.last_integration_time = 0

    def check_branch(self, system_state: Dict[str, Any]) -> bool:
        """分岐条件をチェックし、条件を満たせば分岐を実行"""
        # 分岐が無効になっている場合は分岐を実行しない
        if not self.branchCondition.branch_enabled:
            return False

        # 分岐条件チェック
        follower_count = system_state.get("follower_count", 0)
        valid_directions = system_state.get("valid_directions", [])
        avg_mobility = system_state.get("avg_mobility", 0.0)

        should_branch = (
            follower_count >= 3 and
            len(valid_directions) >= 2 and
            avg_mobility >= self.branch_threshold
        )

        if should_branch:
            self._execute_branch(system_state)
            return True

        return False

    def check_integration(self, system_state: Dict[str, Any]) -> bool:
        """統合条件をチェックし、条件を満たせば統合を実行"""
        # 統合が無効になっている場合は統合を実行しない
        if not self.integrationCondition.integration_enabled:
            return False

        # 統合条件チェック
        avg_mobility = system_state.get("avg_mobility", 0.0)
        swarm_count = system_state.get("swarm_count", 1)

        base_condition = (
            swarm_count >= self.integrationCondition.min_swarms_for_integration and
            (avg_mobility < self.integration_threshold or swarm_count >= 5)
        )

        if base_condition:
            self._execute_integration(system_state)
            return True

        return False
```

#### SwarmAgent

低レベル行動を担当し、VFH-Fuzzy アルゴリズムを使用して移動方向を決定します。

```python
class SwarmAgent(BaseAgent):
    """Swarm-level agent for movement control"""

    def __init__(self, env, algorithm, model, action_space, param, system_agent=None, swarm_id=None):
        super().__init__(env, algorithm, model, action_space=action_space)
        self.action_space = action_space
        self.param = param
        self.isLearning = param.isLearning
        self.learningParameter = param.learningParameter
        self.debug = param.debug
        self.system_agent = system_agent
        self.swarm_id = swarm_id

    def get_action(self, state: Dict[str, Any], episode: int = 0, log_dir: Optional[str] = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """アクションを取得（学習ログ機能付き）"""
        # 学習なしモードの場合
        if not self.isLearning or self.model is None:
            # デフォルトのパラメータを使用してアルゴリズムを実行
            default_params = np.array([0.5, 10.0, 5.0])  # th, k_e, k_c
            theta, valid_directions = self.algorithm.policy(state, default_params)

            # 分岐・統合条件の判定
            follower_scores = state.get("follower_mobility_scores", [])
            follower_count = len(follower_scores)
            avg_mobility = np.mean(follower_scores) if follower_count > 0 else 0.0

            # SystemAgentから閾値を取得
            branch_threshold = 0.5  # デフォルト値
            integration_threshold = 0.3  # デフォルト値

            should_branch = (
                follower_count >= 3 and
                valid_directions and len(valid_directions) >= 2 and
                avg_mobility >= branch_threshold
            )
            should_integrate = avg_mobility < integration_threshold

            # 分岐・統合判定
            if self.system_agent and should_branch:
                self.system_agent.check_branch(system_state)
            if self.system_agent and should_integrate:
                self.system_agent.check_integration(system_state)

            return {"theta": theta}, {
                'theta': theta,
                'valid_directions': valid_directions
            }

        # 学習ありモードの場合
        assert self.model is not None, "model must not be None"

        # 状態をテンソルに変換
        state_vec = tf.convert_to_tensor([flatten_state(state)], dtype=tf.float32)

        # モデルから学習パラメータとアクションパラメータを取得
        learning_mu, learning_std, theta_mu, theta_std, value = self.model(state_vec)

        # 学習パラメータをサンプリング
        learning_params = self.model.sample_learning_params(learning_mu, learning_std)

        # アルゴリズムに学習パラメータを渡してポリシーを実行
        theta, valid_directions = self.algorithm.policy(state, learning_params)

        # アクション（theta）をサンプリング
        action_theta = self.model.sample_action(theta_mu, theta_std)

        return {"theta": theta_value}, {
            'theta': float(action_theta) if not isinstance(action_theta, tuple) else float(action_theta[0]),
            'learning_params': learning_params.numpy().tolist() if hasattr(learning_params, 'numpy') else list(learning_params),
            'valid_directions': valid_directions,
            'value': float(value)
        }
```

### 3. VFH-Fuzzy アルゴリズム

群ロボットの移動方向決定を担当する VFH-Fuzzy アルゴリズムを実装しています。

> **詳細仕様**:
>
> - 理論的背景: [VFH-Fuzzy 理論](vfh_fuzzy_theory.md)
> - 数式仕様: [VFH-Fuzzy 推論の数式仕様](vfh_fuzzy_mathematical_specification.md)

```python
class AlgorithmVfhFuzzy():
    """VFH-Fuzzy algorithm for swarm robot navigation"""

    KAPPA = 1.0
    BIN_SIZE_DEG = 10  # ビンのサイズ（度）- 360/36=10度
    BIN_NUM = 36  # ビン数
    ANGLES = np.linspace(0, 2 * np.pi, BIN_NUM, endpoint=False)  # 角度

    def __init__(self, env=None):
        """初期化"""
        self.env = env
        self.th = 0.5
        self.k_e = 10.0
        self.k_c = 5.0

        # 状態情報
        self.agent_coordinate_x = 0.0
        self.agent_coordinate_y = 0.0
        self.agent_azimuth = None
        self.agent_collision_flag = False
        self.follower_collision_data = []

        # 衝突回避用
        self.last_theta = None
        self.last_collision_theta = None

        # ヒストグラム
        self.BIN_NUM = 36
        self.BIN_SIZE_DEG = 360 // self.BIN_NUM
        self.ANGLES = np.linspace(0, 2 * np.pi, self.BIN_NUM, endpoint=False)

        # 結果
        self.drivability_histogram = None
        self.exploration_improvement_histogram = None
        self.result_histogram = None
        self.theta = 0.0
        self.mode = 0

    def policy(self, state, sampled_params, episode=None, log_dir=None):
        """SwarmAgent用の行動決定ポリシー: thetaとvalid_directionsを返す"""
        theta, valid_directions = self.select_direction_with_candidates(state, sampled_params)
        return theta, valid_directions

    def get_obstacle_density_histogram(self):
        """障害物密度に基づく走行可能性分布を生成"""
        histogram = np.zeros(self.BIN_NUM)

        # 前回衝突した方向を避ける
        if self.last_collision_theta is not None:
            collision_bin = int(self.last_collision_theta * self.BIN_NUM / (2 * np.pi)) % self.BIN_NUM
            for i in range(self.BIN_NUM):
                distance = min(abs(i - collision_bin), abs(i - collision_bin + self.BIN_NUM), abs(i - collision_bin - self.BIN_NUM))
                if distance < 4:  # 前回衝突した方向の周辺4bin（40度範囲）を低評価
                    histogram[i] = 0.05
                else:
                    histogram[i] = 1.0
            return histogram

        # フォロワーの衝突データのみを使用
        histogram = np.ones(self.BIN_NUM, dtype=np.float32)

        for azimuth, distance in self.follower_collision_data:
            if distance < 1e-6:
                continue

            azimuth_deg = azimuth % 360.0
            bin_index = int(azimuth_deg) // self.BIN_SIZE_DEG

            obstacle_weight = 1.0 / (distance + 1e-3)
            histogram[bin_index] -= obstacle_weight

        histogram = np.maximum(histogram, 0.01)
        histogram += 1e-6
        histogram /= np.sum(histogram)

        return histogram

    def get_exploration_improvement_histogram(self) -> np.ndarray:
        """探査向上性に基づく探索欲求分布を生成"""
        histogram = np.ones(self.BIN_NUM)

        def apply_direction_weight_gauss(base_azimuth: float, k: float, sharpness: float = 10.0):
            """base_azimuth方向を中心にガウス分布で【抑制】をかける"""
            for i in range(self.BIN_NUM):
                angle = 2 * np.pi * i / self.BIN_NUM
                angle_diff = abs(angle - base_azimuth)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                decay = 1 - (1 - k) * np.exp(-sharpness * angle_diff ** 2)
                histogram[i] *= decay

        def apply_direction_weight_von(base_azimuth: float, kappa: float):
            """von Mises分布による方向重みの適用"""
            for i in range(self.BIN_NUM):
                angle = 2 * np.pi * i / self.BIN_NUM
                angle_diff = abs(angle - base_azimuth)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                decay = np.exp(kappa * np.cos(angle_diff)) / (2 * np.pi * i0(kappa))
                histogram[i] *= decay

        # 探査逆方向（過去方向）の抑制
        if hasattr(self, 'agent_azimuth') and self.agent_azimuth is not None:
            apply_direction_weight_gauss(self.agent_azimuth, self.k_e)

        # 衝突方向の抑制
        if hasattr(self, 'agent_azimuth') and self.agent_azimuth is not None:
            apply_direction_weight_von(self.agent_azimuth, self.k_c)

        histogram += 1e-6
        histogram /= np.sum(histogram)

        return histogram

    def get_final_result_histogram(self, drivability, exploration_improvement) -> np.ndarray:
        """ファジィ推論に基づき、方向ごとのスコアを統合したヒストグラムを返す"""
        fused_score = np.zeros(self.BIN_NUM, dtype=np.float32)

        # 元々の設計パラメータ
        alpha = 10.0  # ソフト抑制の鋭さ

        for bin in range(self.BIN_NUM):
            drive_val = drivability[bin]
            explore_val = exploration_improvement[bin]

            # ソフト抑制係数
            suppression = 1 / (1 + np.exp(-alpha * (drive_val - self.th)))

            # ファジィ積推論
            fused_score[bin] = suppression * drive_val * explore_val

        fused_score += 1e-6
        fused_score /= np.sum(fused_score)

        return fused_score

    def select_direction_with_candidates(self, state, sampled_params):
        """theta（選択方向）とvalid_directions（有効方向リスト）を返す"""
        self.get_state_info(state, sampled_params)
        self.drivability_histogram = self.get_obstacle_density_histogram()
        self.exploration_improvement_histogram = self.get_exploration_improvement_histogram()
        self.result_histogram = self.get_final_result_histogram(self.drivability_histogram, self.exploration_improvement_histogram)
        mean_score = np.mean(self.result_histogram)
        valid_bins = [i for i, score in enumerate(self.result_histogram) if score >= mean_score and score > 0.0]

        # 全方向が有効な場合（36個すべて）は個別のbinを選択
        if len(valid_bins) >= self.BIN_NUM * 0.8:  # 80%以上のbinが有効な場合（29個以上）
            groups = [[bin] for bin in valid_bins]
        else:
            # 隣接binをグループ化
            groups = []
            current_group = []
            for idx in range(len(valid_bins)):
                if not current_group:
                    current_group.append(valid_bins[idx])
                else:
                    prev = current_group[-1]
                    if (valid_bins[idx] == (prev + 1) % self.BIN_NUM):
                        current_group.append(valid_bins[idx])
                    else:
                        groups.append(current_group)
                        current_group = [valid_bins[idx]]
            if current_group:
                if groups and (current_group[0] == 0 and groups[0][-1] == self.BIN_NUM - 1):
                    groups[0] = current_group + groups[0]
                else:
                    groups.append(current_group)

        # 各グループの中心角度・代表スコア
        valid_directions = []
        for group in groups:
            if len(group) == 1:
                center_angle = 2 * np.pi * group[0] / self.BIN_NUM
            else:
                angles = [2 * np.pi * i / self.BIN_NUM for i in group]
                center_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

            group_score = np.mean([self.result_histogram[i] for i in group])
            valid_directions.append({"angle": center_angle, "score": group_score})

        # thetaはグループのスコアで重み付けサンプリング
        scores = np.array([d["score"] for d in valid_directions])
        if len(scores) > 0:
            probs = scores / scores.sum()
            selected_idx = np.random.choice(len(valid_directions), p=probs)
            theta = valid_directions[selected_idx]["angle"]
        else:
            theta = np.random.uniform(0, 2 * np.pi)

        # 選択した方向を記録（衝突回避用）
        self.last_theta = theta

        return theta, valid_directions
```

## 学習システム

### 1. Actor-Critic (A2C)

強化学習アルゴリズムとして Actor-Critic を採用しています。

```python
class ActorCritic:
    """Actor-Critic model for reinforcement learning"""

    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.actor = self._build_actor(state_size, action_size)
        self.critic = self._build_critic(state_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state):
        """Get action from current state"""
        state = np.array([state])
        action_probs = self.actor.predict(state)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs

    def update(self, states, actions, rewards, next_states, dones):
        """Update actor and critic networks"""
        # Actor-Critic更新ロジック
```

### 2. 学習パラメータ管理

```python
@dataclass
class LearningParameter:
    type: Literal["a2c"]
    model: Literal["actor-critic"]
    optimizer: Literal["adam"]
    gamma: float
    learningLate: float
    nStep: int
    inherit_learning_info: bool = True
    merge_learning_info: bool = True
```

## 分岐・統合システム

### 1. 分岐条件

```python
def check_branch(self, system_state: Dict[str, Any]) -> bool:
    """分岐条件をチェックし、条件を満たせば分岐を実行"""
    # 分岐が無効になっている場合は分岐を実行しない
    if not self.branchCondition.branch_enabled:
        return False

    # 分岐条件チェック
    follower_count = system_state.get("follower_count", 0)
    valid_directions = system_state.get("valid_directions", [])
    avg_mobility = system_state.get("avg_mobility", 0.0)

    should_branch = (
        follower_count >= 3 and
        len(valid_directions) >= 2 and
        avg_mobility >= self.branch_threshold
    )

    if should_branch:
        self._execute_branch(system_state)
        return True

    return False
```

### 2. 統合条件

```python
def check_integration(self, system_state: Dict[str, Any]) -> bool:
    """統合条件をチェックし、条件を満たせば統合を実行"""
    # 統合が無効になっている場合は統合を実行しない
    if not self.integrationCondition.integration_enabled:
        return False

    # 統合条件チェック
    avg_mobility = system_state.get("avg_mobility", 0.0)
    swarm_count = system_state.get("swarm_count", 1)

    base_condition = (
        swarm_count >= self.integrationCondition.min_swarms_for_integration and
        (avg_mobility < self.integration_threshold or swarm_count >= 5)
    )

    if base_condition:
        self._execute_integration(system_state)
        return True

    return False
```

## パフォーマンス最適化

### 1. メモリ効率

- 学習履歴の定期的なクリーンアップ
- 不要なデータの自動削除
- メモリ使用量の監視

### 2. 計算効率

- バッチ処理による学習の効率化
- 並列処理によるシミュレーション加速
- キャッシュ機能による重複計算の回避

### 3. 学習効率

- 適応的学習率調整
- 早期停止条件の実装
- 学習進捗の可視化

## 今後の改善方向

### 1. 動的パラメータ調整

- 環境状態に応じたリアルタイム調整
- 探査進捗に基づく動的閾値調整
- 群の状態に応じた制御パラメータ調整

### 2. 階層的学習

- 個体レベルと群レベルの同時最適化
- 群制御と個体行動の協調学習
- マルチスケール最適化

### 3. メタ学習

- 様々な環境への適応能力の向上
- 環境特性の自動認識
- 汎用的な最適化戦略の学習

### 4. 多群最適化

- 複数群の協調的最適化
- 群間の競合と協調のバランス
- 大規模群システムの最適化

### 5. 予測的群制御

- 将来の動きやすさ予測
- 予防的な群再編成
- 長期的な効率最適化
