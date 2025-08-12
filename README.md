# Multi-Agent-Simulation - Swarm Robot Exploration

群ロボットによる未知環境探査シミュレーションシステム。本格的な強化学習による動的な群の分岐・統合、SystemAgent と SwarmAgent の階層的制御、200×200 マップでの効率的な探査を実現。

## 🚀 特徴

- **階層的エージェントシステム**: SystemAgent（高レベル制御）と SwarmAgent（低レベル行動）の分離
- **動的群分岐・統合**: 学習による適応的な群管理
- **本格的な強化学習**: 100 エピソード、50 ステップ/エピソードの大規模学習
- **VFH-Fuzzy アルゴリズム**: 衝突回避と探査効率の最適化
- **200×200 マップ**: 広大な探査空間での大規模シミュレーション
- **学習情報の継承・統合**: 分岐・統合時の知識移転
- **リアルタイム可視化**: 探査過程の可視化と GIF 保存
- **モジュラーアーキテクチャ**: 保守性と拡張性を重視した設計

## 🎯 シミュレーション概要

### 群ロボット探査システム

- **目的**: 未知環境での効率的な探査
- **ロボット数**: 設定可能（デフォルト: 10 台）
- **環境**: 200×200 マップ（障害物を含む 2 次元マップ）
- **探査目標**: 指定された探査率の達成
- **学習規模**: 100 エピソード × 50 ステップ/エピソード

### 主要機能

#### 1. 階層的エージェントシステム

- **SystemAgent**: 高レベル制御（分岐・統合判断、学習閾値の最適化）
- **SwarmAgent**: 低レベル行動（VFH-Fuzzy 移動、衝突回避学習）
- **協調学習**: 両エージェントの協調的な学習

#### 2. 動的群分岐・統合システム

- **学習可能な閾値**: 分岐・統合の判断を学習で最適化
- **分岐アルゴリズム**: MobilityBasedBranchAlgorithm、RandomBranchAlgorithm
- **統合アルゴリズム**: NearestIntegrationAlgorithm
- **探査領域重複チェック**: 遠距離統合の防止
- **最小群サイズ制限**: 3 台以下の群は分岐しない

#### 3. VFH-Fuzzy 学習システム

- **パラメータ最適化**: th（閾値）、k_e（探査抑制）、k_c（衝突抑制）
- **衝突方向の確率低下**: 学習による衝突回避の改善
- **フロンティアベース探査**: 効率的な未探査領域への移動

#### 4. 学習情報の継承・統合

- **分岐時の学習継承**: 新しい群への知識移転
- **統合時の学習統合**: 重み付き平均による知識集約
- **経験バッファの共有**: 学習経験の効率的な活用

#### 5. 可視化システム

- **リアルタイム描画**: 探査過程のリアルタイム表示
- **GIF 自動保存**: エピソードごとの GIF 自動生成
- **軌跡表示**: リーダーごとの色分けされた軌跡
- **統合リーダー軌跡**: 統合されたリーダーの軌跡保持

## 🔬 アルゴリズムと数式

### 1. VFH-Fuzzy アルゴリズム

#### 基本概念

VFH (Vector Field Histogram) Fuzzy は、障害物回避と目標指向行動を組み合わせたナビゲーションアルゴリズムです。学習によりパラメータが最適化され、衝突方向の確率が低下します。

#### 学習パラメータ

**1. 閾値パラメータ (th)**

```
th_new = th_current + Δth * learning_rate
```

- 学習により上昇すると、衝突方向の抑制が強くなる

**2. 探査抑制パラメータ (k_e)**

```
k_e_new = k_e_current + Δk_e * learning_rate
```

- 学習により調整され、探査効率が最適化される

**3. 衝突抑制パラメータ (k_c)**

```
k_c_new = k_c_current + Δk_c * learning_rate
```

- 学習により上昇すると、衝突方向の確率が減少

#### 衝突回避メカニズム

```python
# 走行可能性ヒストグラム
risk_histogram[i] += angle_weight * dist_weight
drivability = 1.0 - risk_histogram  # 反転処理

# 探査向上性ヒストグラム
if self.agent_collision_flag:
    apply_direction_weight_von(self.agent_azimuth, self.k_c)

# 最終結果ヒストグラム
suppression = 1 / (1 + np.exp(-alpha * (drive_val - self.th)))
```

### 2. 分岐アルゴリズム

#### MobilityBasedBranchAlgorithm

```python
def select_branch_followers(self, source_swarm, valid_directions, mobility_scores):
    """mobility_scoreに基づいてフォロワーを選択"""
    # 最低6人のフォロワーが必要
    if len(followers) < 6:
        return []

    # mobility_scoreでソート
    follower_scores.sort(key=lambda x: x[1], reverse=True)

    # 上位のフォロワーを新しい群に割り当て
    selected_followers = [follower for follower, _ in follower_scores[:split_count]]
    return selected_followers

def select_branch_leader(self, source_swarm, valid_directions, mobility_scores):
    """mobility_scoreが最も高いfollowerを新しいleaderとして選択"""
    best_follower = None
    best_score = -1.0

    for i, follower in enumerate(source_swarm.followers):
        score = mobility_scores[i] if i < len(mobility_scores) else 0.5
        if score > best_score:
            best_score = score
            best_follower = follower

    return best_follower
```

#### 分岐条件

```python
def check_branch_condition(self, direction_count, mobility_score, follower_count):
    """分岐条件のチェック"""
    return (
        self.branch_enabled and
        direction_count >= self.min_directions and  # 2以上
        mobility_score >= self.branch_threshold and  # 0.3以上
        follower_count >= self.min_followers_for_branch  # 3以上
    )
```

### 3. 統合アルゴリズム

#### NearestIntegrationAlgorithm

```python
def select_integration_target(self, source_swarm, target_swarms):
    """最も近い群を統合対象として選択"""
    for target_swarm in target_swarms:
        # 探査領域の重複をチェック
        if not self.env.check_exploration_area_overlap(
            source_swarm.swarm_id, target_swarm.swarm_id
        ):
            continue  # 探査領域が重複していない場合はスキップ

        distance = np.linalg.norm(source_pos - target_pos)
        if distance < min_distance:
            min_distance = distance
            nearest_swarm = target_swarm

    return nearest_swarm
```

#### 統合条件

```python
def check_integration_condition(self, mobility_score, swarm_count, exploration_overlap):
    """統合条件のチェック"""
    return (
        self.integration_enabled and
        mobility_score < self.integration_threshold and  # 0.2以下
        swarm_count >= self.min_swarms_for_integration and  # 2以上
        exploration_overlap and  # 探査領域が重複
        self.check_cooldown()  # クールダウン期間（15秒）
    )
```

### 4. 学習システム

#### Actor-Critic (A2C)

```python
class ActorCritic:
    """Actor-Critic model for reinforcement learning"""

    def get_action(self, state):
        """状態から行動を選択"""
        action_probs = self.actor.predict(state)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs

    def update(self, states, actions, rewards, next_states, dones):
        """ActorとCriticネットワークの更新"""
        # Actor-Critic更新ロジック
        pass
```

#### 学習情報の継承・統合

```python
def inherit_learning_info(self, source_swarm, new_swarm):
    """分岐時の学習情報の継承"""
    # 学習パラメータの継承
    new_swarm.learning_params = source_swarm.learning_params.copy()

    # VFH-Fuzzyパラメータの継承
    new_swarm.algorithm.th = source_swarm.algorithm.th
    new_swarm.algorithm.k_e = source_swarm.algorithm.k_e
    new_swarm.algorithm.k_c = source_swarm.algorithm.k_c

def merge_learning_info(self, source_swarm, target_swarm):
    """統合時の学習情報の統合"""
    # 重み付き平均による統合
    source_weight = 0.3
    target_weight = 0.7

    for param_name in ['th', 'k_e', 'k_c']:
        source_value = getattr(source_swarm.algorithm, param_name)
        target_value = getattr(target_swarm.algorithm, param_name)
        merged_value = (source_value * source_weight + target_value * target_weight)
        setattr(target_swarm.algorithm, param_name, merged_value)
```

### 5. 報酬設計

#### SwarmAgent 報酬

```python
def calculate_swarm_reward(self, collision_flag, exploration_improvement, movement_distance):
    """SwarmAgentの報酬計算"""
    reward = 0.0

    # 衝突ペナルティ
    if collision_flag:
        reward -= 5.0

    # 探査向上報酬
    reward += exploration_improvement * 10.0

    # 移動距離報酬
    reward += movement_distance * 0.1

    return reward
```

#### SystemAgent 報酬

```python
def calculate_system_reward(self, swarm_state):
    """SystemAgentの報酬計算"""
    reward = 0.0

    # 探査効率報酬
    exploration_efficiency = swarm_state.get('exploration_efficiency', 0.0)
    reward += exploration_efficiency * 10.0

    # 群数バランス報酬
    swarm_count_balance = self.calculate_swarm_count_balance()
    reward += swarm_count_balance * 2.0

    # 移動性スコア報酬
    mobility_score = swarm_state.get('mobility_score', 0.0)
    reward += mobility_score * 5.0

    return reward
```

## 📁 プロジェクト構造

```
Multi-Agent-Simulation/
├── core/                          # コアモジュール
│   ├── __init__.py
│   ├── application.py             # メインアプリケーション
│   ├── config.py                  # 設定管理
│   ├── interfaces.py              # 共通インターフェース
│   ├── factories.py               # ファクトリパターン
│   └── logging.py                 # ログ管理
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
│   ├── env.py                     # 探索環境（200×200マップ）
│   ├── env_map.py                 # マップ生成
│   ├── action_space.py            # アクション空間
│   ├── observation_space.py       # 状態空間
│   └── reward.py                  # 報酬設計
├── models/                        # モデル関連
│   ├── actor_critic.py            # Actor-Criticモデル
│   └── model.py                   # モデルファクトリ
├── params/                        # パラメータ関連
│   ├── simulation.py              # シミュレーション設定
│   ├── environment.py             # 環境設定（200×200）
│   ├── agent.py                   # エージェント設定（100エピソード）
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
├── analysis_scripts/              # 分析スクリプト
│   ├── first_episode_analysis.py  # 第1エピソード分析
│   ├── config_comparison_analysis.py # 設定比較分析
│   ├── exploration_speed_analysis.py # 探査速度分析
│   ├── analyze_verification_results.py # 検証結果分析
│   └── README.md                  # 分析スクリプト説明
├── docs/                          # ドキュメント
│   ├── architecture.md            # アーキテクチャ設計
│   ├── branch_algorithm.md        # 分岐アルゴリズム
│   ├── integration_algorithm.md   # 統合アルゴリズム
│   ├── learning_system.md         # 学習システム
│   ├── reward_system.md           # 報酬システム
│   ├── vfh_fuzzy.md               # VFH-Fuzzyアルゴリズム
│   └── swarm_optimization.md      # 群最適化
├── main.py                        # メインエントリーポイント
├── requirements.txt               # 依存関係
└── README.md                      # このファイル
```

## 🛠️ セットアップ

### 必要条件

- Python 3.12+
- TensorFlow 2.15+
- Docker & Docker Compose（推奨）
- NumPy
- Matplotlib
- Pandas
- PIL (Pillow)
- imageio

### インストール

#### 方法 1: Docker（推奨）

```bash
# リポジトリをクローン
git clone <repository-url>
cd Multi-Agent-Simulation

# Dockerコンテナをビルドして実行
docker-compose up --build
```

#### 方法 2: ローカル環境

```bash
# リポジトリをクローン
git clone <repository-url>
cd Multi-Agent-Simulation

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

## 🚀 使用方法

### 基本的な実行

#### Docker 環境

```bash
# コンテナを起動
docker-compose up

# バックグラウンドで実行
docker-compose up -d

# ログを確認
docker-compose logs -f

# コンテナを停止
docker-compose down
```

#### ローカル環境

```bash
python main.py
```

### TensorBoard での学習状況確認

学習状況を可視化するために TensorBoard を使用できます：

#### 1. シミュレーション実行

```bash
# シミュレーションを実行（TensorBoardログが自動生成される）
python main.py
```

#### 2. TensorBoard 起動

```bash
# 自動的に最新のログディレクトリを使用してTensorBoardを起動
python start_tensorboard.py

# または、特定のログディレクトリを指定
python start_tensorboard.py --log-dir ./logs/experiment_001

# ポートを変更する場合
python start_tensorboard.py --port 6007

# リモートアクセスを許可する場合
python start_tensorboard.py --host 0.0.0.0
```

#### 3. ブラウザで確認

TensorBoard が起動したら、ブラウザで以下の URL にアクセス：

```
http://localhost:6006
```

#### 4. 確認できる情報

**Scalars（スカラー値）**

- `episode/total_reward`: エピソード総報酬
- `episode/exploration_rate`: 探査率
- `episode/swarm_count`: 群の数
- `episode/steps`: エピソードステップ数
- `episode/avg_reward_per_step`: ステップあたりの平均報酬

**Swarm 固有のメトリクス**

- `swarm/{swarm_id}/episode_reward`: 各群のエピソード報酬
- `swarm/{swarm_id}/step_reward`: 各群のステップ報酬
- `swarm/{swarm_id}/avg_th`: VFH-Fuzzy パラメータ th の平均
- `swarm/{swarm_id}/avg_k_e`: VFH-Fuzzy パラメータ k_e の平均
- `swarm/{swarm_id}/avg_k_c`: VFH-Fuzzy パラメータ k_c の平均
- `swarm/{swarm_id}/avg_value`: Actor-Critic の価値関数の平均

**学習メトリクス**

- `learning/swarm_{swarm_id}/th`: VFH-Fuzzy パラメータ th
- `learning/swarm_{swarm_id}/k_e`: VFH-Fuzzy パラメータ k_e
- `learning/swarm_{swarm_id}/k_c`: VFH-Fuzzy パラメータ k_c
- `learning/swarm_{swarm_id}/value`: Actor-Critic の価値関数
- `learning/swarm_{swarm_id}/valid_directions_count`: 有効方向の数
- `learning/system/branch_threshold`: 分岐閾値
- `learning/system/integration_threshold`: 統合閾値

**イベント**

- 分岐イベントの記録
- 統合イベントの記録

#### 5. Docker 環境での TensorBoard

```bash
# コンテナ内でTensorBoardを起動
docker-compose exec multi-agent-simulation python start_tensorboard.py --host 0.0.0.0

# ホストマシンからアクセス
# http://localhost:6006
```

### 設定のカスタマイズ

`params/` ディレクトリ内のパラメータファイルを編集して設定を変更できます：

- `params/simulation.py`: 全体的なシミュレーション設定
- `params/agent.py`: エージェント設定（エピソード数、ステップ数）
- `params/environment.py`: 環境設定
- `params/system_agent.py`: SystemAgent 設定
- `params/swarm_agent.py`: SwarmAgent 設定
- `params/robot.py`: ロボット設定
- `params/explore.py`: 探査設定
- `params/reward.py`: 報酬設定
- `params/learning.py`: 学習パラメータ

### 実験の実行

```python
from core.application import create_application, ExperimentConfig
from params.simulation import Param

# パラメータを設定
param = Param()
param.agent.episodeNum = 100  # 100エピソード
param.agent.maxStepsPerEpisode = 50  # 50ステップ/エピソード

# アプリケーションを作成
app = create_application(param)

# 実験設定
experiment_config = ExperimentConfig(
    name="multi_agent_experiment",
    num_episodes=100,
    save_models=True,
    save_logs=True,
    save_visualizations=True
)

# 実験を実行
app.run_experiment(experiment_config)
```

## 📊 出力

実行後、以下のファイルが生成されます：

### 自動生成されるファイル

- `logs/sim_YYYYMMDD_HHMMSS_xxxxxx/`: 実験セッションごとのディレクトリ
  - `gifs/episode_0000.gif`: エピソードごとの GIF アニメーション
  - `metrics/episode_0000.json`: エピソードごとのメトリクス
  - `models/checkpoint_0000.keras`: エピソードごとの学習済みモデル
  - `csvs/trajectory_episode_0000.csv`: エピソードごとの軌跡データ
  - `csvs/experiment_xxx.json`: 実験結果のサマリー
  - `tensorboard/`: TensorBoard 用のログファイル

### 可視化内容

- **環境マップ**: 200×200 マップ、障害物と探査済み領域の表示
- **ロボット位置**: リーダー（★）とフォロワー（●）の現在位置
- **軌跡**: リーダーごとに色分けされた移動軌跡
- **統合リーダー軌跡**: 統合されたリーダーの軌跡（太線で表示）
- **探査領域**: リーダーを中心とした探査範囲の円
- **衝突点**: フォロワーの衝突位置（× マーク）
- **凡例**: 探査率と群数の表示（マップ外側）

## 🔧 アーキテクチャ

### 設計原則

1. **階層的設計**: SystemAgent と SwarmAgent の分離
2. **モジュラー設計**: 各コンポーネントが独立して動作
3. **インターフェース分離**: 共通インターフェースによる一貫性
4. **ファクトリパターン**: 動的なオブジェクト生成
5. **設定管理**: 一元化された設定システム
6. **学習統合**: 分岐・統合時の知識移転

### 主要コンポーネント

- **SystemAgent**: 高レベル制御（分岐・統合判断、学習閾値の最適化）
- **SwarmAgent**: 低レベル行動（VFH-Fuzzy 移動、衝突回避学習）
- **Environment**: シミュレーション環境（200×200 マップ、群管理）
- **Algorithms**: 行動決定アルゴリズム（VFH-Fuzzy、分岐・統合）
- **Robots**: 個別ロボット実装（リーダー・フォロワー機能）
- **Models**: ニューラルネットワーク（Actor-Critic）

### 群管理システム

```python
@dataclass
class Swarm:
    swarm_id: int
    leader: Red
    followers: List[Red]
    exploration_rate: float = 0.0
    step_count: int = 0
    learning_params: LearningParameter = None
```

- **群の作成**: 新しいリーダーとフォロワーで群を形成
- **群の統合**: 探査領域が重複する群の自動統合
- **群の分岐**: 学習可能な閾値による分岐判断
- **学習継承**: 分岐時の学習情報の引き継ぎ
- **学習統合**: 統合時の学習情報の重み付き平均

詳細は `docs/architecture.md` を参照してください。

## 🧪 実験

### 実験設定

```python
# 実験設定例
experiment_config = ExperimentConfig(
    name="multi_agent_exploration_test",
    description="階層的エージェント群ロボット探査実験",
    parameters={
        "algorithm": "vfh_fuzzy",
        "learning_enabled": True,
        "robot_count": 10,
        "map_size": "200x200",
        "episodes": 100,
        "steps_per_episode": 50
    },
    num_episodes=100,
    max_steps_per_episode=50,
    save_models=True,
    save_logs=True,
    save_visualizations=True
)
```

### メトリクス

- **探査率**: 探査済み領域の割合
- **衝突回数**: ロボットの衝突回数
- **移動距離**: 総移動距離
- **群分岐回数**: 群が分岐した回数
- **群統合回数**: 群が統合した回数
- **学習パラメータ**: th、k_e、k_c の変化
- **システム報酬**: SystemAgent の報酬推移
- **SwarmAgent 報酬**: 各 SwarmAgent の報酬推移

### 学習パラメータ

- **学習アルゴリズム**: A2C (Advantage Actor-Critic)
- **SystemAgent 報酬**: 探査効率、群数バランス、移動性スコア
- **SwarmAgent 報酬**: 探査向上、衝突回避、移動距離
- **状態空間**: 位置、方位、衝突フラグ、フォロワー情報
- **行動空間**: 移動方向（theta）、群制御モード

## 🔍 技術詳細

### 階層的エージェントシステム

- **SystemAgent**: 高レベル制御、分岐・統合判断
- **SwarmAgent**: 低レベル行動、VFH-Fuzzy 移動
- **協調学習**: 両エージェントの協調的な学習

### 分岐・統合システム

- **分岐アルゴリズム**: MobilityBasedBranchAlgorithm、RandomBranchAlgorithm
- **統合アルゴリズム**: NearestIntegrationAlgorithm
- **探査領域重複チェック**: 遠距離統合の防止
- **学習情報の継承・統合**: 知識の効率的な移転

### 学習システム

- **TensorFlow**: ニューラルネットワーク実装
- **Actor-Critic**: 行動価値と状態価値の同時学習
- **VFH-Fuzzy 最適化**: パラメータの動的調整
- **経験バッファ**: 経験の蓄積と再利用

### 可視化システム

- **matplotlib**: リアルタイム描画と GIF 生成
- **PIL**: 画像処理とフレーム合成
- **imageio**: GIF ファイルの保存
- **統合軌跡表示**: 統合されたリーダーの軌跡保持

## 📚 ドキュメント

- [アーキテクチャ設計](docs/architecture.md)
- [分岐アルゴリズム](docs/branch_algorithm.md)
- [統合アルゴリズム](docs/integration_algorithm.md)
- [学習システム](docs/learning_system.md)
- [報酬システム](docs/reward_system.md)
- [VFH-Fuzzy アルゴリズム](docs/vfh_fuzzy.md)
- [群最適化](docs/swarm_optimization.md)

## 🐛 トラブルシューティング

### よくある問題

1. **GIF が作成されない**

   - matplotlib のバージョン確認
   - メモリ不足の可能性

2. **学習が収束しない**

   - 報酬設計の見直し
   - 学習率の調整
   - VFH-Fuzzy パラメータの初期値確認

3. **群分岐・統合が頻繁に発生**

   - 閾値パラメータの調整
   - 最小群サイズの確認
   - クールダウン期間の延長

4. **リーダーが停止する**
   - フロンティアベース探査の特性（期待される動作）
   - 学習による衝突回避の改善を待つ
   - フォロワーの探査による脱出

## 🤝 貢献

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 🐳 Docker

### コンテナ構成

- **ベースイメージ**: Python 3.12.10-slim
- **主要パッケージ**: TensorFlow 2.15.0, Gymnasium 0.29.1, NumPy 1.26.3
- **可視化**: matplotlib 3.8.3, imageio 2.34.0
- **データ処理**: pandas 2.1.1, scipy 1.12.0

### ボリュームマウント

- `./logs:/app/logs`: ログファイルの永続化
- `./gifs:/app/gifs`: GIF ファイルの永続化
- `./scores:/app/scores`: スコアファイルの永続化

### 環境変数

- `PYTHONUNBUFFERED=1`: Python 出力のバッファリング無効化
- `PYTHONDONTWRITEBYTECODE=1`: .pyc ファイルの生成無効化

### 開発用コマンド

```bash
# コンテナ内でシェルを起動
docker-compose exec multi-agent-simulation bash

# ログをリアルタイムで確認
docker-compose logs -f multi-agent-simulation

# コンテナを再ビルド
docker-compose build --no-cache

# 特定のエピソード数のみ実行
docker-compose run --rm multi-agent-simulation python main.py --episodes 10
```

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は`LICENSE`ファイルを参照してください。

## 🙏 謝辞

- 群ロボット研究コミュニティ
- オープンソースライブラリの開発者
- プロジェクトに貢献したすべての方々

## 📁 出力ファイル構成

### 1. **GIF ファイル** (`logs/gifs/`)

- **ファイル名**: `episode_XXXX.gif`
- **内容**: 各エピソードの探査過程のアニメーション
- **形式**: GIF アニメーション（0.2 秒間隔）
- **特徴**: 200×200 マップ、統合リーダー軌跡表示

### 2. **Metrics ファイル** (`logs/metrics/`)

- **ファイル名**: `episode_XXXX.json`
- **内容**: 各エピソードの詳細なメトリクス
- **データ**: 探査率、報酬、衝突回数、群情報、学習パラメータ

### 3. **モデルファイル** (`logs/models/`)

- **ファイル名**: `episode_XXXX.h5`
- **内容**: 学習済みのニューラルネットワークモデル
- **形式**: TensorFlow/Keras モデル

### 4. **初期状態ファイル** (`logs/initial_state.json`)

- **ファイル名**: `initial_state.json`
- **内容**: シミュレーション開始時の全設定値と初期状態
- **データ**:
  - シミュレーション設定（200×200 マップ、障害物設定など）
  - エージェント設定（SystemAgent、SwarmAgent 設定）
  - 学習設定（100 エピソード、50 ステップ/エピソード）
  - 分岐・統合設定（アルゴリズム、閾値など）
  - 各ロボットの初期情報（位置、役割、群 ID など）
  - 報酬設定
  - 状態空間の情報

### 5. **TensorBoard ログ** (`logs/tensorboard/`)

- **内容**: 学習過程の可視化データ
- **形式**: TensorBoard 互換ログ

### 6. **軌跡データファイル** (`logs/csvs/trajectory_episode_XXXX.csv`)

- **ファイル名**: `trajectory_episode_XXXX.csv`
- **内容**: 各エピソードの詳細な軌跡データ
- **データ**:
  - `episode`: エピソード番号
  - `global_step`: エピソード内のグローバルステップ番号
  - `step`: 各 trajectory 内のステップ番号
  - `s_*`: 状態変数（座標、衝突フラグ、ステップ数など）
  - `a_*`: 行動変数（移動方向、群制御モードなど）
  - `reward`: 各ステップの報酬
  - `done`: エピソード終了フラグ

### 7. **実験結果** (`logs/experiment_results.json`)

- **内容**: 実験全体のサマリー
- **データ**: 全エピソードの結果、統計情報、学習パラメータの推移
