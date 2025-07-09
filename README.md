# Red Group Behavior - Swarm Robot Exploration

群ロボットによる未知環境探査シミュレーションシステム。リーダー・フォロワー動的切り替え、非同期動作、学習による群分岐・統合機能を実装。

## 🚀 特徴

- **動的リーダー・フォロワーシステム**: ロボットの役割を動的に切り替え
- **非同期群動作**: フォロワーが独立して動作
- **学習による群制御**: 動きやすさ指標に基づく分岐・統合
- **リアルタイム可視化**: 探査過程の可視化と GIF 保存
- **モジュラーアーキテクチャ**: 保守性と拡張性を重視した設計
- **エピソードごとの自動保存**: GIF、メトリクス、モデルの自動保存

## 🎯 シミュレーション概要

### 群ロボット探査システム

- **目的**: 未知環境での効率的な探査
- **ロボット数**: 設定可能（デフォルト: 10 台）
- **環境**: 障害物を含む 2 次元マップ
- **探査目標**: 指定された探査率の達成

### 主要機能

#### 1. 動的リーダー・フォロワーシステム

- **リーダー切り替え**: 定期的なリーダーの交代
- **役割動的変更**: リーダー ⇔ フォロワーの動的切り替え
- **独立した状態空間**: 各リーダーが独自の状態・アルゴリズム・エージェントを持つ

#### 2. 群分岐・統合システム

- **学習可能な閾値**: 分岐・統合の判断を学習で最適化
- **動きやすさ指標**: フォロワーの移動性を状態空間に含める
- **最小群サイズ制限**: 3 台以下の群は分岐しない
- **近接群統合**: 最も近い群への自動統合

#### 3. 非同期動作システム

- **並列処理**: フォロワーの動作を ThreadPoolExecutor で並列実行
- **独立した探査**: 各フォロワーが所属群のリーダーを参照
- **リアルタイム更新**: 探査マップのリアルタイム更新

#### 4. 学習システム

- **A2C (Advantage Actor-Critic)**: 強化学習による行動最適化
- **独立したエージェント**: 各リーダーが独自の学習エージェント
- **報酬設計**: 探査率向上、衝突回避、移動距離を考慮

#### 5. 可視化システム

- **リアルタイム描画**: 探査過程のリアルタイム表示
- **GIF 自動保存**: エピソードごとの GIF 自動生成
- **軌跡表示**: リーダーごとの色分けされた軌跡
- **探査領域表示**: リーダーを中心とした探査範囲の可視化

## 🔬 アルゴリズムと数式

### 1. VFH Fuzzy アルゴリズム

#### 基本概念

VFH (Vector Field Histogram) Fuzzy は、障害物回避と目標指向行動を組み合わせたナビゲーションアルゴリズムです。

#### 数式

**1. 極座標ヒストグラムの構築**

```
H(k) = ∑[i=1 to n] μ(di) * w(αi - k)
```

- `H(k)`: セクター k のヒストグラム値
- `μ(di)`: 距離 i のメンバーシップ関数
- `w(αi - k)`: 角度重み関数
- `di`: 障害物までの距離
- `αi`: 障害物の角度

**2. ファジィ推論による方向決定**

```
μ(θ) = min(μ_goal(θ), μ_obstacle(θ))
```

- `μ(θ)`: 方向 θ の総合メンバーシップ度
- `μ_goal(θ)`: 目標方向へのメンバーシップ度
- `μ_obstacle(θ)`: 障害物回避のメンバーシップ度

**3. 最終方向の決定**

```
θ_final = argmax[θ] μ(θ)
```

### 2. A2C (Advantage Actor-Critic) 学習

#### 基本概念

Actor-Critic は、行動選択（Actor）と価値評価（Critic）を同時に学習する強化学習手法です。

#### 数式

**1. Advantage 関数**

```
A(s, a) = Q(s, a) - V(s)
```

- `A(s, a)`: Advantage 値
- `Q(s, a)`: 行動価値関数
- `V(s)`: 状態価値関数

**2. Actor の損失関数**

```
L_actor = -log(π(a|s)) * A(s, a)
```

- `π(a|s)`: ポリシー関数
- `A(s, a)`: Advantage 値

**3. Critic の損失関数**

```
L_critic = (R + γV(s') - V(s))²
```

- `R`: 報酬
- `γ`: 割引率
- `V(s')`: 次の状態の価値

**4. 総損失関数**

```
L_total = L_actor + β * L_critic
```

- `β`: Critic 損失の重み係数

### 3. 群分岐・統合アルゴリズム

#### 動きやすさ指標 (Mobility Score)

```
M_i = w1 * D_i + w2 * C_i + w3 * E_i
```

- `M_i`: ロボット i の動きやすさ指標
- `D_i`: 移動距離（正規化済み）
- `C_i`: 衝突回避能力（1 - 衝突回数/総ステップ数）
- `E_i`: 探査効率（新規探査領域/移動距離）
- `w1, w2, w3`: 重み係数

#### 分岐条件

```
if (群サイズ > 3) and (平均動きやすさ > θ_branch):
    分岐実行
```

- `θ_branch`: 分岐閾値（学習可能）

#### 統合条件

```
if (群間距離 < θ_merge) and (統合後の効率 > 現在の効率):
    統合実行
```

- `θ_merge`: 統合距離閾値

### 4. 探査効率計算

#### 探査率

```
探査率 = (探査済みセル数) / (探査可能セル数)
```

#### 新規探査領域の計算

```
新規探査 = 現在の探査率 - 前ステップの探査率
```

#### 探査報酬

```
R_exploration = α * 新規探査 + β * 探査率
```

- `α`: 新規探査の重み
- `β`: 総探査率の重み

### 5. 非同期処理の仕組み

#### スレッドプール実行

```python
# フォロワーの並列実行
futures = []
for swarm in swarms:
    for follower in swarm.followers:
        future = executor.submit(execute_follower_step, swarm, follower)
        futures.append(future)

# 完了待機
for future in as_completed(futures):
    result = future.result()
    update_environment(result)
```

#### 状態更新の同期

```python
# スレッドセーフな状態更新
with lock:
    self.explored_map.update(result['explored_cells'])
    self.follower_positions.update(result['positions'])
```

### 6. 衝突検出アルゴリズム

#### 線分補間による衝突検出

```python
def interpolate_line(p1, p2):
    """Bresenham アルゴリズムによる線分補間"""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    points = []
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break

        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points
```

### 7. 報酬設計の詳細

#### 総合報酬関数

```
R_total = R_exploration + R_collision + R_movement + R_swarm
```

**各報酬成分:**

1. **探査報酬**

   ```
   R_exploration = w1 * Δ探査率 + w2 * 探査率
   ```

2. **衝突ペナルティ**

   ```
   R_collision = -w3 * 衝突回数
   ```

3. **移動報酬**

   ```
   R_movement = w4 * 移動距離 * (1 - 衝突フラグ)
   ```

4. **群制御報酬**
   ```
   R_swarm = w5 * 群効率 + w6 * 分岐・統合成功
   ```

### 8. 状態空間の設計

#### 基本状態ベクトル

```
S = [x, y, θ, collision_flag, step_count,
     follower_mobility_scores[10],
     follower_collision_data[100]]
```

- `x, y`: リーダーの位置座標
- `θ`: リーダーの方位角
- `collision_flag`: 衝突フラグ（0/1）
- `step_count`: ステップ数
- `follower_mobility_scores`: フォロワーの動きやすさ指標（10 次元）
- `follower_collision_data`: フォロワーの衝突データ（100×2 次元）

#### 状態正規化

```
S_normalized = (S - S_min) / (S_max - S_min)
```

### 9. 行動空間の設計

#### 行動ベクトル

```
A = [dx, dy, mode]
```

- `dx, dy`: 移動方向ベクトル（-1 ～ 1）
- `mode`: 群制御モード（0: 通常, 1: 分岐, 2: 統合）

#### 行動の実行

```python
def execute_action(action):
    # 移動方向の計算
    dx, dy = action[0], action[1]
    new_x = current_x + dx * movement_speed
    new_y = current_y + dy * movement_speed

    # 衝突検出
    if not is_collision(new_x, new_y):
        update_position(new_x, new_y)

    # 群制御モードの実行
    mode = action[2]
    execute_swarm_mode(mode)
```

### 10. 学習の収束条件

#### 収束判定

```
収束 = (平均報酬 > 閾値) and (報酬分散 < 閾値) and (エピソード数 > 最小エピソード数)
```

#### 学習率スケジューリング

```
learning_rate = initial_lr * (1 - episode / max_episodes) ^ decay_rate
```

## 📁 プロジェクト構造

```
red-group-behavior-async/
├── core/                    # コアモジュール
│   ├── __init__.py
│   ├── application.py      # メインアプリケーション
│   ├── config.py          # 設定管理
│   ├── interfaces.py      # 共通インターフェース
│   ├── factories.py       # ファクトリパターン
│   └── logging.py         # ログ管理
├── agents/                 # エージェント
│   ├── base_agent.py      # 基底エージェントクラス
│   ├── agent_a2c.py       # A2Cエージェント
│   ├── agent_config.py    # エージェント設定
│   └── agent_factory.py   # エージェントファクトリ
├── algorithms/            # アルゴリズム
│   ├── base_algorithm.py  # 基底アルゴリズムクラス
│   ├── vfh_fuzzy.py       # VFH Fuzzyアルゴリズム
│   └── algorithm_factory.py # アルゴリズムファクトリ
├── envs/                  # 環境
│   ├── env.py            # メイン環境クラス
│   ├── env_map.py        # マップ生成
│   ├── action_space.py   # 行動空間
│   ├── observation_space.py # 観測空間
│   └── reward.py         # 報酬関数
├── models/               # ニューラルネットワーク
│   ├── model.py          # モデル基底クラス
│   └── actor_critic.py   # Actor-Criticモデル
├── robots/               # ロボット
│   └── red.py           # ロボットクラス
├── params/               # パラメータ
│   ├── simulation.py    # シミュレーションパラメータ
│   ├── agent.py         # エージェントパラメータ
│   ├── environment.py   # 環境パラメータ
│   ├── robot.py         # ロボットパラメータ
│   └── explore.py       # 探査パラメータ
├── utils/               # ユーティリティ
│   ├── utils.py         # 共通ユーティリティ
│   └── metrics.py       # メトリクス
├── scores/              # スコア計算
│   └── score.py         # スコアクラス
├── docs/                # ドキュメント
│   ├── architecture.md  # アーキテクチャ設計
│   ├── reward_system.md # 報酬システム
│   └── vfh_fuzzy.md     # VFH Fuzzyアルゴリズム
├── main.py              # メインエントリーポイント
├── requirements.txt     # 依存関係
└── README.md           # このファイル
```

## 🛠️ セットアップ

### 必要条件

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- PIL (Pillow)
- imageio

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd red-group-behavior-async

# 依存関係をインストール
pip install -r requirements.txt
```

## 🚀 使用方法

### 基本的な実行

```bash
python main.py
```

### 設定のカスタマイズ

`params/` ディレクトリ内のパラメータファイルを編集して設定を変更できます：

- `params/simulation.py`: 全体的なシミュレーション設定
- `params/agent.py`: エージェント設定
- `params/environment.py`: 環境設定
- `params/robot.py`: ロボット設定
- `params/explore.py`: 探査設定

### 実験の実行

```python
from core.application import create_application, ExperimentConfig
from params.simulation import Param

# パラメータを設定
param = Param()
param.agent.isLearning = True
param.agent.learningParameter.episodeNum = 100

# アプリケーションを作成
app = create_application(param)

# 実験設定
experiment_config = ExperimentConfig(
    name="my_experiment",
    num_episodes=100,
    save_models=True,
    save_logs=True,
    save_visualizations=True  # GIF保存を有効化
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
  - `csvs/experiment_xxx.json`: 実験結果のサマリー
  - `tensorboard/`: TensorBoard 用のログファイル

### 可視化内容

- **環境マップ**: 障害物と探査済み領域の表示
- **ロボット位置**: リーダー（★）とフォロワー（●）の現在位置
- **軌跡**: リーダーごとに色分けされた移動軌跡
- **探査領域**: リーダーを中心とした探査範囲の円
- **衝突点**: フォロワーの衝突位置（× マーク）

## 🔧 アーキテクチャ

### 設計原則

1. **モジュラー設計**: 各コンポーネントが独立して動作
2. **インターフェース分離**: 共通インターフェースによる一貫性
3. **ファクトリパターン**: 動的なオブジェクト生成
4. **設定管理**: 一元化された設定システム
5. **ログ管理**: 統一されたログシステム

### 主要コンポーネント

- **Application**: メインアプリケーション管理
- **Environment**: シミュレーション環境（群管理、非同期処理）
- **Agents**: 学習エージェント（A2C）
- **Algorithms**: 行動決定アルゴリズム（VFH Fuzzy）
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
```

- **群の作成**: 新しいリーダーとフォロワーで群を形成
- **群の統合**: 近接する群の自動統合
- **群の分岐**: 学習可能な閾値による分岐判断

詳細は `docs/architecture.md` を参照してください。

## 🧪 実験

### 実験設定

```python
# 実験設定例
experiment_config = ExperimentConfig(
    name="swarm_exploration_test",
    description="群ロボット探査実験",
    parameters={
        "algorithm": "vfh_fuzzy",
        "learning_enabled": True,
        "robot_count": 10,
        "map_size": "100x100"
    },
    num_episodes=50,
    max_steps_per_episode=1000,
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
- **リーダー交代回数**: リーダーが交代した回数

### 学習パラメータ

- **学習アルゴリズム**: A2C (Advantage Actor-Critic)
- **報酬設計**: 探査率向上、衝突回避、移動距離
- **状態空間**: 位置、方位、衝突フラグ、フォロワー情報
- **行動空間**: 移動方向、群制御モード

## 🔍 技術詳細

### 非同期処理

- **ThreadPoolExecutor**: フォロワーの並列実行
- **as_completed**: 非同期タスクの完了待機
- **スレッドセーフ**: 環境状態の安全な更新

### 可視化システム

- **matplotlib**: リアルタイム描画と GIF 生成
- **PIL**: 画像処理とフレーム合成
- **imageio**: GIF ファイルの保存

### 学習システム

- **TensorFlow**: ニューラルネットワーク実装
- **Actor-Critic**: 行動価値と状態価値の同時学習
- **Experience Replay**: 経験の蓄積と再利用

## 📚 ドキュメント

- [アーキテクチャ設計](docs/architecture.md)
- [報酬システム](docs/reward_system.md)
- [VFH Fuzzy アルゴリズム](docs/vfh_fuzzy.md)
- [群最適化](docs/swarm_optimization.md)

## 🐛 トラブルシューティング

### よくある問題

1. **GIF が作成されない**

   - matplotlib のバージョン確認
   - メモリ不足の可能性

2. **学習が収束しない**

   - 報酬設計の見直し
   - 学習率の調整

3. **群分岐・統合が頻繁に発生**
   - 閾値パラメータの調整
   - 最小群サイズの確認

## 🤝 貢献

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は `LICENSE` ファイルを参照してください。

## 🙏 謝辞

- 群ロボット研究コミュニティ
- オープンソースライブラリの開発者
- プロジェクトに貢献したすべての方々

## 📁 出力ファイル構成

### 1. **GIF ファイル** (`logs/gifs/`)

- **ファイル名**: `episode_XXXX.gif`
- **内容**: 各エピソードの探査過程のアニメーション
- **形式**: GIF アニメーション（0.1 秒間隔）

### 2. **Metrics ファイル** (`logs/metrics/`)

- **ファイル名**: `episode_XXXX.json`
- **内容**: 各エピソードの詳細なメトリクス
- **データ**: 探査率、報酬、衝突回数、群情報など

### 3. **モデルファイル** (`logs/models/`)

- **ファイル名**: `episode_XXXX.h5`
- **内容**: 学習済みのニューラルネットワークモデル
- **形式**: TensorFlow/Keras モデル

### 4. **初期状態ファイル** (`logs/initial_state.json`)

- **ファイル名**: `initial_state.json`
- **内容**: シミュレーション開始時の全設定値と初期状態
- **データ**:
  - シミュレーション設定（マップサイズ、障害物設定など）
  - ロボット設定（移動パラメータ、境界設定など）
  - 探査設定（境界、初期座標など）
  - 群設定（リーダー切り替え間隔、群数など）
  - 各ロボットの初期情報（位置、役割、群 ID など）
  - 報酬設定
  - 状態空間の情報

### 5. **TensorBoard ログ** (`logs/tensorboard/`)

- **内容**: 学習過程の可視化データ
- **形式**: TensorBoard 互換ログ

### 6. **実験結果** (`logs/experiment_results.json`)

- **内容**: 実験全体のサマリー
- **データ**: 全エピソードの結果、統計情報
