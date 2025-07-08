# red-group-behavior

群ロボットシステム「RED」のデモンストレーションに向けた群誘導分岐アルゴリズムの開発・評価を行うリポジトリです。

本プロジェクトでは、リーダー・フォロワー構成の群探索を想定し、以下の特徴を備えた自律分散型アルゴリズムを構築しています：

- 探索環境は障害物を含む 2 次元グリッドマップ
- フォロワーロボットは RED モデルに基づく確率密度制御で動作

---

## 📦 依存ライブラリ（主要）

- tensorflow
- gym
- matplotlib, Pillow, imageio
- numpy, pandas, scipy

## 📁 ディレクトリ構成

```bash
red-group-behavior/
├── main.py                        # 実行エントリーポイント
├── Dockerfile                     # Docker ビルド定義
├── docker-compose.yml             # 開発用コンテナ構成
├── requirements.txt               # Python 依存パッケージ一覧
├── algorithm/                     # アルゴリズム関連
├── agents/                        # エージェント関連
│   └── a2c.py                     # A2C エージェント定義
├── envs/                          # 環境関連
│   ├── env.py                     # 探索環境 (gym.Env)
│   ├── env_parameter.py           # 環境パラメータ定義
│   ├── action_space.py            # アクション空間生成
│   └── observation_space.py       # 状態空間生成
├── models/                        # モデル関連
│   └── actor_critic.py            # ParamActorCritic モデル
├── params/                        # パラメータ関連
├── utils/                         # ユーティリティ処理等
│   └── utils.py                   # 状態変換などの補助関数
├── scores/                        # スコア関連
└── robots/                        # ロボット関連
    ├── red.py                     # REDクラス（フォロワー挙動）
    └── red_parameter.py           # REDのパラメータ定義
```

## 🚀 実行方法

### Docker ビルド

```bash
docker-compose build
```

### トレーニング実行

```bash
docker-compose up --build
```

## 環境構築（ローカルで実行する場合）

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## TODO

- 各種 score の作成, 取得
- 動作 csv 作成, 取得
- csv を用いた gif 作成関数の作成

## 機能

### スコアログ機能

シミュレーション実行時に以下のスコアデータが自動的に保存されます：

#### 1. エピソードごとの詳細データ (CSV)

- 保存場所: `logs/{simulation_id}/csvs/episode_{episode:04d}_exploration.csv`
- 内容:
  - `step`: ステップ番号
  - `exploration_rate`: 探査率
  - `explored_area`: 探査済みエリア数
  - `total_area`: 総エリア数
  - `agent_collision_flag`: エージェント衝突フラグ
  - `follower_collision_count`: フォロワ衝突回数
  - `reward`: 報酬

#### 2. エピソード全体のサマリー (JSON)

- 保存場所: `logs/{simulation_id}/metrics/episode_{episode:04d}_summary.json`
- 内容:
  - `episode`: エピソード番号
  - `total_reward`: 総報酬
  - `final_exploration_rate`: 最終探査率
  - `max_exploration_rate`: 最大探査率
  - `total_steps`: 総ステップ数
  - `goal_reaching_step`: 目標到達ステップ
  - `agent_collision_count`: エージェント衝突回数
  - `follower_collision_count`: フォロワ衝突回数
  - `total_distance_traveled`: 総走行距離
  - `exploration_rate_curve`: 探査率履歴

#### 3. 全エピソードのサマリー (CSV)

- 保存場所: `logs/{simulation_id}/csvs/all_episodes_summary.csv`
- 内容: 全エピソードのサマリーデータ

#### 4. 最終スコア (JSON)

- 保存場所: `logs/{simulation_id}/metrics/score.json`
- 内容: シミュレーション全体の最終スコア

## 使用方法

```bash
python main.py
```

実行後、`logs/`ディレクトリ以下に以下の構造でファイルが保存されます：

```
logs/
└── {simulation_id}/
    ├── csvs/
    │   ├── episode_0000_exploration.csv
    │   ├── episode_0001_exploration.csv
    │   ├── ...
    │   └── all_episodes_summary.csv
    ├── metrics/
    │   ├── episode_0000_summary.json
    │   ├── episode_0001_summary.json
    │   ├── ...
    │   └── score.json
    ├── gifs/
    ├── models/
    └── tensorboard/
```

## データ分析例

### 探査率の推移を確認

```python
import pandas as pd
import matplotlib.pyplot as plt

# 特定エピソードの探査率推移
df = pd.read_csv('logs/sim_20241201_120000_abc123/csvs/episode_0000_exploration.csv')
plt.plot(df['step'], df['exploration_rate'])
plt.xlabel('Step')
plt.ylabel('Exploration Rate')
plt.title('Exploration Rate Progress')
plt.show()

# 全エピソードの最終探査率比較
df_all = pd.read_csv('logs/sim_20241201_120000_abc123/csvs/all_episodes_summary.csv')
plt.plot(df_all['episode'], df_all['final_exploration_rate'])
plt.xlabel('Episode')
plt.ylabel('Final Exploration Rate')
plt.title('Final Exploration Rate by Episode')
plt.show()
```

## 設定

パラメータは `params/` ディレクトリ内のファイルで設定できます：

- `params/simulation.py`: シミュレーション全体の設定
- `params/environment.py`: 環境設定
- `params/agent.py`: エージェント設定
- `params/robot.py`: ロボット設定
- `params/explore.py`: 探査設定
- `params/robot_logging.py`: ロボットデータ保存設定

### ロボットデータ保存設定

ロボットのステップデータは膨大になるため、効率的な保存方法を提供しています：

#### 1. 基本設定

```python
# params/robot_logging.py で設定
save_robot_data: bool = False  # ロボットデータを保存するか
save_episode_summary: bool = True  # エピソードサマリーは常に保存
```

#### 2. サンプリング設定

```python
sampling_rate: float = 0.1  # 保存するステップの割合（0.1 = 10%）
save_collision_only: bool = True  # 衝突時のみ保存
```

#### 3. 保存するデータの種類

```python
save_position: bool = True  # 位置情報
save_collision: bool = True  # 衝突情報
save_boids: bool = True  # boids情報
save_distance: bool = True  # エージェントとの距離
```

#### 4. 使用例

```python
# 衝突時のみ保存（推奨）
robot_config = RobotLoggingConfig(
    save_robot_data=True,
    save_collision_only=True,
    sampling_rate=0.0  # サンプリング無効
)

# 10%サンプリング + 衝突時保存
robot_config = RobotLoggingConfig(
    save_robot_data=True,
    sampling_rate=0.1,
    save_collision_only=True
)

# 位置情報のみ保存
robot_config = RobotLoggingConfig(
    save_robot_data=True,
    save_position=True,
    save_collision=False,
    save_boids=False,
    save_distance=False
)
```

#### 5. 保存されるファイル

- `episode_{episode:04d}_robots.csv`: ロボットのステップデータ
- 内容: step, robot_id, x, y, collision_flag, boids_flag, distance
