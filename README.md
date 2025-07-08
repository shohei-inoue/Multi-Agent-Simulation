# red-group-behavior

群ロボットシステム「RED」のデモンストレーションに向けた群誘導分岐アルゴリズムの開発・評価を行うリポジトリです。

本プロジェクトでは、リーダー・フォロワー構成の群探索を想定し、以下の特徴を備えた自律分散型アルゴリズムを構築しています：

- 探索環境は障害物を含む 2 次元グリッドマップ
- フォロワーロボットは RED モデルに基づく確率密度制御で動作
- VFH-Fuzzyアルゴリズムによる効率的な探査戦略
- A2C（Advantage Actor-Critic）による強化学習

---

## 📦 依存ライブラリ（主要）

- tensorflow==2.15.0
- keras==2.15.0
- gym==0.26.2
- numpy==1.26.3
- matplotlib==3.8.3
- pandas==2.1.1
- scipy==1.12.0
- pydantic==2.5.0
- Pillow==10.2.0
- imageio==2.34.0

## 📁 ディレクトリ構成

```bash
red-group-behavior/
├── main.py                        # 実行エントリーポイント
├── Dockerfile                     # Docker ビルド定義
├── docker-compose.yml             # 開発用コンテナ構成
├── requirements.txt               # Python 依存パッケージ一覧
├── algorithms/                    # アルゴリズム関連
│   ├── vfh_fuzzy.py              # VFH-Fuzzyアルゴリズム（改善版）
│   ├── algorithm_factory.py      # アルゴリズムファクトリ
│   └── base_algorithm.py         # アルゴリズム基底クラス
├── agents/                        # エージェント関連
│   ├── agent_a2c.py              # A2C エージェント定義
│   ├── agent_config.py           # エージェント設定
│   ├── agent_factory.py          # エージェントファクトリ
│   └── base_agent.py             # エージェント基底クラス
├── envs/                          # 環境関連
│   ├── env.py                     # 探索環境 (gym.Env)
│   ├── env_map.py                 # マップ生成
│   ├── action_space.py            # アクション空間生成
│   ├── observation_space.py       # 状態空間生成
│   └── reward.py                  # 報酬設計
├── models/                        # モデル関連
│   ├── actor_critic.py            # Actor-Critic モデル
│   └── model.py                   # モデルファクトリ
├── params/                        # パラメータ関連
│   ├── simulation.py              # シミュレーション設定
│   ├── environment.py             # 環境設定
│   ├── agent.py                   # エージェント設定
│   ├── robot.py                   # ロボット設定
│   ├── explore.py                 # 探査設定
│   └── robot_logging.py           # ロボットデータ保存設定
├── utils/                         # ユーティリティ処理等
│   ├── utils.py                   # 状態変換などの補助関数
│   ├── logger.py                  # ログ設定
│   └── metrics.py                 # メトリクス保存
├── scores/                        # スコア関連
│   └── score.py                   # スコア計算・保存
├── robots/                        # ロボット関連
│   └── red.py                     # REDクラス（フォロワー挙動）
└── docs/                          # ドキュメント
    └── vfh_fuzzy_improvements.md  # VFH-Fuzzy改善ドキュメント
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
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 main.py
```

## 🧠 アルゴリズム

### VFH-Fuzzyアルゴリズム（改善版）

群ロボットの全探索問題に特化したVFH-Fuzzyアルゴリズムを実装しています：

#### 主要な改善点

1. **パラメータベースの探査戦略**
   - 環境情報に依存しない汎用的な設計
   - k_eパラメータによる探査行動の動的調整
   - ランダム性と方向性の最適バランス

2. **群ロボットの分散促進**
   - フォロワーの衝突データを活用した分散戦略
   - 効率的な群ロボット間の協調

3. **ファジィ推論の最適化**
   - 探査向上性の重みを増加（1.5倍）
   - より積極的な探査行動の促進

4. **方向選択アルゴリズムの改善**
   - 四分位数ベースの方向選択
   - 探査効率を重視した重み付け

#### 推奨パラメータ範囲

- **th**: 0.5 - 2.0（走行可能性の閾値）
- **k_e**: 10.0 - 50.0（探査向上性の重み）
- **k_c**: 5.0 - 20.0（衝突回避の重み）

### A2C（Advantage Actor-Critic）

強化学習によるパラメータ最適化を実装：

- Actor-Criticモデルによる方策学習
- エントロピーボーナスによる探索促進
- 動的報酬設計による探査率向上

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

#### 5. GIFアニメーション

- 保存場所: `logs/{simulation_id}/gifs/episode_{episode:04d}.gif`
- 内容: エピソード中の探査行動の可視化

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
    │   ├── episode_0000.gif
    │   ├── episode_0001.gif
    │   └── ...
    ├── models/
    │   └── agent_model.keras
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
save_position: bool = True    # 位置情報
save_collision: bool = True   # 衝突情報
save_boids: bool = True       # Boids情報
save_distance: bool = True    # 距離情報
```

## 期待される効果

### 探査効率の向上
- より効率的な方向選択
- 重複探査の削減
- 探査率の大幅な向上

### 協調性の向上
- 群ロボット間の効率的な分散
- 衝突回避の強化
- 協調的な探査行動

### 汎用性の向上
- 環境情報に依存しない設計
- 様々な環境での適用可能性
- パラメータ調整による柔軟性

## 今後の改善方向

1. **適応的パラメータ調整**: 探査状況に応じた動的パラメータ調整
2. **マルチエージェント学習**: 群ロボット間の協調学習
3. **環境適応**: 異なる環境での自動適応機能
4. **リアルタイム最適化**: 実行時のパラメータ最適化
