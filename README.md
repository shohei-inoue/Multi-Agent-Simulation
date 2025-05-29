# red-group-behavior

群ロボットシステム「RED」のデモンストレーションに向けた群誘導分岐アルゴリズムの開発・評価を行うリポジトリです。

本プロジェクトでは、リーダー・フォロワー構成の群探索を想定し、以下の特徴を備えた自律分散型アルゴリズムを構築しています：

- 探索環境は障害物を含む 2 次元グリッドマップ
- フォロワーロボットは RED モデルに基づく確率密度制御で動作
- リーダーロボットは VFH + ファジィ推論によって最適方向を決定
- 強化学習（A2C）によりパラメータ `th`, `k_e`, `k_c` をオンライン最適化

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

├── agents/
│   └── a2c.py                     # A2C エージェント定義

├── envs/
│   ├── env.py                     # 探索環境 (gym.Env)
│   ├── env_parameter.py           # 環境パラメータ定義
│   ├── action_space.py            # アクション空間生成
│   └── observation_space.py       # 状態空間生成

├── models/
│   └── actor_critic.py            # ParamActorCritic モデル

├── utils/
│   └── utils.py                   # 状態変換などの補助関数

└── robots/
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

初回実行時、エージェントは探索環境を 10 エピソード分学習します。
結果は results/gif/ 以下に GIF として自動保存されます。

## 環境構築（ローカルで実行する場合）

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
