# マルチエージェント群分岐・統合システム

## 概要

群ロボットシミュレーションにおいて、学習により行動空間の mode が決定され、それに応じて群の構造が動的に変化するシステムを実装しました。

- **mode 0**: 通常動作（単一群）
- **mode 1**: 群分岐（新しい leader を作成し、群を分割）
- **mode 2**: 群統合（他の群に統合）

## システム構成

### 1. 群クラス（Swarm）

```python
@dataclass
class Swarm:
    swarm_id: int           # 群ID
    leader: Red            # 群のleader
    followers: List[Red]   # 群のfollower
    exploration_rate: float = 0.0
    step_count: int = 0
```

### 2. ロボットの群管理

```python
class Red:
    # 群管理属性
    self.swarm_id: int = 0    # 所属群ID
```

## 実装詳細

### 1. Mode 決定アルゴリズム

```python
def _determine_swarm_mode(self, state, sampled_params):
    """
    群分岐・統合のmodeを決定
    """
    exploration_rate = state.get("exploration_rate", 0.0)
    th, k_e, k_c = sampled_params

    # 探査率が高い場合（30%以上）で分岐を検討
    if exploration_rate > 0.3:
        branch_probability = min(0.8, k_e / 50.0)
        if np.random.random() < branch_probability:
            return 1  # 群分岐

    # 探査率が低い場合（10%以下）で統合を検討
    if exploration_rate < 0.1:
        merge_probability = min(0.6, k_c / 30.0)
        if np.random.random() < merge_probability:
            return 2  # 群統合

    return 0  # 通常動作
```

### 2. 群分岐機能

```python
def _handle_swarm_mode(self, mode: int):
    if mode == 1:  # 群分岐
        current_swarm = self._find_swarm_by_leader(self.current_leader)
        if current_swarm and len(current_swarm.followers) >= 2:
            # 新しいleaderを選択（最初のfollower）
            new_leader = current_swarm.followers[0]
            new_leader.set_role(RobotRole.LEADER)

            # 新しいfollowerを選択（残りのfollowerの半分）
            remaining_followers = current_swarm.followers[1:]
            split_point = len(remaining_followers) // 2
            new_followers = remaining_followers[:split_point]

            # 新しい群を作成
            self._create_new_swarm(new_leader, new_followers)
```

### 3. 群統合機能

```python
def _handle_swarm_mode(self, mode: int):
    elif mode == 2:  # 群統合
        current_swarm = self._find_swarm_by_leader(self.current_leader)
        if current_swarm and len(self.swarms) > 1:
            # 最も近い群を探す
            closest_swarm = self._find_closest_swarm(current_swarm)
            if closest_swarm and closest_swarm != current_swarm:
                self._merge_swarms(closest_swarm, current_swarm)
```

## 動作フロー

### 1. 初期状態

- 単一の群（Swarm 0）で開始
- 1 つの leader と複数の follower

### 2. 群分岐（mode 1）

```
Before: [Leader A] + [Follower B, C, D, E]
After:  [Leader A] + [Follower C, D]  (Swarm 0)
        [Leader B] + [Follower E]     (Swarm 1)
```

### 3. 群統合（mode 2）

```
Before: [Leader A] + [Follower C, D]  (Swarm 0)
        [Leader B] + [Follower E]     (Swarm 1)
After:  [Leader A] + [Follower C, D, B, E]  (Swarm 0)
```

## 可視化

### 群別色分け表示

- **Leader**: 各群の色で星マーカー（サイズ 20）
- **Follower**: 各群の色で円マーカー（サイズ 10）
- **軌跡**: 群ごとに色分けされた線

```python
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
for swarm_idx, swarm in enumerate(self.swarms):
    color = colors[swarm_idx % len(colors)]

    # leaderの描画（星マーカー）
    ax.scatter(x=leader.x, y=leader.y, color=color, s=20, marker='*')

    # followerの描画（円マーカー）
    for follower in swarm.followers:
        ax.scatter(x=follower.x, y=follower.y, color=color, s=10)
```

## 学習パラメータの影響

### 1. k_e（探査向上性）の影響

- **高い k_e**: 群分岐が起こりやすい
- **低い k_e**: 単一群で探査を継続

### 2. k_c（衝突回避）の影響

- **高い k_c**: 群統合が起こりやすい
- **低い k_c**: 群を維持しやすい

### 3. 探査率の影響

- **高探査率（>30%）**: 分岐の可能性が高まる
- **低探査率（<10%）**: 統合の可能性が高まる

## 利点

### 1. 適応的探査

- 環境に応じて群の構造が変化
- 効率的な探査領域の分割

### 2. 学習による制御

- 強化学習により最適な分岐・統合タイミングを学習
- パラメータに基づく動的制御

### 3. スケーラビリティ

- ロボット数の増加に対応
- 複数群による並列探査

### 4. 故障耐性

- 群の分割によるリスク分散
- 統合による冗長性確保

## 設定パラメータ

### 分岐・統合の閾値

```python
# 分岐判定
exploration_rate > 0.3  # 30%以上で分岐検討
branch_probability = min(0.8, k_e / 50.0)

# 統合判定
exploration_rate < 0.1  # 10%以下で統合検討
merge_probability = min(0.6, k_c / 30.0)
```

### 推奨設定

- **小規模群（3-5 台）**: 分岐閾値 0.4, 統合閾値 0.05
- **中規模群（6-10 台）**: 分岐閾値 0.3, 統合閾値 0.1
- **大規模群（10 台以上）**: 分岐閾値 0.25, 統合閾値 0.15

## 今後の拡張

### 1. 適応的閾値

- 探査履歴に基づく動的閾値調整
- 環境複雑度による自動調整

### 2. 複数分岐

- 同時に複数の群に分岐
- 階層的群構造

### 3. 通信機能

- 群間通信による協調
- 情報共有による効率化

### 4. 学習ベース制御

- 群分岐・統合の最適化
- 各群の専門化学習
