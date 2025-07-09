# マルチエージェント群分岐・統合システム

## 概要

群ロボットシミュレーションにおいて、フォロワーの動きやすさ指標（Mobility Score）に基づいて群の構造が動的に変化するシステムを実装しました。学習可能な閾値により分岐・統合の判断を最適化し、効率的な探査を実現します。

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

    def get_all_robots(self) -> List[Red]:
        """群に属する全ロボットを取得"""
        return [self.leader] + self.followers

    def add_follower(self, robot: Red):
        """followerを追加"""
        self.followers.append(robot)
        robot.swarm_id = self.swarm_id

    def remove_follower(self, robot: Red):
        """followerを削除"""
        if robot in self.followers:
            self.followers.remove(robot)

    def get_robot_count(self) -> int:
        """群のロボット数を取得"""
        return 1 + len(self.followers)  # leader + followers
```

### 2. ロボットの群管理

```python
class Red:
    # 群管理属性
    self.swarm_id: int = 0    # 所属群ID
    self.mobility_score: float = 0.0  # 動きやすさ指標
```

## 実装詳細

### 1. 動きやすさ指標の計算

```python
def calculate_mobility_score(self):
    """
    フォロワーの動きやすさ指標を計算
    """
    # 障害物までの距離を計算
    obstacle_distance = self._calculate_obstacle_distance()

    # 他のロボットとの距離を計算
    robot_distance = self._calculate_robot_distance()

    # 探査済み領域の密度を計算
    exploration_density = self._calculate_exploration_density()

    # 動きやすさ指標を統合
    mobility_score = (
        0.4 * obstacle_distance +      # 障害物回避（40%）
        0.3 * robot_distance +         # ロボット間距離（30%）
        0.3 * exploration_density      # 探査密度（30%）
    )

    self.mobility_score = max(0.0, min(1.0, mobility_score))
    return self.mobility_score
```

### 2. Mode 決定アルゴリズム

```python
def _determine_swarm_mode(self, state, sampled_params):
    """
    群分岐・統合のmodeを決定（動きやすさ指標ベース）
    """
    # フォロワーの動きやすさ指標を取得
    follower_mobility_scores = state.get('follower_mobility_scores', [])

    if len(follower_mobility_scores) >= 3:
        # 上位3台のフォロワーの平均動きやすさを計算
        avg_mobility = np.mean(follower_mobility_scores[:3])

        # 分岐閾値（学習可能）
        branch_threshold = sampled_params[3] if len(sampled_params) > 3 else 0.5

        if avg_mobility > branch_threshold:
            return 1  # 分岐モード
        else:
            return 2  # 統合モード

    return 0  # 通常モード
```

### 3. 群分岐機能

```python
def _handle_swarm_mode(self, mode: int):
    """群分岐・統合の処理"""
    if mode == 0:  # 通常動作
        return

    elif mode == 1:  # 群分岐
        # 現在の群から新しい群を作成
        current_swarm = self._find_swarm_by_leader(self.current_leader)
        if current_swarm and len(current_swarm.followers) >= 3:  # followerが3台以上の場合のみ分岐
            # 新しいleaderを選択（最初のfollower）
            new_leader = current_swarm.followers[0]
            new_leader.set_role(RobotRole.LEADER)

            # 新しいfollowerを選択（残りのfollowerの半分）
            remaining_followers = current_swarm.followers[1:]
            split_point = len(remaining_followers) // 2
            new_followers = remaining_followers[:split_point]

            # 新しい群を作成
            self._create_new_swarm(new_leader, new_followers)

            # 元の群からfollowerを削除
            for follower in [new_leader] + new_followers:
                current_swarm.remove_follower(follower)
        else:
            print(f"Swarm {current_swarm.swarm_id if current_swarm else 'Unknown'} has insufficient followers ({len(current_swarm.followers) if current_swarm else 0}) for branching. Minimum required: 3")

    elif mode == 2:  # 群統合
        # 最も近い群に統合
        current_swarm = self._find_swarm_by_leader(self.current_leader)
        if current_swarm and len(self.swarms) > 1:
            # 最も近い群を探す
            closest_swarm = self._find_closest_swarm(current_swarm)
            if closest_swarm and closest_swarm != current_swarm:
                self._merge_swarms(closest_swarm, current_swarm)
```

### 4. 群統合機能

```python
def _merge_swarms(self, target_swarm: Swarm, source_swarm: Swarm):
    """
    群を統合
    """
    # source_swarmのfollowerをtarget_swarmに移動
    for follower in source_swarm.followers:
        target_swarm.add_follower(follower)

    # source_swarmのleaderもfollowerとして追加
    source_swarm.leader.set_role(RobotRole.FOLLOWER)
    target_swarm.add_follower(source_swarm.leader)

    # source_swarmを削除
    self.swarms.remove(source_swarm)
    print(f"Swarm {source_swarm.swarm_id} merged into swarm {target_swarm.swarm_id}")
```

## 動作フロー

### 1. 初期状態

- 単一の群（Swarm 0）で開始
- 1 つの leader と複数の follower
- 全ロボットの動きやすさ指標を計算

### 2. 群分岐（mode 1）

```
Before: [Leader A] + [Follower B, C, D, E] (Mobility: [0.8, 0.7, 0.6, 0.5])
        Avg Mobility: 0.65 > Branch Threshold (0.5)
After:  [Leader A] + [Follower C, D]  (Swarm 0)
        [Leader B] + [Follower E]     (Swarm 1)
```

### 3. 群統合（mode 2）

```
Before: [Leader A] + [Follower C, D]  (Swarm 0)
        [Leader B] + [Follower E]     (Swarm 1)
        Avg Mobility: 0.3 < Branch Threshold (0.5)
After:  [Leader A] + [Follower C, D, B, E]  (Swarm 0)
```

## 可視化

### 群別色分け表示

- **Leader**: 各群の色で星マーカー（サイズ 25）
- **Follower**: 各群の色で円マーカー（サイズ 10）
- **軌跡**: 群ごとに色分けされた線

```python
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']

for swarm_idx, swarm in enumerate(self.swarms):
    # リーダーIDに基づいてカラーを決定
    leader_id = int(swarm.leader.id) if swarm.leader.id.isdigit() else hash(swarm.leader.id) % len(colors)
    leader_color = colors[leader_id % len(colors)]
    trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]

    # leaderの描画（星マーカー）
    ax.scatter(
        x=leader.data['x'].iloc[-1],
        y=leader.data['y'].iloc[-1],
        color=leader_color,
        s=25,
        marker='*',
        edgecolors='black',
        linewidth=1,
        label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})" if swarm_idx == 0 else None
    )

    # followerの描画（円マーカー）
    for i, follower in enumerate(swarm.followers):
        if follower.role == RobotRole.FOLLOWER:
            fx = follower.data['x'].iloc[-1]
            fy = follower.data['y'].iloc[-1]
            ax.scatter(
                x=fx,
                y=fy,
                color=leader_color,  # リーダーと同じカラーを使用
                s=10,
                marker='o',
                alpha=0.7,
                label=f"Follower (Leader ID: {leader.id})" if i == 0 and swarm_idx == 0 else None
            )
```

## 学習パラメータの影響

### 1. 分岐閾値（branch_threshold）の影響

- **高い閾値（>0.7）**: 群分岐が起こりにくい
- **低い閾値（<0.3）**: 群分岐が起こりやすい
- **推奨範囲**: 0.3-0.7

### 2. 動きやすさ指標の構成要素

- **障害物距離（40%）**: 障害物が多い環境では分岐しにくい
- **ロボット間距離（30%）**: ロボットが密集していると分岐しやすい
- **探査密度（30%）**: 未探査領域が多いと分岐しやすい

### 3. 群サイズ制限の影響

- **最小群サイズ（3 台）**: 3 台以下の群は分岐しない
- **最大群サイズ**: 制限なし（環境に応じて動的調整）

## 利点

### 1. 適応的探査

- 動きやすさ指標に基づく客観的な判断
- 環境に応じて群の構造が変化
- 効率的な探査領域の分割

### 2. 学習による制御

- 強化学習により最適な分岐・統合タイミングを学習
- 動きやすさ指標の重みを動的調整
- 閾値の自動最適化

### 3. スケーラビリティ

- ロボット数の増加に対応
- 複数群による並列探査
- 動的群サイズ調整

### 4. 故障耐性

- 群の分割によるリスク分散
- 統合による冗長性確保
- 動きやすさに基づく適応的再編成

### 5. 効率性

- 動きやすさの高い環境での積極的分岐
- 動きにくい環境での統合による協調
- 探査効率の最大化

## 設定パラメータ

### 分岐・統合の閾値

```python
# 学習可能パラメータ
sampled_params = [
    th,              # 抑制閾値
    k_e,             # 探査抑制強度
    k_c,             # 衝突抑制強度
    branch_threshold, # 分岐閾値（学習可能）
    merge_threshold   # 統合閾値（学習可能）
]

# 動きやすさ指標の重み
mobility_weights = {
    'obstacle_distance': 0.4,    # 障害物距離（40%）
    'robot_distance': 0.3,       # ロボット間距離（30%）
    'exploration_density': 0.3   # 探査密度（30%）
}

# 群制御制限
MIN_SWARM_SIZE = 3  # 最小群サイズ
```

### 推奨設定

- **小規模群（3-5 台）**: 分岐閾値 0.4, 統合閾値 0.2
- **中規模群（6-10 台）**: 分岐閾値 0.3, 統合閾値 0.3
- **大規模群（10 台以上）**: 分岐閾値 0.25, 統合閾値 0.4

## 今後の拡張

### 1. 適応的閾値

- 探査履歴に基づく動的閾値調整
- 環境特性に応じた閾値最適化
- 学習による自動閾値調整

### 2. 高度な動きやすさ指標

- エネルギー効率の考慮
- 通信品質の影響
- 地形の複雑さの評価

### 3. 多段階群制御

- 階層的な群構造
- 複数レベルの分岐・統合
- 動的階層調整

### 4. 予測的群制御

- 将来の動きやすさ予測
- 予防的な群再編成
- 長期的な効率最適化

### 5. 協調学習

- 群間の情報共有
- 協調的な学習戦略
- 集団知能の活用
