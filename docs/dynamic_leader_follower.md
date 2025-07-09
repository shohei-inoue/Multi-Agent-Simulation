# 動的 Leader-Follower システム

## 概要

群ロボットシミュレーションにおいて、実際の Red ロボットが leader と follower の役割を動的に切り替えるシステムを実装しました。各リーダーが独立した状態空間、アルゴリズム、学習エージェントを持ち、群分岐・統合機能と組み合わせて効率的な探査を実現します。

## システム構成

### 1. ロボットの役割

```python
class RobotRole(Enum):
    FOLLOWER = 0  # followerロボット
    LEADER = 1    # leaderロボット
```

### 2. 群システムの構造

```python
@dataclass
class Swarm:
    swarm_id: int
    leader: Red
    followers: List[Red]
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

### 3. 動的切り替えメカニズム

- **切り替え間隔**: 設定可能なステップ数（デフォルト: 10 ステップ）
- **切り替え方式**: ラウンドロビン方式（順番に切り替え）
- **切り替えタイミング**: 各ステップでカウンターをチェック
- **群制限**: 3 台以下の群は分岐しない

## 実装詳細

### 1. Red ロボットクラスの拡張

#### Leader 機能の追加

```python
def set_role(self, role: RobotRole):
    """ロボットの役割を設定"""
    self.role = role
    if role == RobotRole.LEADER:
        self.leader_step_count = 0
        self.leader_azimuth = self.direction_angle
        # リーダーになった時点からの軌跡を記録
        self.leader_trajectory_data = {
            'x': [self.x],
            'y': [self.y]
        }

def get_leader_action(self, algorithm, state, sampled_params, episode, log_dir):
    """leaderとしての行動を取得"""
    # leaderの状態を構築
    leader_state = {
        "agent_coordinate_x": self.x,
        "agent_coordinate_y": self.y,
        "agent_azimuth": self.leader_azimuth,
        "agent_collision_flag": 1.0 if self.collision_flag else 0.0,
        "agent_step_count": self.leader_step_count,
        "follower_collision_data": state.get("follower_collision_data", []),
        "follower_mobility_scores": state.get("follower_mobility_scores", [])
    }

    # アルゴリズムで行動決定
    return algorithm.policy(leader_state, sampled_params, episode, log_dir)

def execute_leader_action(self, action_dict):
    """leaderとしての行動を実行"""
    theta = action_dict.get("theta", 0.0)

    # 移動量を計算
    dx = np.cos(theta) * self.amount_of_movement
    dy = np.sin(theta) * self.amount_of_movement

    # 次の座標を計算
    next_coordinate, collision_flag = self.env.next_coordinate(dy, dx)

    if not collision_flag:
        # 移動が成功
        self.x = next_coordinate[1]
        self.y = next_coordinate[0]
        self.coordinate = next_coordinate
        self.leader_azimuth = theta
        self.leader_step_count += 1

        # 軌跡を記録
        if hasattr(self, 'leader_trajectory_data'):
            self.leader_trajectory_data['x'].append(self.x)
            self.leader_trajectory_data['y'].append(self.y)

        return True
    else:
        # 衝突が発生
        self.collision_flag = True
        return False
```

### 2. 環境クラスの変更

#### 動的システムの管理

```python
# 動的leader-followerシステム
self.current_leader_index   = 0    # 現在のleaderのインデックス
self.leader_switch_interval = 10   # leader切り替え間隔
self.leader_switch_counter  = 0    # leader切り替えカウンター

# 群システム
self.swarms = []                    # 群のリスト
self.swarm_id_counter = 0          # 群IDカウンター
self.initial_swarm_id = 0          # 初期群ID

# リーダーエージェント
self.leader_agents = {}            # 各リーダーの独立したエージェント
self.leader_models = {}            # 各リーダーの独立したモデル
```

#### Leader 切り替え機能

```python
def _switch_leader(self):
    """leaderを切り替える"""
    # 現在のleaderをfollowerに変更
    self.current_leader.set_role(RobotRole.FOLLOWER)

    # 次のleaderを選択（ラウンドロビン方式）
    self.current_leader_index = (self.current_leader_index + 1) % self.__robot_num
    self.current_leader = self.robots[self.current_leader_index]
    self.current_leader.set_role(RobotRole.LEADER)

    print(f"Leader switched to robot_{self.current_leader_index}")
```

#### 群分岐・統合機能

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

    elif mode == 2:  # 群統合
        # 最も近い群に統合
        current_swarm = self._find_swarm_by_leader(self.current_leader)
        if current_swarm and len(self.swarms) > 1:
            # 最も近い群を探す
            closest_swarm = self._find_closest_swarm(current_swarm)
            if closest_swarm and closest_swarm != current_swarm:
                self._merge_swarms(closest_swarm, current_swarm)
```

### 3. Step 処理の変更

#### 新しい処理

```python
def step(self, action):
    """環境のステップ → 動的leader-followerシステム + 群分岐・統合"""
    # 群分岐・統合の処理
    mode = action.get('mode', 0)
    self._handle_swarm_mode(mode)

    # leader切り替えチェック（通常時のみ）
    if mode == 0:
        self.leader_switch_counter += 1
        if self.leader_switch_counter >= self.leader_switch_interval:
            self._switch_leader()
            self.leader_switch_counter = 0

    # 各群のleaderの行動を実行
    leader_rewards = {}  # 各リーダーの報酬を個別に管理
    leader_states = {}   # 各リーダーの独立した状態空間

    for swarm in self.swarms:
        if swarm.leader.role == RobotRole.LEADER:
            # 各リーダーの個別状態空間を構築
            leader_state = self._build_leader_state(swarm.leader, swarm)
            leader_states[swarm.leader.id] = leader_state

            # 各リーダーが独立した行動を決定
            action_tensor, action_dict = swarm.leader.get_leader_action(
                algorithm=None,  # リーダー固有のアルゴリズムを使用
                state=leader_state,
                sampled_params=action.get('sampled_params', [0.5, 10.0, 5.0, 0.3, 0.7]),
                episode=self.agent_step,
                log_dir=None
            )

            # 各leaderの行動を実行
            previous_coordinate = swarm.leader.coordinate.copy()
            success = swarm.leader.execute_leader_action(action_dict)
            if not success:
                print(f"Leader {swarm.leader.id} collision detected")
            else:
                # リーダーの移動が成功した場合、探査マップを更新
                self.update_exploration_map(previous_coordinate, swarm.leader.coordinate)

            # 各リーダーの個別報酬を計算
            leader_reward = self._calculate_leader_reward(swarm.leader, previous_coordinate)
            leader_rewards[swarm.leader.id] = leader_reward

    # 各群のfollowerの探査行動（非同期実行）
    follower_results = []
    for _ in range(self.__offset.one_explore_step):
        # 各群のfollowerを非同期で実行
        futures = []
        for swarm in self.swarms:
            for follower in swarm.followers:
                if follower.role == RobotRole.FOLLOWER:
                    future = self.follower_executor.submit(
                        self._execute_follower_step_async_by_swarm, swarm, follower
                    )
                    futures.append(future)

        # 全followerの完了を待機
        for future in as_completed(futures):
            try:
                result = future.result()
                follower_results.append(result)
            except Exception as e:
                print(f"Error in follower execution: {e}")

    return self.state, reward, done, truncated, {}
```

## 可視化の改善

### ロボットの表示

- **Leader**: 各群ごとに異なる色、星マーカー（★）
- **Follower**: リーダーと同じ色、円マーカー（●）
- **軌跡**: リーダーごとに色分けされた軌跡

```python
# 各群のロボットを描画
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']

for swarm_idx, swarm in enumerate(self.swarms):
    # リーダーIDに基づいてカラーを決定
    leader_id = int(swarm.leader.id) if swarm.leader.id.isdigit() else hash(swarm.leader.id) % len(colors)
    leader_color = colors[leader_id % len(colors)]
    trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]

    # leaderの描画（探査中心として表示）
    leader = swarm.leader
    ax.scatter(
        x=leader.data['x'].iloc[-1],
        y=leader.data['y'].iloc[-1],
        color=leader_color,  # リーダー固有のカラー
        s=25,
        marker='*',
        edgecolors='black',
        linewidth=1,
        label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})" if swarm_idx == 0 else None
    )

    # leaderの軌跡（リーダーになった時点からの軌跡を表示）
    if hasattr(leader, 'leader_trajectory_data') and len(leader.leader_trajectory_data) > 0:
        ax.plot(
            leader.leader_trajectory_data['x'],
            leader.leader_trajectory_data['y'],
            color=trajectory_color,  # リーダー固有の軌跡カラー
            linewidth=1.5,
            alpha=0.6,  # 適度な透明度
            linestyle='-',
            label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})" if swarm_idx == 0 else None
        )

    # followerの描画
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

## 利点

### 1. 現実性の向上

- 架空の agent ではなく、実際のロボットが leader として動作
- より現実的な群ロボットシステムのシミュレーション

### 2. 動的適応性

- 環境変化に応じた leader の動的切り替え
- 故障や性能低下に対する耐性向上
- 群分岐・統合による適応的な組織化

### 3. 分散制御

- 各ロボットが leader としての能力を持つ
- 中央集権的な制御からの脱却
- 各群が独立して動作

### 4. 学習効果の向上

- 各ロボットが leader 経験を積む
- より豊富な学習データの生成
- 各リーダーが独立した学習エージェントを持つ

### 5. 効率的な探査

- 群分岐による並列探査
- 群統合による協調探査
- 動きやすさ指標に基づく最適な群制御

## 設定パラメータ

### Leader 切り替え設定

```python
self.leader_switch_interval = 10   # 切り替え間隔（ステップ数）
```

### 群制御設定

```python
# 分岐・統合の閾値（学習可能）
branch_threshold = 0.3  # 分岐閾値
merge_threshold = 0.7   # 統合閾値

# 最小群サイズ制限
MIN_SWARM_SIZE = 3  # 3台以下の群は分岐しない
```

### 推奨設定

- **小規模群（3-5 台）**: 5-10 ステップ
- **中規模群（6-10 台）**: 10-15 ステップ
- **大規模群（10 台以上）**: 15-20 ステップ

## 今後の拡張

### 1. 適応的切り替え

- 探査効率に基づく leader 選択
- 性能評価による動的切り替え
- 学習による最適な切り替えタイミング

### 2. 複数 leader 対応

- 同時に複数の leader が存在
- 領域分割による並列探査
- 群間の協調制御

### 3. 学習ベース切り替え

- 強化学習による最適な leader 選択
- 各ロボットの学習履歴を考慮
- 群制御の学習最適化

### 4. 通信機能

- ロボット間の通信による協調
- 情報共有による効率化
- 群間の情報交換

### 5. 動的群制御

- 環境に応じた群サイズの最適化
- 探査効率に基づく群再編成
- 障害物回避に特化した群制御
