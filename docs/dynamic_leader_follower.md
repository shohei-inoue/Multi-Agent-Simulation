# 動的 Leader-Follower システム

## 概要

群ロボットシミュレーションにおいて、架空の agent ではなく、実際の Red ロボットが leader と follower の役割を動的に切り替えるシステムを実装しました。

## システム構成

### 1. ロボットの役割

```python
class RobotRole(Enum):
    FOLLOWER = 0  # followerロボット
    LEADER = 1    # leaderロボット
```

### 2. 動的切り替えメカニズム

- **切り替え間隔**: 設定可能なステップ数（デフォルト: 10 ステップ）
- **切り替え方式**: ラウンドロビン方式（順番に切り替え）
- **切り替えタイミング**: 各ステップでカウンターをチェック

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

def get_leader_action(self, algorithm, state, sampled_params, episode, log_dir):
    """leaderとしての行動を取得"""
    # leaderの状態を構築
    leader_state = {
        "agent_coordinate_x": self.x,
        "agent_coordinate_y": self.y,
        "agent_azimuth": self.leader_azimuth,
        "agent_collision_flag": 1.0 if self.collision_flag else 0.0,
        "agent_step_count": self.leader_step_count,
        "follower_collision_data": state.get("follower_collision_data", [])
    }

    # アルゴリズムで行動決定
    return algorithm.policy(leader_state, sampled_params, episode, log_dir)

def execute_leader_action(self, action_dict):
    """leaderとしての行動を実行"""
    # 行動を実行し、衝突判定を行う
    # 成功/失敗を返す
```

### 2. 環境クラスの変更

#### 動的システムの管理

```python
# 動的leader-followerシステム
self.current_leader_index   = 0    # 現在のleaderのインデックス
self.leader_switch_interval = 10   # leader切り替え間隔
self.leader_switch_counter  = 0    # leader切り替えカウンター
self.current_leader         = None # 現在のleaderオブジェクト
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
```

### 3. Step 処理の変更

#### 従来の処理

```python
# 架空のagentが行動を決定・実行
agent_coordinate, collision_flag = self.next_coordinate(dy, dx)
# followerがagentを追従
for follower in self.follower_robots:
    follower.step_motion(agent_coordinate=agent_coordinate)
```

#### 新しい処理

```python
# leader切り替えチェック
if self.leader_switch_counter >= self.leader_switch_interval:
    self._switch_leader()
    self.leader_switch_counter = 0

# 現在のleaderの行動を実行
success = self.current_leader.execute_leader_action(action)

# followerが現在のleaderを追従
for robot in self.robots:
    if robot.role == RobotRole.FOLLOWER:
        robot.step_motion(agent_coordinate=self.current_leader.coordinate)
```

## 可視化の改善

### ロボットの表示

- **Leader**: 青色、大きいサイズ（15）
- **Follower**: 赤色、小さいサイズ（10）
- **軌跡**: 全ロボットの移動軌跡を表示

```python
for i, robot in enumerate(self.robots):
    color = 'blue' if robot.role == RobotRole.LEADER else 'red'
    size = 15 if robot.role == RobotRole.LEADER else 10
    label = "leader" if robot.role == RobotRole.LEADER else "follower"

    ax.scatter(x=robot.x, y=robot.y, color=color, s=size, label=label)
```

## 利点

### 1. 現実性の向上

- 架空の agent ではなく、実際のロボットが leader として動作
- より現実的な群ロボットシステムのシミュレーション

### 2. 動的適応性

- 環境変化に応じた leader の動的切り替え
- 故障や性能低下に対する耐性向上

### 3. 分散制御

- 各ロボットが leader としての能力を持つ
- 中央集権的な制御からの脱却

### 4. 学習効果の向上

- 各ロボットが leader 経験を積む
- より豊富な学習データの生成

## 設定パラメータ

### Leader 切り替え設定

```python
self.leader_switch_interval = 10   # 切り替え間隔（ステップ数）
```

### 推奨設定

- **小規模群（3-5 台）**: 5-10 ステップ
- **中規模群（6-10 台）**: 10-15 ステップ
- **大規模群（10 台以上）**: 15-20 ステップ

## 今後の拡張

### 1. 適応的切り替え

- 探査効率に基づく leader 選択
- 性能評価による動的切り替え

### 2. 複数 leader 対応

- 同時に複数の leader が存在
- 領域分割による並列探査

### 3. 学習ベース切り替え

- 強化学習による最適な leader 選択
- 各ロボットの学習履歴を考慮

### 4. 通信機能

- ロボット間の通信による協調
- 情報共有による効率化
