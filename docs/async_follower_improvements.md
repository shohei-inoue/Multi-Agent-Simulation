# Follower 非同期化による改善

## 概要

群ロボットシミュレーションにおいて、follower の動きを同期式から非同期式に変更することで、より現実的で効率的な群行動を実現しました。群ベースの非同期実行により、各群の follower が独立して並列に動作します。

## 変更内容

### 1. 同期式（従来）の問題点

```python
# 従来の同期式実装
for _ in range(self.__offset.one_explore_step):
    for index in range(len(self.follower_robots)):
        # 各followerが順番に実行
        self.follower_robots[index].step_motion(agent_coordinate=self.agent_coordinate)
        # 次のfollowerが前のfollowerの完了を待つ
```

**問題点:**

- 各 follower が順番に実行されるため、後続の follower は前の follower の完了を待つ
- 現実の群ロボットでは、各ロボットは独立して並列に動作する
- シミュレーション時間が長くなる
- 群システムに対応していない

### 2. 非同期式（改善後）の実装

```python
# 群ベースの非同期式実装
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
    follower_results = []
    for future in as_completed(futures):
        try:
            result = future.result()
            follower_results.append(result)

            # 衝突点を追加
            if result['collision_point']:
                self.follower_collision_points.append(result['collision_point'])
        except Exception as e:
            print(f"Error in follower execution: {e}")
```

**改善点:**

- 各群の follower が並列に実行される
- より現実的な群行動のシミュレーション
- 実行時間の短縮
- 群システムとの完全な統合

## 実装詳細

### 1. ThreadPoolExecutor の導入

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# 非同期実行用のスレッドプール
self.follower_executor = ThreadPoolExecutor(max_workers=self.__robot_num)
self.follower_futures = []
```

### 2. 群ベース非同期実行メソッド

```python
def _execute_follower_step_async_by_swarm(self, swarm: Swarm, follower: Red):
    """
    群ベースのfollowerの1ステップを非同期実行する
    """
    try:
        if follower.role != RobotRole.FOLLOWER:
            return {
                'index': -1,
                'collision_point': None,
                'collision_data': []
            }

        previous_coordinate = follower.coordinate.copy()

        # followerの動きを実行（所属群のleaderを参照）
        follower.step_motion(agent_coordinate=swarm.leader.coordinate)

        # 探査マップを更新
        self.update_exploration_map(previous_coordinate, follower.coordinate)

        # ロボットデータを記録
        self.scorer.add_robot_data(
            step=self.agent_step,
            robot_id=follower.id,
            x=follower.x,
            y=follower.y,
            collision_flag=follower.collision_flag,
            boids_flag=follower.boids_flag.value,
            distance=follower.distance
        )

        # 衝突フラグがTrueなら座標を追加
        collision_point = None
        if follower.collision_flag:
            cx = follower.coordinate[1]
            cy = follower.coordinate[0]
            collision_point = (cx, cy)

        return {
            'index': -1,
            'collision_point': collision_point,
            'collision_data': follower.get_collision_data()
        }
    except Exception as e:
        print(f"Error in follower {follower.id}: {e}")
        return {
            'index': -1,
            'collision_point': None,
            'collision_data': []
        }
```

### 3. 従来方式の非同期実行メソッド（後方互換性）

```python
def _execute_follower_step_async(self, follower_index):
    """
    followerの1ステップを非同期実行する（従来方式）
    """
    try:
        robot = self.robots[follower_index]
        if robot.role != RobotRole.FOLLOWER:
            return {
                'index': follower_index,
                'collision_point': None,
                'collision_data': []
            }

        previous_coordinate = robot.coordinate.copy()

        # followerの動きを実行（現在のleaderを参照）
        robot.step_motion(agent_coordinate=self.current_leader.coordinate)

        # 探査マップを更新
        self.update_exploration_map(previous_coordinate, robot.coordinate)

        # ロボットデータを記録
        self.scorer.add_robot_data(
            step=self.agent_step,
            robot_id=robot.id,
            x=robot.x,
            y=robot.y,
            collision_flag=robot.collision_flag,
            boids_flag=robot.boids_flag.value,
            distance=robot.distance
        )

        # 衝突フラグがTrueなら座標を追加
        collision_point = None
        if robot.collision_flag:
            cx = robot.coordinate[1]
            cy = robot.coordinate[0]
            collision_point = (cx, cy)

        return {
            'index': follower_index,
            'collision_point': collision_point,
            'collision_data': robot.get_collision_data()
        }
    except Exception as e:
        print(f"Error in follower {follower_index}: {e}")
        return {
            'index': follower_index,
            'collision_point': None,
            'collision_data': []
        }
```

### 4. 結果の収集と処理

```python
# followerから衝突データ収集
follower_collision_data = []
current_follower_collision_count = 0
follower_mobility_scores = []

for result in follower_results:
    collision_data = result['collision_data']
    follower_collision_data.extend(collision_data)
    current_follower_collision_count += len(collision_data)
    self.scorer.follower_collision_count += len(collision_data)

# 各群の全followerのmobility_scoreを集約
for swarm in self.swarms:
    for follower in swarm.followers:
        follower_mobility_scores.append(follower.mobility_score)

# 最大ロボット数（10個）に合わせてパディング
MAX_ROBOT_NUM = 10
while len(follower_mobility_scores) < MAX_ROBOT_NUM:
    follower_mobility_scores.append(0.0)
follower_mobility_scores = follower_mobility_scores[:MAX_ROBOT_NUM]  # 最大数を超える場合は切り詰め

self.state['follower_mobility_scores'] = follower_mobility_scores
```

## 性能改善

### 実行時間の短縮

- **同期式**: 各 follower が順次実行されるため、実行時間は `follower数 × 1ステップ時間` に比例
- **非同期式**: 全 follower が並列実行されるため、実行時間は `max(各followerの1ステップ時間)` に比例
- **群ベース**: 各群の follower が独立して並列実行されるため、さらに効率的

### 現実性の向上

- 各 follower が独立して動作するため、より現実的な群行動をシミュレート
- 個体間の相互作用がより自然に表現される
- 群ごとの独立した動作により、より現実的な群制御を実現

### スケーラビリティの向上

- 群数が増えても効率的に並列実行
- ロボット数の増加に対応した動的スレッドプール管理
- 群分岐・統合時の動的な並列度調整

## 注意事項

### 1. スレッドセーフティ

- 各 follower の状態更新は独立して行われる
- 共有リソース（探査マップなど）へのアクセスは適切に同期化
- 群システムの状態変更時の適切な同期処理

### 2. エラーハンドリング

- 各 follower の実行でエラーが発生した場合の適切な処理
- 例外が発生しても他の follower の実行に影響しない
- 群システムの整合性を保つためのエラー処理

### 3. リソース管理

- ThreadPoolExecutor の適切なクリーンアップ
- 環境終了時のスレッドプールのシャットダウン
- 群分岐・統合時のリソース再割り当て

```python
def close(self):
    """
    環境を閉じる際のクリーンアップ
    """
    if hasattr(self, 'follower_executor'):
        self.follower_executor.shutdown(wait=True)
```

### 4. 状態の整合性

- 群システムの状態と非同期実行の整合性確保
- 動きやすさ指標の正確な計算
- 探査マップの一貫性維持

## 今後の拡張

### 1. より細かい制御

- 各群の実行タイミングの個別制御
- 優先度付き実行
- 群ごとの実行速度調整

### 2. 分散処理

- マルチプロセスによる並列化
- GPU 並列化の検討
- クラウド分散処理への対応

### 3. リアルタイム性

- リアルタイムシミュレーションへの対応
- 可変タイムステップの実装
- 動的負荷分散

### 4. 群間協調

- 群間の情報共有による協調実行
- 群間の実行タイミング調整
- 協調的な探査戦略の実装

### 5. 適応的並列化

- 環境に応じた並列度の動的調整
- 計算負荷に基づく最適化
- エネルギー効率を考慮した並列化
