# Follower 非同期化による改善

## 概要

群ロボットシミュレーションにおいて、follower の動きを同期式から非同期式に変更することで、より現実的で効率的な群行動を実現しました。

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

### 2. 非同期式（改善後）の実装

```python
# 非同期式実装
for _ in range(self.__offset.one_explore_step):
    # 各followerを非同期で実行
    futures = []
    for index in range(len(self.follower_robots)):
        future = self.follower_executor.submit(self._execute_follower_step_async, index)
        futures.append(future)

    # 全followerの完了を待機
    follower_results = []
    for future in as_completed(futures):
        result = future.result()
        follower_results.append(result)
```

**改善点:**

- 各 follower が並列に実行される
- より現実的な群行動のシミュレーション
- 実行時間の短縮

## 実装詳細

### 1. ThreadPoolExecutor の導入

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# 非同期実行用のスレッドプール
self.follower_executor = ThreadPoolExecutor(max_workers=self.__robot_num)
```

### 2. 非同期実行メソッド

```python
def _execute_follower_step_async(self, follower_index):
    """
    followerの1ステップを非同期実行する
    """
    try:
        follower = self.follower_robots[follower_index]
        previous_coordinate = follower.coordinate.copy()

        # followerの動きを実行
        follower.step_motion(agent_coordinate=self.agent_coordinate)

        # 探査マップを更新
        self.update_exploration_map(previous_coordinate, follower.coordinate)

        # 結果を返す
        return {
            'index': follower_index,
            'collision_point': collision_point,
            'collision_data': follower.get_collision_data()
        }
    except Exception as e:
        print(f"Error in follower {follower_index}: {e}")
        return {'index': follower_index, 'collision_point': None, 'collision_data': []}
```

### 3. 結果の収集

```python
# 非同期実行結果から衝突データを収集
for result in follower_results:
    collision_data = result['collision_data']
    follower_collision_data.extend(collision_data)
    current_follower_collision_count += len(collision_data)
```

## 性能改善

### 実行時間の短縮

- **同期式**: 各 follower が順次実行されるため、実行時間は `follower数 × 1ステップ時間` に比例
- **非同期式**: 全 follower が並列実行されるため、実行時間は `max(各followerの1ステップ時間)` に比例

### 現実性の向上

- 各 follower が独立して動作するため、より現実的な群行動をシミュレート
- 個体間の相互作用がより自然に表現される

## 注意事項

### 1. スレッドセーフティ

- 各 follower の状態更新は独立して行われる
- 共有リソース（探査マップなど）へのアクセスは適切に同期化

### 2. エラーハンドリング

- 各 follower の実行でエラーが発生した場合の適切な処理
- 例外が発生しても他の follower の実行に影響しない

### 3. リソース管理

- ThreadPoolExecutor の適切なクリーンアップ
- 環境終了時のスレッドプールのシャットダウン

## 今後の拡張

### 1. より細かい制御

- 各 follower の実行タイミングの個別制御
- 優先度付き実行

### 2. 分散処理

- マルチプロセスによる並列化
- GPU 並列化の検討

### 3. リアルタイム性

- リアルタイムシミュレーションへの対応
- 可変タイムステップの実装
