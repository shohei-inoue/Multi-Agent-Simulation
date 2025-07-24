# 統合アルゴリズム (Integration Algorithm)

## 概要

統合アルゴリズムは、群ロボットシステムにおいて複数の群を動的に統合するためのアルゴリズム群です。探査効率の最適化とリソースの効率的な利用を実現するため、複数の統合戦略を提供します。

## 設計思想

### 1. 効率性の向上

- 重複する探査領域の統合
- リソースの最適化
- 探査効率の最大化

### 2. 学習統合

- 統合時の学習パラメータの統合
- 経験の共有
- 知識の集約

### 3. 適応性

- 環境変化への対応
- 動的な統合判断
- システム安定性の維持

## アーキテクチャ

### 1. 基底クラス

```python
class IntegrationAlgorithm:
    """Base class for integration algorithms"""

    def select_integration_target(self, source_swarm, target_swarms):
        """Select target swarm for integration"""

    def merge_learning_info(self, source_swarm, target_swarm):
        """Merge learning information between swarms"""
```

### 2. 実装クラス

#### NearestIntegrationAlgorithm

```python
class NearestIntegrationAlgorithm(IntegrationAlgorithm):
    """Nearest-based integration algorithm"""

    def select_integration_target(self, source_swarm, target_swarms):
        """Select nearest swarm as integration target"""

    def merge_learning_info(self, source_swarm, target_swarm):
        """Merge learning information using weighted average"""
```

## 統合条件

### 1. 基本条件

```python
def check_integration_condition(self, mobility_score, swarm_count, exploration_overlap):
    """Check if integration conditions are met"""
    return (
        self.integration_enabled and
        mobility_score < self.integration_threshold and
        swarm_count >= self.min_swarms_for_integration and
        exploration_overlap and
        self.check_cooldown()
    )
```

### 2. 条件パラメータ

- **integration_enabled**: 統合機能の有効化
- **integration_threshold**: 統合閾値（デフォルト: 0.2）
- **min_swarms_for_integration**: 統合に必要な最小群数（デフォルト: 2）
- **swarm_merge_cooldown**: 統合クールダウン（デフォルト: 15.0 秒）

### 3. 探査領域の重複チェック

```python
def check_exploration_area_overlap(self, swarm1_id, swarm2_id):
    """Check if two swarms have overlapping exploration areas"""
    swarm1 = self._find_swarm_by_id(swarm1_id)
    swarm2 = self._find_swarm_by_id(swarm2_id)

    if swarm1 is None or swarm2 is None:
        return False

    # 2つの群のleader間の距離を計算
    leader1_pos = swarm1.leader.coordinate
    leader2_pos = swarm2.leader.coordinate
    distance = np.linalg.norm(leader1_pos - leader2_pos)

    # 探査領域の半径（outer_boundary）の2倍以内なら重複とみなす
    overlap_threshold = self.exploration_radius * 2.0

    return distance <= overlap_threshold
```

## 実装詳細

### 1. NearestIntegrationAlgorithm

#### 統合対象の選択

```python
def select_integration_target(self, source_swarm, target_swarms):
    """Select nearest swarm as integration target"""
    if not target_swarms:
        return None

    source_pos = source_swarm.leader.coordinate
    nearest_swarm = None
    min_distance = float('inf')

    for target_swarm in target_swarms:
        # 探査領域の重複をチェック
        if hasattr(self.env, 'check_exploration_area_overlap'):
            if not self.env.check_exploration_area_overlap(
                source_swarm.swarm_id, target_swarm.swarm_id
            ):
                continue  # 探査領域が重複していない場合はスキップ

        target_pos = target_swarm.leader.coordinate
        distance = np.linalg.norm(source_pos - target_pos)

        if distance < min_distance:
            min_distance = distance
            nearest_swarm = target_swarm

    return nearest_swarm
```

#### 学習情報の統合

```python
def merge_learning_info(self, source_swarm, target_swarm):
    """Merge learning information using weighted average"""
    if not hasattr(source_swarm, 'learning_params') or not hasattr(target_swarm, 'learning_params'):
        return

    # 重み付き平均による統合
    source_weight = 0.3  # 統合される群の重み
    target_weight = 0.7  # 統合先の群の重み

    # 学習パラメータの統合
    for param_name in ['th', 'k_e', 'k_c']:
        if hasattr(source_swarm.learning_params, param_name) and hasattr(target_swarm.learning_params, param_name):
            source_value = getattr(source_swarm.learning_params, param_name)
            target_value = getattr(target_swarm.learning_params, param_name)

            merged_value = (source_value * source_weight + target_value * target_weight)
            setattr(target_swarm.learning_params, param_name, merged_value)
```

### 2. 統合プロセス

#### 統合実行

```python
def execute_integration(self, source_swarm_id, target_swarm_id):
    """Execute integration process"""

    # 1. 統合条件のチェック
    if not self.check_integration_condition(...):
        return None

    # 2. 統合対象の選択
    target_swarm = self.integration_algorithm.select_integration_target(...)

    if target_swarm is None:
        return None

    # 3. 学習情報の統合
    self.integration_algorithm.merge_learning_info(source_swarm, target_swarm)

    # 4. フォロワーの移動
    for follower in source_swarm.followers:
        target_swarm.add_follower(follower)
        follower.agent_coordinate = target_swarm.leader.coordinate

    # 5. リーダーの統合
    source_swarm.leader.set_role(RobotRole.FOLLOWER)
    target_swarm.add_follower(source_swarm.leader)
    source_swarm.leader.agent_coordinate = target_swarm.leader.coordinate

    # 6. 軌跡の保存
    self.save_integrated_leader_trajectory(source_swarm.leader)

    # 7. 元の群の削除
    self.remove_swarm(source_swarm)

    return target_swarm
```

#### 統合後の処理

```python
def post_integration_processing(self, target_swarm, integrated_followers):
    """Post-integration processing"""

    # 1. フォロワーの再配置
    for follower in integrated_followers:
        follower.swarm_id = target_swarm.swarm_id
        follower.agent_coordinate = target_swarm.leader.coordinate

    # 2. 統合されたリーダーの処理
    if hasattr(target_swarm, 'integrated_leader'):
        integrated_leader = target_swarm.integrated_leader
        integrated_leader.set_role(RobotRole.FOLLOWER)
        integrated_leader.agent_coordinate = target_swarm.leader.coordinate

    # 3. システムエージェントの更新
    self.system_agent.remove_swarm_agent(source_swarm_id)
```

## 学習情報の統合

### 1. 重み付き平均統合

```python
def weighted_average_merge(self, source_params, target_params, weights):
    """Merge parameters using weighted average"""
    merged_params = {}

    for param_name in source_params.keys():
        if param_name in target_params:
            source_value = source_params[param_name]
            target_value = target_params[param_name]

            merged_value = (
                source_value * weights['source'] +
                target_value * weights['target']
            )
            merged_params[param_name] = merged_value

    return merged_params
```

### 2. 経験バッファの統合

```python
def merge_experience_buffers(self, source_buffer, target_buffer):
    """Merge experience buffers"""
    if not hasattr(source_buffer, 'sample') or not hasattr(target_buffer, 'extend'):
        return

    # 統合される群の経験の一部を移譲
    shared_experience = source_buffer.sample(
        size=min(50, len(source_buffer))
    )

    # 統合先の群の経験バッファに追加
    target_buffer.extend(shared_experience)
```

## 設定パラメータ

### 1. SystemAgentParam

```python
@dataclass
class IntegrationConditionParam:
    integration_enabled: bool = True
    integration_threshold: float = 0.2  # 統合閾値を下げて、より低いmobility_scoreで統合
    min_swarms_for_integration: int = 2
    integration_learning_merge: bool = True
    integration_target_selection: str = "nearest"
    integration_learning_merge_method: str = "weighted_average"
    swarm_merge_cooldown: float = 15.0  # クールダウンを15秒に延長
    integration_algorithm: str = "nearest"
```

### 2. アルゴリズム選択

- **"nearest"**: NearestIntegrationAlgorithm
- **"largest"**: LargestIntegrationAlgorithm（将来実装）
- **"performance_based"**: PerformanceBasedIntegrationAlgorithm（将来実装）

## 使用例

### 1. 基本的な使用

```python
# 統合アルゴリズムの作成
integration_algorithm = NearestIntegrationAlgorithm()

# 統合条件のチェック
if system_agent.check_integration(swarm_state):
    # 統合の実行
    integrated_swarm = system_agent.execute_integration(
        source_swarm_id=1,
        target_swarm_id=2
    )
```

### 2. カスタム設定

```python
# カスタム統合条件
integration_condition = IntegrationConditionParam(
    integration_threshold=0.1,  # より低い閾値
    swarm_merge_cooldown=30.0,  # より長いクールダウン
    integration_algorithm="nearest"
)

system_agent.integration_condition = integration_condition
```

## パフォーマンス最適化

### 1. 統合頻度の制御

- クールダウン期間の設定
- 統合条件の動的調整
- システム負荷の監視

### 2. メモリ効率

- 不要なデータの削除
- 学習情報の効率的な統合
- リソース使用量の最適化

### 3. 計算効率

- 統合条件の早期チェック
- 並列処理の活用
- キャッシュの活用

## 安全性と安定性

### 1. 統合検証

```python
def validate_integration(self, source_swarm, target_swarm):
    """Validate integration process"""

    # 1. 群の存在確認
    if source_swarm is None or target_swarm is None:
        return False

    # 2. 探査領域の重複確認
    if not self.check_exploration_area_overlap(
        source_swarm.swarm_id, target_swarm.swarm_id
    ):
        return False

    # 3. クールダウン確認
    if not self.check_cooldown():
        return False

    return True
```

### 2. エラーハンドリング

```python
def safe_integration(self, source_swarm_id, target_swarm_id):
    """Safe integration with error handling"""
    try:
        # 統合の実行
        result = self.execute_integration(source_swarm_id, target_swarm_id)

        if result is None:
            self.logger.warning(f"Integration failed for swarms {source_swarm_id} -> {target_swarm_id}")
            return False

        self.logger.info(f"Integration successful: swarm {source_swarm_id} -> {target_swarm_id}")
        return True

    except Exception as e:
        self.logger.error(f"Integration error: {e}")
        return False
```

## 今後の拡張

### 1. 新しい統合戦略

- パフォーマンスベース統合
- 適応的統合閾値
- マルチ群統合

### 2. 学習統合の強化

- より高度な知識統合
- メタ学習の導入
- 分散学習の実装

### 3. 安全性の向上

- より厳密な統合検証
- ロールバック機能
- 異常検知システム

### 4. 可視化の強化

- 統合プロセスの可視化
- 学習情報の統合過程の表示
- リアルタイムモニタリング
