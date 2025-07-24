# 分岐アルゴリズム (Branch Algorithm)

## 概要

分岐アルゴリズムは、群ロボットシステムにおいて動的に新しい群を作成するためのアルゴリズム群です。探査効率の向上と適応的な群管理を実現するため、複数の分岐戦略を提供します。

## 設計思想

### 1. 動的適応性

- 環境の変化に応じた群の再編成
- 探査効率の最適化
- 学習情報の継承

### 2. 学習統合

- 分岐時の学習パラメータの引き継ぎ
- 新しい群への知識移転
- 適応的な行動調整

### 3. 安全性

- 最小フォロワー数の保証
- 分岐条件の厳密なチェック
- システム安定性の維持

## アーキテクチャ

### 1. 基底クラス

```python
class BranchAlgorithm:
    """Base class for branching algorithms"""

    def select_branch_followers(self, source_swarm, valid_directions, mobility_scores):
        """Select followers for the new swarm"""

    def select_branch_leader(self, source_swarm, valid_directions, mobility_scores):
        """Select leader for the new swarm"""
```

### 2. 実装クラス

#### MobilityBasedBranchAlgorithm

```python
class MobilityBasedBranchAlgorithm(BranchAlgorithm):
    """Mobility-based branching algorithm"""

    def select_branch_followers(self, source_swarm, valid_directions, mobility_scores):
        """Select followers based on mobility scores"""

    def select_branch_leader(self, source_swarm, valid_directions, mobility_scores):
        """Select leader with highest mobility score"""
```

#### RandomBranchAlgorithm

```python
class RandomBranchAlgorithm(BranchAlgorithm):
    """Random branching algorithm"""

    def select_branch_followers(self, source_swarm, valid_directions, mobility_scores):
        """Select followers randomly"""

    def select_branch_leader(self, source_swarm, valid_directions, mobility_scores):
        """Select leader randomly"""
```

## 分岐条件

### 1. 基本条件

```python
def check_branch_condition(self, direction_count, mobility_score, follower_count):
    """Check if branching conditions are met"""
    return (
        self.branch_enabled and
        direction_count >= self.min_directions and
        mobility_score >= self.branch_threshold and
        follower_count >= self.min_followers_for_branch
    )
```

### 2. 条件パラメータ

- **branch_enabled**: 分岐機能の有効化
- **min_directions**: 最小方向数（デフォルト: 2）
- **branch_threshold**: 分岐閾値（デフォルト: 0.3）
- **min_followers_for_branch**: 分岐に必要な最小フォロワー数（デフォルト: 3）

### 3. 分岐検証

```python
def validate_branch_result(self, new_follower_count, remaining_follower_count):
    """Validate that both swarms have sufficient followers"""
    return new_follower_count >= 3 and remaining_follower_count >= 3
```

## 実装詳細

### 1. MobilityBasedBranchAlgorithm

#### フォロワー選択

```python
def select_branch_followers(self, source_swarm, valid_directions, mobility_scores):
    """Select followers based on mobility scores"""
    followers = source_swarm.followers
    if len(followers) < 6:
        # 最低6人のフォロワーが必要
        return []

    # 最低3人を新しい群に割り当て
    split_count = max(3, len(followers) // 2)

    # mobility_scoreに基づいてソート
    follower_scores = []
    for i, follower in enumerate(followers):
        score = mobility_scores[i] if i < len(mobility_scores) else 0.5
        follower_scores.append((follower, score))

    # スコアの高い順にソート
    follower_scores.sort(key=lambda x: x[1], reverse=True)

    # 上位のフォロワーを新しい群に割り当て
    selected_followers = [follower for follower, _ in follower_scores[:split_count]]

    return selected_followers
```

#### リーダー選択

```python
def select_branch_leader(self, source_swarm, valid_directions, mobility_scores):
    """Select leader with highest mobility score"""
    if not source_swarm.followers:
        return None

    # mobility_scoreが最も高いfollowerを新しいleaderとして選択
    best_follower = None
    best_score = -1.0

    for i, follower in enumerate(source_swarm.followers):
        if i < len(mobility_scores):
            score = mobility_scores[i]
        else:
            score = 0.5  # デフォルトスコア

        if score > best_score:
            best_score = score
            best_follower = follower

    return best_follower
```

### 2. RandomBranchAlgorithm

#### フォロワー選択

```python
def select_branch_followers(self, source_swarm, valid_directions, mobility_scores):
    """Select followers randomly"""
    followers = source_swarm.followers
    if len(followers) < 6:
        return []

    # 最低3人を新しい群に割り当て
    split_count = max(3, len(followers) // 2)

    # ランダムにフォロワーを選択
    selected_followers = np.random.choice(
        followers,
        size=split_count,
        replace=False
    ).tolist()

    return selected_followers
```

#### リーダー選択

```python
def select_branch_leader(self, source_swarm, valid_directions, mobility_scores):
    """Select leader randomly"""
    if not source_swarm.followers:
        return None

    # ランダムにfollowerを新しいleaderとして選択
    return np.random.choice(source_swarm.followers)
```

## 学習情報の引き継ぎ

### 1. 学習パラメータの継承

```python
def inherit_learning_info(self, source_swarm, new_swarm):
    """Inherit learning information from source swarm"""
    if hasattr(source_swarm, 'learning_params'):
        new_swarm.learning_params = source_swarm.learning_params.copy()
```

### 2. 経験の共有

```python
def share_experience(self, source_swarm, new_swarm):
    """Share experience between swarms"""
    if hasattr(source_swarm, 'experience_buffer'):
        # 経験バッファの一部を新しい群に移譲
        shared_experience = source_swarm.experience_buffer.sample(
            size=min(100, len(source_swarm.experience_buffer))
        )
        new_swarm.experience_buffer.extend(shared_experience)
```

## 分岐プロセス

### 1. 分岐実行

```python
def execute_branch(self, source_swarm_id, valid_directions, mobility_scores):
    """Execute branching process"""

    # 1. 分岐条件のチェック
    if not self.check_branch_condition(...):
        return None

    # 2. 新しい群IDの取得
    new_swarm_id = self.env.get_next_swarm_id()

    # 3. フォロワーの選択
    selected_followers = self.branch_algorithm.select_branch_followers(...)

    # 4. リーダーの選択
    new_leader = self.branch_algorithm.select_branch_leader(...)

    # 5. 新しい群の作成
    new_swarm = self.create_new_swarm(new_leader, selected_followers)

    # 6. 学習情報の引き継ぎ
    self.inherit_learning_info(source_swarm, new_swarm)

    # 7. 分岐結果の検証
    if not self.validate_branch_result(...):
        self.cancel_branch()
        return None

    return new_swarm
```

### 2. 分岐後の処理

```python
def post_branch_processing(self, source_swarm, new_swarm):
    """Post-branching processing"""

    # 1. フォロワーの再配置
    for follower in new_swarm.followers:
        follower.swarm_id = new_swarm.swarm_id
        follower.agent_coordinate = new_swarm.leader.coordinate

    # 2. 新しい群の初期化
    new_swarm.leader.set_role(RobotRole.LEADER)
    new_swarm.leader.swarm_id = new_swarm.swarm_id

    # 3. 初期行動の設定
    initial_theta = self.set_initial_action_for_new_swarm(new_swarm)

    # 4. システムエージェントの更新
    self.system_agent.add_swarm_agent(new_swarm.swarm_id)
```

## 設定パラメータ

### 1. SystemAgentParam

```python
@dataclass
class BranchConditionParam:
    branch_enabled: bool = True
    branch_threshold: float = 0.3
    min_directions: int = 2
    min_followers_for_branch: int = 3
    branch_learning_inheritance: bool = True
    branch_leader_selection_method: str = "highest_score"
    branch_follower_selection_method: str = "random"
    swarm_creation_cooldown: float = 5.0
    next_swarm_id: int = 2
    branch_algorithm: str = "mobility_based"
```

### 2. アルゴリズム選択

- **"mobility_based"**: MobilityBasedBranchAlgorithm
- **"random"**: RandomBranchAlgorithm

## 使用例

### 1. 基本的な使用

```python
# 分岐アルゴリズムの作成
branch_algorithm = MobilityBasedBranchAlgorithm()

# 分岐条件のチェック
if system_agent.check_branch(swarm_state):
    # 分岐の実行
    new_swarm = system_agent.execute_branch(
        source_swarm_id=1,
        valid_directions=valid_dirs,
        mobility_scores=mobility_scores
    )
```

### 2. カスタム設定

```python
# カスタム分岐条件
branch_condition = BranchConditionParam(
    branch_threshold=0.5,  # より高い閾値
    min_followers_for_branch=4,  # より多くのフォロワーが必要
    branch_algorithm="mobility_based"
)

system_agent.branch_condition = branch_condition
```

## パフォーマンス最適化

### 1. 分岐頻度の制御

- クールダウン期間の設定
- 分岐条件の動的調整
- システム負荷の監視

### 2. メモリ効率

- 不要なデータの削除
- 学習情報の効率的な継承
- リソース使用量の最適化

### 3. 計算効率

- 分岐条件の早期チェック
- 並列処理の活用
- キャッシュの活用

## 今後の拡張

### 1. 新しい分岐戦略

- パフォーマンスベース分岐
- 適応的分岐閾値
- マルチオブジェクト分岐

### 2. 学習統合の強化

- より高度な知識移転
- メタ学習の導入
- 分散学習の実装

### 3. 安全性の向上

- より厳密な分岐検証
- ロールバック機能
- 異常検知システム
