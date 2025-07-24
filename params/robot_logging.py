from dataclasses import dataclass, asdict

@dataclass
class RobotLoggingConfig:
    """
    ロボットデータ保存の設定
    """
    # 基本設定
    save_robot_data: bool = False  # ロボットデータを保存するか
    save_episode_summary: bool = True  # エピソードサマリーは常に保存
    
    # サンプリング設定
    sampling_rate: float = 0.1  # 保存するステップの割合（0.1 = 10%）
    save_collision_only: bool = True  # 衝突時のみ保存
    
    # 保存するデータの種類
    save_position: bool = True  # 位置情報
    save_collision: bool = True  # 衝突情報
    save_boids: bool = True  # boids情報
    save_distance: bool = True  # エージェントとの距離
    
    # ファイル設定
    separate_files: bool = False  # ロボットごとに別ファイルにするか
    compress_data: bool = False  # データを圧縮するか
    
    # メモリ管理
    max_robot_records: int = 10000  # 最大記録数（メモリ保護）

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def copy(self):
        return RobotLoggingConfig(**asdict(self)) 