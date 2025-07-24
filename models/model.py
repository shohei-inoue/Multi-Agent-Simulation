"""
Model class for neural network architectures.
Provides model creation and management for learning agents.
"""

from models.actor_critic import ModelActorCritic
from models.swarm_actor_critic import SwarmActorCritic
from models.system_actor_critic import SystemActorCritic
from enum import Enum
import tensorflow as tf
import json
from pathlib import Path
from datetime import datetime

from core.interfaces import Configurable, Stateful, Loggable


class ModelType(str, Enum):
    """Supported model types"""
    ACTOR_CRITIC = "actor-critic"
    SWARM_ACTOR_CRITIC = "swarm-actor-critic"
    SYSTEM_ACTOR_CRITIC = "system-actor-critic"


class Model(Configurable, Stateful, Loggable):
    def __init__(self, model_name: str, input_dim: int = 32, **kwargs):
        self.model_name = model_name
        self.model = self.__init__model(model_name, input_dim, **kwargs)

    def __init__model(self, model_name: str, input_dim: int, **kwargs):
        if model_name == ModelType.ACTOR_CRITIC:
            return ModelActorCritic(input_dim)
        elif model_name == ModelType.SWARM_ACTOR_CRITIC:
            return SwarmActorCritic(input_dim)
        elif model_name == ModelType.SYSTEM_ACTOR_CRITIC:
            max_swarms = kwargs.get('max_swarms', 10)
            return SystemActorCritic(input_dim, max_swarms)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def __call__(self, *args, **kwargs):
        # モデル呼び出し（forward pass）をラップ
        return self.model(*args, **kwargs)

    def sample_action(self, *args, **kwargs):
        return self.model.sample_action(*args, **kwargs)

    def compute_log_prob(self, *args, **kwargs):
        return self.model.compute_log_prob(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.model.train_step(*args, **kwargs)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)
    
    def get_branch_threshold(self):
        """分岐閾値を取得（SystemActorCriticのみ）"""
        if hasattr(self.model, 'get_branch_threshold'):
            return self.model.get_branch_threshold()
        return None
    
    def get_integration_threshold(self):
        """統合閾値を取得（SystemActorCriticのみ）"""
        if hasattr(self.model, 'get_integration_threshold'):
            return self.model.get_integration_threshold()
        return None
    
    def set_branch_threshold(self, value):
        """分岐閾値を設定（SystemActorCriticのみ）"""
        if hasattr(self.model, 'set_branch_threshold'):
            self.model.set_branch_threshold(value)
    
    def set_integration_threshold(self, value):
        """統合閾値を設定（SystemActorCriticのみ）"""
        if hasattr(self.model, 'set_integration_threshold'):
            self.model.set_integration_threshold(value)
    
    # Loggableインターフェースの実装
    def get_log_data(self):
        """ログデータを取得"""
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__
        }
    
    def save_log(self, log_dir: str):
        """ログを保存"""
        # モデルのログ保存は必要に応じて実装
        pass
    
    # Configurableインターフェースの実装
    def get_config(self):
        """設定を取得"""
        return {
            "model_name": self.model_name,
            "input_dim": getattr(self.model, 'input_dim', None)
        }
    
    def load_config(self, config: dict):
        """設定を読み込み"""
        # 設定の読み込みは必要に応じて実装
        pass
    
    # Statefulインターフェースの実装
    def get_state(self):
        """状態を取得"""
        return {
            "model_weights": self.model.get_weights() if hasattr(self.model, 'get_weights') else None
        }
    
    def set_state(self, state: dict):
        """状態を設定"""
        if state.get("model_weights") and hasattr(self.model, 'set_weights'):
            self.model.set_weights(state["model_weights"])
    
    def reset_state(self):
        """状態をリセット"""
        # モデルの状態リセットは必要に応じて実装
        pass
    
    def set_config(self, config: dict):
        """設定を設定"""
        # 設定の設定は必要に応じて実装
        pass

    def save_model(self, save_dir: str, model_name: str = "model"):
        """モデルを保存"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(save_dir) / "models"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Kerasモデルとして保存
        model_file = save_path / f"{model_name}.h5"
        self.model.save_weights(str(model_file))
        
        # モデル設定を保存
        config_file = save_path / f"{model_name}_config.json"
        config = {
            'model_type': self.model_name,
            'input_dim': getattr(self.model, 'input_dim', None),
            'hidden_dims': getattr(self.model, 'hidden_dims', None),
            'learning_rate': getattr(self.model, 'learning_rate', None),
            'model_architecture': self.model.get_config() if hasattr(self.model, 'get_config') else None
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"Model saved to {model_file}")
        return str(model_file)
    
    def load_model(self, load_dir: str, model_name: str = "model"):
        """モデルを読み込み"""
        load_path = Path(load_dir) / "models"
        
        # モデルファイルを確認
        model_file = load_path / f"{model_name}.h5"
        config_file = load_path / f"{model_name}_config.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # 設定を読み込み
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # モデルを作成（設定に基づいて）
        if config['model_type'] == ModelType.SWARM_ACTOR_CRITIC.value:
            self.model = SwarmActorCritic(
                input_dim=config['input_dim']
            )
        elif config['model_type'] == ModelType.SYSTEM_ACTOR_CRITIC.value:
            self.model = SystemActorCritic(
                input_dim=config['input_dim'],
                max_swarms=config.get('max_swarms', 10)
            )
        else:
            raise ValueError(f"Unsupported model type: {config['model_type']}")
        
        # 重みを読み込み
        self.model.load_weights(str(model_file))
        
        print(f"Model loaded from {model_file}")
        return self.model
    
    def save_checkpoint(self, save_dir: str, episode: int, model_name: str = "model"):
        """チェックポイントを保存"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(save_dir) / "models" / "checkpoints"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # エピソード番号付きで保存
        checkpoint_file = save_path / f"{model_name}_episode_{episode:04d}.h5"
        self.model.save_weights(str(checkpoint_file))
        
        # チェックポイント情報を保存
        checkpoint_info = {
            'episode': episode,
            'model_type': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'model_file': str(checkpoint_file)
        }
        
        info_file = save_path / f"{model_name}_episode_{episode:04d}_info.json"
        with open(info_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_file}")
        return str(checkpoint_file)
    
    def load_latest_checkpoint(self, load_dir: str, model_name: str = "model"):
        """最新のチェックポイントを読み込み"""
        load_path = Path(load_dir) / "models" / "checkpoints"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {load_path}")
        
        # チェックポイントファイルを検索
        checkpoint_files = list(load_path.glob(f"{model_name}_episode_*.h5"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {load_path}")
        
        # 最新のチェックポイントを取得
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        episode = int(latest_checkpoint.stem.split('_')[-1])
        
        # 設定ファイルを確認
        config_file = Path(load_dir) / "models" / f"{model_name}_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # 設定を読み込み
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # モデルを作成
        if config['model_type'] == ModelType.SWARM_ACTOR_CRITIC.value:
            self.model = SwarmActorCritic(
                input_dim=config['input_dim']
            )
        elif config['model_type'] == ModelType.SYSTEM_ACTOR_CRITIC.value:
            self.model = SystemActorCritic(
                input_dim=config['input_dim'],
                max_swarms=config.get('max_swarms', 10)
            )
        else:
            raise ValueError(f"Unsupported model type: {config['model_type']}")
        
        # 重みを読み込み
        self.model.load_weights(str(latest_checkpoint))
        
        print(f"Latest checkpoint loaded from {latest_checkpoint} (episode {episode})")
        return self.model, episode


def create_swarm_model(input_dim: int) -> Model:
    """SwarmAgent用のモデルを作成"""
    return Model(ModelType.SWARM_ACTOR_CRITIC, input_dim)


def create_system_model(input_dim: int, max_swarms: int = 10) -> Model:
    """SystemAgent用のモデルを作成"""
    return Model(ModelType.SYSTEM_ACTOR_CRITIC, input_dim, max_swarms=max_swarms)


def create_legacy_model(input_dim: int) -> Model:
    """従来のActor-Criticモデルを作成（後方互換性のため）"""
    return Model(ModelType.ACTOR_CRITIC, input_dim)