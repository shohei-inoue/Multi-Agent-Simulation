from models.actor_critic import ModelActorCritic
from enum import Enum
import tensorflow as tf

class ModelType(str, Enum):
    ACTOR_CRITIC = "actor-critic"

class Model:
    def __init__(self, model_name: str, input_dim: int = 32):
        self.model_name = model_name
        self.model = self.__init__model(model_name, input_dim)

    def __init__model(self, model_name: str, input_dim: int):
        if model_name == ModelType.ACTOR_CRITIC:
            return ModelActorCritic(input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def __call__(self, *args, **kwargs):
        # モデル呼び出し（forward pass）をラップ
        return self.model(*args, **kwargs)

    def sample_action(self, mu, std):
        return self.model.sample_action(mu, std)

    def compute_log_prob(self, mu, std, action):
        return self.model.compute_log_prob(mu, std, action)

    def train_step(self, *args, **kwargs):
        return self.model.train_step(*args, **kwargs)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)