import tensorflow as tf
import numpy as np

SCALE_MAX = 20.0

class ModelActorCritic(tf.keras.Model):
  """
  Actor-Critic model with shared layers for policy and value function.
  This model outputs a probability distribution for actions and a value estimate.
  The actions are sampled from a Gaussian distribution parameterized by the output of the model.
  The model also computes the log probabilities of the actions for policy gradient updates.
  """
  def __init__(self, input_dim):
    """
    Initializes the modelActorCritic model.
    Args:
      input_dim (int): The dimension of the input state.
    """
    super(ModelActorCritic, self).__init__()
    super().__init__()
    self.shared_dense1 = tf.keras.layers.Dense(128, activation='relu')
    self.shared_dense2 = tf.keras.layers.Dense(64, activation='relu')

    self.mu_output  = tf.keras.layers.Dense(3)
    self.log_std    = tf.Variable(initial_value=tf.zeros(3), trainable=True)

    self.critic_output = tf.keras.layers.Dense(1)

  def call(self, state):
    x = self.shared_dense1(state)
    x = self.shared_dense2(x)

    raw_mu = self.mu_output(x)

    # thだけ0-1、k_e, k_cは0-SCALE_MAX
    th     = tf.nn.sigmoid(raw_mu[:, 0:1])               # shape = (batch, 1)
    k_e_c  = tf.nn.sigmoid(raw_mu[:, 1:]) * SCALE_MAX    # shape = (batch, 2)

    mu = tf.concat([th, k_e_c], axis=-1)  # shape = (batch, 3)
    std = tf.exp(self.log_std)

    value = self.critic_output(x)
    return mu, std, value


  @staticmethod
  def sample_action(mu, std):
    """
    Samples an action from the Gaussian distribution defined by mu and std.
    Args:
      mu (tf.Tensor): Mean of the action distribution.
      std (tf.Tensor): Standard deviation of the action distribution.
    Returns:
      action (tf.Tensor): Sampled action, clipped to [0, 1].
      epsilon (tf.Tensor): Random noise used for sampling.
    """
    epsilon = tf.random.normal(shape=mu.shape)
    action = mu + std * epsilon
    action_clipped = tf.clip_by_value(action, 0.0, SCALE_MAX)
    return action_clipped, epsilon

  @staticmethod
  def compute_log_prob(mu, std, action):
    """
    Computes the log probability of the action under the Gaussian distribution defined by mu and std.
    Args:
      mu (tf.Tensor): Mean of the action distribution.
      std (tf.Tensor): Standard deviation of the action distribution.
      action (tf.Tensor): Action taken.
    Returns:
      log_prob (tf.Tensor): Log probability of the action.
    """
    var = tf.square(std)
    log_std = tf.math.log(std + 1e-8)
    log_prob = -0.5 * tf.reduce_sum(
      ((action - mu) ** 2) / (var + 1e-8) + 2.0 * log_std + tf.math.log(2.0 * np.pi),
      axis=-1
    )
    return log_prob

  def compute_loss(self, states, actions, returns, advantages):
    """
    Computes the loss for the actor-critic model.
    Args:
      states (tf.Tensor): Input states.
      actions (tf.Tensor): Actions taken.
      returns (tf.Tensor): Returns for the actions.
      advantages (tf.Tensor): Advantages for the actions.
    Returns:
      total_loss (tf.Tensor): Total loss combining actor and critic losses.
      actor_loss (tf.Tensor): Loss for the actor.
      critic_loss (tf.Tensor): Loss for the critic.
    """
    mu, std, values = self(states)
    log_probs = self.compute_log_prob(mu, std, actions)

    actor_loss = -tf.reduce_mean(log_probs * advantages)
    critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values, axis=-1)))

    entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * tf.square(std)) + 0.5)
    entropy_bonus = 0.01 * entropy

    total_loss = actor_loss + 0.5 * critic_loss - entropy_bonus
    return total_loss, actor_loss, critic_loss

  @tf.function
  def train_step(self, optimizer, states, actions, returns, advantages):
    """
    Performs a single training step.
    Args:
      optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
      states (tf.Tensor): Input states.
      actions (tf.Tensor): Actions taken.
      returns (tf.Tensor): Returns for the actions.
      advantages (tf.Tensor): Advantages for the actions. 
    Returns:
      loss (tf.Tensor): Total loss for the training step.
      actor_loss (tf.Tensor): Loss for the actor.
      critic_loss (tf.Tensor): Loss for the critic.
    """
    with tf.GradientTape() as tape:
      loss, actor_loss, critic_loss = self.compute_loss(states, actions, returns, advantages)

    grads = tape.gradient(loss, self.trainable_variables)
    optimizer.apply_gradients(zip(grads, self.trainable_variables))
    return loss, actor_loss, critic_loss
  

  # 末尾に追加（モデルが保存/ロード可能になる）
  def get_config(self):
    return {"input_dim": 32}  # input_dim は任意。環境依存で適宜設定。

  @classmethod
  def from_config(cls, config):
    return cls(**config)