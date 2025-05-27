import tensorflow as tf
from agents.a2c import A2CAgent
from envs.env import Env
from models.actor_critic import ParamActorCritic
from utils.utils import flatten_state
import datetime
import os

def main():
    # 環境とモデルの初期化
    env = Env()
    input_dim = flatten_state(env.reset()).shape[0]
    model = ParamActorCritic(input_dim=input_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # エージェントの初期化
    agent = A2CAgent(
      env=env,
      model=model,
      optimizer=optimizer,
      gamma=0.99,
      n_steps=5,
      max_steps_per_episode=10
    )

    # 出力ディレクトリの作成
    timestamp_root = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./gifs/{timestamp_root}"
    os.makedirs(output_dir, exist_ok=True)

    # トレーニング実行 + 各エピソードごとにGIF保存
    episodes = 10
    for ep in range(1, episodes + 1):
        total_reward = agent.train_one_episode()
        gif_name = f"{timestamp_root}"
        env.save_gif(episode=ep, date_time=gif_name)
        print(f"Episode {ep}: reward = {total_reward}, GIF saved as {gif_name}.gif")

    print("Training finished and all GIFs saved.")

if __name__ == "__main__":
    main()