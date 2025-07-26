import numpy as np
import os
import uuid
import matplotlib.pyplot as plt

"""
environment map関連の処理
"""

def generate_rect_obstacle_map(
      map_width,
      map_height, 
      obstacle_prob,
      obstacle_max_size,  
      obstacle_val,
      seed,
      wall_thickness=3
  ) -> np.ndarray:
    """
    矩形障害物を配置した地図を作成
    """
    np.random.seed(seed) # シードの設定

    # 初期値
    obstacle_map = np.zeros((map_height, map_width), dtype=np.uint16)

    # 分厚い壁の生成
    # 上端の壁
    obstacle_map[0:wall_thickness, :] = obstacle_val
    # 下端の壁
    obstacle_map[map_height-wall_thickness:map_height, :] = obstacle_val
    # 左端の壁
    obstacle_map[:, 0:wall_thickness] = obstacle_val
    # 右端の壁
    obstacle_map[:, map_width-wall_thickness:map_width] = obstacle_val

    # 内部にランダムに障害物を配置（壁の内側から開始）
    for y in range(wall_thickness, map_height - wall_thickness):
      for x in range(wall_thickness, map_width - wall_thickness):
        if np.random.rand() < obstacle_prob:
          rect_h = np.random.randint(2, obstacle_max_size)
          rect_w = np.random.randint(2, obstacle_max_size)
          y2 = min(y + rect_h, map_height - wall_thickness)
          x2 = min(x + rect_w, map_width - wall_thickness)
          obstacle_map[y:y2, x:x2] = obstacle_val

    return obstacle_map


def generate_explored_map(map_width, map_height) -> np.ndarray:
    """
    探査済み地図の作成
    """
    return np.zeros((map_height, map_width), dtype=np.uint16)


def save_map_image(map_array: np.ndarray, env_width: int, env_height: int) -> str:
  """
  マップのイメージを作成
  """
  save_dir = 'public/upload'
  os.makedirs(save_dir, exist_ok=True)
  filename = f"{uuid.uuid4().hex}.png"
  filepath = os.path.join(save_dir, filename)

  plt.figure(figsize=(12, 12)) # TODO map_arrayからサイズを取得するように変更
  plt.imshow(
    map_array,
    cmap='gray_r',
    origin='lower',
    extent=[0, env_width, 0, env_height]
  )
  plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

  return filepath


