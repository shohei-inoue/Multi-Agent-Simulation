#!/usr/bin/env python3
"""
探査環境のマップ画像生成スクリプト
現在の設定でマップと障害物の配置を可視化
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """環境設定"""
    from params.simulation import SimulationParam
    
    sim_param = SimulationParam()
    
    # 環境設定
    sim_param.environment.map.width = 200
    sim_param.environment.map.height = 100
    sim_param.environment.obstacle.probability = 0.0  # デフォルト
    
    # 探査設定
    sim_param.explore.robotNum = 20
    sim_param.explore.coordinate.x = 10.0
    sim_param.explore.coordinate.y = 10.0
    sim_param.explore.boundary.inner = 0.0
    sim_param.explore.boundary.outer = 20.0
    
    return sim_param

def generate_map_image(obstacle_density=0.0, output_dir="map_images"):
    """マップ画像生成"""
    print(f"=== マップ画像生成開始 (障害物密度: {obstacle_density}) ===")
    
    try:
        # 1. 環境設定
        print("1. 環境設定中...")
        sim_param = setup_environment()
        sim_param.environment.obstacle.probability = obstacle_density
        print("✓ 環境設定完了")
        
        # 2. 環境作成
        print("2. 環境作成中...")
        from envs.env import Env
        env = Env(sim_param)
        print("✓ 環境作成完了")
        
        # 3. マップ情報取得
        print("3. マップ情報取得中...")
        # 環境のマップデータにアクセス
        map_data = env._Env__map  # プライベート変数にアクセス
        obstacle_value = env._Env__obstacle_value
        print("✓ マップ情報取得完了")
        
        # 4. 画像生成
        print("4. 画像生成中...")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # マップサイズ
        map_width = sim_param.environment.map.width
        map_height = sim_param.environment.map.height
        
        # 背景（白）
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)
        ax.set_aspect('equal')
        
        # 軸のメモリとラベルを非表示
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # マップ境界
        border = patches.Rectangle((0, 0), map_width, map_height, 
                                 linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(border)
        
        # 障害物の描画（黒色）
        obstacle_positions = np.where(map_data == obstacle_value)
        for y, x in zip(obstacle_positions[0], obstacle_positions[1]):
            circle = patches.Circle((x, y), radius=1, 
                                  color='black', alpha=0.7)
            ax.add_patch(circle)
        
        # 初期位置の描画（青色）
        init_x = sim_param.explore.coordinate.x
        init_y = sim_param.explore.coordinate.y
        
        # 開始位置（中心）
        start_point = patches.Circle((init_x, init_y), radius=0.8, 
                                   color='blue', alpha=0.8)
        ax.add_patch(start_point)
        
        # Follower20台の初期位置を描画
        robot_num = sim_param.explore.robotNum
        offset_position = 5.0  # デフォルトのオフセット位置
        
        for index in range(robot_num):
            # 円形配置の計算
            angle = 2 * np.pi * index / robot_num
            follower_x = init_x + offset_position * np.cos(angle)
            follower_y = init_y + offset_position * np.sin(angle)
            
            # Followerの位置を小さな円で描画
            follower_point = patches.Circle((follower_x, follower_y), radius=0.5, 
                                          color='red', alpha=0.6)
            ax.add_patch(follower_point)
        
        # 探査境界の描画
        inner_boundary = patches.Circle((init_x, init_y), 
                                      radius=sim_param.explore.boundary.inner,
                                      linewidth=2, edgecolor='blue', 
                                      facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(inner_boundary)
        
        outer_boundary = patches.Circle((init_x, init_y), 
                                      radius=sim_param.explore.boundary.outer,
                                      linewidth=2, edgecolor='blue', 
                                      facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(outer_boundary)
        
        # グリッド線
        ax.grid(True, alpha=0.3)
        

        
        # 保存
        filename = f"map_obstacle_{obstacle_density}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ 画像保存完了: {filepath}")
        
        # 表示
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_all_map_images():
    """全ての障害物密度でマップ画像生成"""
    print("=== 全マップ画像生成開始 ===")
    
    obstacle_densities = [0.0, 0.003, 0.005]
    
    for density in obstacle_densities:
        print(f"\n--- 障害物密度 {density} のマップ生成 ---")
        success = generate_map_image(density)
        if not success:
            print(f"❌ 障害物密度 {density} のマップ生成に失敗")
            break
    
    print("\n=== 全マップ画像生成完了 ===")

if __name__ == "__main__":
    print("=== マップ画像生成開始 ===")
    print(f"開始時刻: {datetime.now()}")
    
    # 単一マップ生成（障害物なし）
    # success = generate_map_image(0.0)
    
    # 全マップ生成
    generate_all_map_images()
    
    print(f"\n終了時刻: {datetime.now()}")
    print("🎉 マップ画像生成が完了しました！") 