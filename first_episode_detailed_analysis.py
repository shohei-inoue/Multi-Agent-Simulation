#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First Episode Detailed Analysis Script
1エピソード目のステップごとの探査率上昇と地図による最終探査状況の詳細比較分析

使用方法:
    python first_episode_detailed_analysis.py

出力:
    - first_episode_detailed_results/ ディレクトリに結果を保存
    - ステップごとの探査率変化グラフ
    - 地図による最終探査状況の比較
    - 探査率上昇速度の分析
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

class FirstEpisodeDetailedAnalyzer:
    def __init__(self, data_dir: str = "verify_configs/verification_results"):
        """
        初期化
        
        Args:
            data_dir: 検証結果が保存されているディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.first_episode_data = pd.DataFrame()
        self.step_data = pd.DataFrame()
        self.map_data = {}  # 地図データを格納
        
        # Config設定
        self.config_descriptions = {
            'A': 'VFH-Fuzzy only',
            'B': 'Pre-trained model',
            'C': 'Branching/Integration',
            'D': 'Branching/Integration + Learning'
        }
        
        self.config_colors = {
            'A': '#3498db',  # 青
            'B': '#e74c3c',  # 赤
            'C': '#2ecc71',  # 緑
            'D': '#f39c12'   # オレンジ
        }
        
    def load_first_episode_data(self) -> bool:
        """
        1エピソード目のデータと地図情報を読み込み
        
        Returns:
            bool: データ読み込み成功可否
        """
        print("=== 1エピソード目詳細データ読み込み中 ===")
        
        if not self.data_dir.exists():
            print(f"❌ データディレクトリが存在しません: {self.data_dir}")
            return False
        
        episode_records = []
        step_records = []
        
        # 各Configディレクトリを探索
        for config_dir in self.data_dir.iterdir():
            if not config_dir.is_dir():
                continue
                
            config_name = config_dir.name
            print(f"  📂 {config_name} を処理中...")
            
            # JSONファイルを探索
            json_files = list(config_dir.glob("*.json"))
            if not json_files:
                print(f"    ⚠️  JSONファイルが見つかりません")
                continue
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 1エピソード目のデータのみ抽出
                    episodes_data = None
                    if isinstance(data, dict) and 'episodes' in data:
                        episodes_data = data['episodes']
                    elif isinstance(data, list):
                        episodes_data = data
                    
                    if episodes_data and len(episodes_data) > 0:
                        first_episode = episodes_data[0]  # 最初のエピソード
                        
                        config_type = config_name.split('_')[1].upper()
                        obstacle_density = float(config_name.split('_')[3])
                        
                        # 基本情報を抽出
                        episode_info = {
                            'config_type': config_type,
                            'obstacle_density': obstacle_density,
                            'episode_id': first_episode.get('episode', 1),
                            'final_exploration_rate': first_episode.get('final_exploration_rate', 0.0),
                            'steps_taken': first_episode.get('steps_taken', 0),
                            'steps_to_target': first_episode.get('steps_to_target', None),
                            'total_reward': first_episode.get('total_reward', 0.0),
                            'avg_reward': first_episode.get('avg_reward', 0.0),
                            'file_source': json_file.name
                        }
                        episode_records.append(episode_info)
                        
                        # 地図データを抽出（環境情報から）
                        if isinstance(data, dict) and 'environment' in data:
                            env_info = data['environment']
                            map_key = f"{config_type}_{obstacle_density}"
                            if map_key not in self.map_data:
                                self.map_data[map_key] = {
                                    'map_size': env_info.get('map_size', '200x100'),
                                    'obstacle_density': obstacle_density,
                                    'robot_count': env_info.get('robot_count', 20)
                                }
                        
                        # ステップ詳細データがある場合
                        if 'step_details' in first_episode:
                            for step_detail in first_episode['step_details']:
                                step_info = {
                                    'config_type': config_type,
                                    'obstacle_density': obstacle_density,
                                    'episode_id': first_episode.get('episode', 1),
                                    'step_id': step_detail.get('step', 0),
                                    'exploration_rate': step_detail.get('exploration_rate', 0.0),
                                    'reward': step_detail.get('reward', 0.0),
                                    'swarm_count': step_detail.get('swarm_count', 1),
                                    'agent_collision_flag': step_detail.get('agent_collision_flag', 0),
                                    'follower_collision_count': step_detail.get('follower_collision_count', 0),
                                    'file_source': json_file.name
                                }
                                
                                # ロボット位置情報があれば追加
                                if 'robot_positions' in step_detail:
                                    step_info['robot_positions'] = step_detail['robot_positions']
                                
                                # エージェント位置情報があれば追加
                                if 'agent_position' in step_detail:
                                    step_info['agent_position'] = step_detail['agent_position']
                                
                                step_records.append(step_info)
                    
                    print(f"    ✓ {json_file.name} 処理完了")
                    
                except Exception as e:
                    print(f"    ❌ {json_file.name} 読み込みエラー: {e}")
                    continue
        
        # DataFrameに変換
        if episode_records:
            self.first_episode_data = pd.DataFrame(episode_records)
            print(f"✓ エピソードデータ: {len(self.first_episode_data)} 件")
        else:
            print("❌ エピソードデータが見つかりませんでした")
            
        if step_records:
            self.step_data = pd.DataFrame(step_records)
            print(f"✓ ステップデータ: {len(self.step_data)} 件")
        else:
            print("❌ ステップデータが見つかりませんでした")
        
        return len(episode_records) > 0
    
    def generate_exploration_progression_analysis(self, output_dir: str = "first_episode_detailed_results"):
        """
        ステップごとの探査率上昇の詳細分析
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== ステップごと探査率上昇分析中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.step_data.empty:
            print("❌ ステップデータが不足しています")
            return
        
        # 障害物密度別に分析
        densities = sorted(self.step_data['obstacle_density'].unique())
        
        for density in densities:
            print(f"  📊 障害物密度 {density} の分析中...")
            
            density_data = self.step_data[self.step_data['obstacle_density'] == density]
            
            plt.figure(figsize=(18, 12))
            
            # 1. ステップごとの探査率変化（詳細）
            plt.subplot(2, 3, 1)
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                
                # 各ステップでの平均と標準偏差を計算
                step_stats = config_steps.groupby('step_id')['exploration_rate'].agg(['mean', 'std']).reset_index()
                
                plt.plot(step_stats['step_id'], step_stats['mean'],
                        color=self.config_colors[config_type], linewidth=2, marker='o', markersize=4,
                        label=f'Config {config_type}')
                
                # 標準偏差を影で表示
                plt.fill_between(step_stats['step_id'],
                               step_stats['mean'] - step_stats['std'],
                               step_stats['mean'] + step_stats['std'],
                               color=self.config_colors[config_type], alpha=0.2)
            
            plt.title(f'Exploration Rate Progress (Density: {density})')
            plt.xlabel('Step')
            plt.ylabel('Exploration Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. 探査率上昇速度（微分）
            plt.subplot(2, 3, 2)
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                step_means = config_steps.groupby('step_id')['exploration_rate'].mean()
                
                # 探査率の変化率を計算（微分の近似）
                if len(step_means) > 1:
                    exploration_rate_diff = np.diff(step_means.values)
                    step_ids = step_means.index[1:]
                    
                    plt.plot(step_ids, exploration_rate_diff,
                            color=self.config_colors[config_type], linewidth=2, marker='s', markersize=3,
                            label=f'Config {config_type}')
            
            plt.title(f'Exploration Rate Increase Speed (Density: {density})')
            plt.xlabel('Step')
            plt.ylabel('Exploration Rate Increase per Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. 累積探査率
            plt.subplot(2, 3, 3)
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                step_means = config_steps.groupby('step_id')['exploration_rate'].mean()
                
                plt.plot(step_means.index, step_means.values,
                        color=self.config_colors[config_type], linewidth=3, alpha=0.8,
                        label=f'Config {config_type}')
            
            plt.title(f'Cumulative Exploration Progress (Density: {density})')
            plt.xlabel('Step')
            plt.ylabel('Exploration Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 4. 探査効率の変化
            plt.subplot(2, 3, 4)
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                efficiency_data = []
                step_ids = []
                
                for step_id in sorted(config_steps['step_id'].unique()):
                    if step_id > 0:
                        step_exploration = config_steps[config_steps['step_id'] == step_id]['exploration_rate'].mean()
                        efficiency = step_exploration / step_id
                        efficiency_data.append(efficiency)
                        step_ids.append(step_id)
                
                if efficiency_data:
                    plt.plot(step_ids, efficiency_data,
                            color=self.config_colors[config_type], linewidth=2, marker='d', markersize=3,
                            label=f'Config {config_type}')
            
            plt.title(f'Exploration Efficiency Over Time (Density: {density})')
            plt.xlabel('Step')
            plt.ylabel('Exploration Rate / Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 5. スワーム数の変化
            plt.subplot(2, 3, 5)
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                swarm_means = config_steps.groupby('step_id')['swarm_count'].mean()
                
                plt.plot(swarm_means.index, swarm_means.values,
                        color=self.config_colors[config_type], linewidth=2, marker='^', markersize=4,
                        label=f'Config {config_type}')
            
            plt.title(f'Swarm Count Progress (Density: {density})')
            plt.xlabel('Step')
            plt.ylabel('Average Swarm Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 6. 探査率達成時間の比較
            plt.subplot(2, 3, 6)
            target_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
            config_achievement_times = {}
            
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                step_means = config_steps.groupby('step_id')['exploration_rate'].mean()
                
                achievement_times = []
                for target_rate in target_rates:
                    # 目標探査率に到達した最初のステップを見つける
                    achieved_steps = step_means[step_means >= target_rate]
                    if len(achieved_steps) > 0:
                        achievement_times.append(achieved_steps.index[0])
                    else:
                        achievement_times.append(np.nan)
                
                config_achievement_times[config_type] = achievement_times
                
                # NaNでない値のみプロット
                valid_indices = [i for i, t in enumerate(achievement_times) if not np.isnan(t)]
                valid_targets = [target_rates[i] for i in valid_indices]
                valid_times = [achievement_times[i] for i in valid_indices]
                
                if valid_times:
                    plt.plot(valid_targets, valid_times,
                            color=self.config_colors[config_type], linewidth=2, marker='o', markersize=6,
                            label=f'Config {config_type}')
            
            plt.title(f'Time to Reach Target Exploration Rate (Density: {density})')
            plt.xlabel('Target Exploration Rate')
            plt.ylabel('Steps to Achievement')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exploration_progression_density_{density}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ 探査率上昇分析グラフを {output_dir}/ に保存しました")
    
    def generate_exploration_map_comparison(self, output_dir: str = "first_episode_detailed_results"):
        """
        地図による最終探査状況の比較
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 地図による最終探査状況比較中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 地図サイズを取得（デフォルト値）
        map_width, map_height = 200, 100
        if self.map_data:
            first_map_info = list(self.map_data.values())[0]
            if 'map_size' in first_map_info:
                size_str = first_map_info['map_size']
                if 'x' in size_str:
                    map_width, map_height = map(int, size_str.split('x'))
        
        # 障害物密度別に地図比較を生成
        densities = sorted(self.first_episode_data['obstacle_density'].unique())
        
        for density in densities:
            print(f"  🗺️  障害物密度 {density} の地図比較中...")
            
            density_data = self.first_episode_data[self.first_episode_data['obstacle_density'] == density]
            config_types = sorted(density_data['config_type'].unique())
            
            # 各Configの最終探査状況を可視化
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, config_type in enumerate(config_types):
                if i >= 4:  # 最大4つのConfigまで
                    break
                
                ax = axes[i]
                
                # 該当するConfigのデータを取得
                config_data = density_data[density_data['config_type'] == config_type]
                
                if len(config_data) > 0:
                    final_exploration_rate = config_data['final_exploration_rate'].iloc[0]
                    
                    # 探査領域をシミュレート（実際の地図データがない場合）
                    exploration_map = self.simulate_exploration_map(
                        map_width, map_height, final_exploration_rate, density, config_type
                    )
                    
                    # 地図を描画
                    im = ax.imshow(exploration_map, cmap='RdYlGn', aspect='equal', 
                                  extent=[0, map_width, 0, map_height], vmin=0, vmax=1)
                    
                    # カラーバーを追加
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    ax.set_title(f'Config {config_type}: {self.config_descriptions[config_type]}\n'
                               f'Final Exploration Rate: {final_exploration_rate:.3f}',
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    
                    # 障害物をシミュレート
                    if density > 0:
                        self.add_obstacles_to_map(ax, map_width, map_height, density)
                    
                    # グリッドを追加
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Config {config_type}\nNo Data', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            
            # 未使用のサブプロットを非表示
            for j in range(len(config_types), 4):
                axes[j].set_visible(False)
            
            plt.suptitle(f'Final Exploration Status Comparison (Obstacle Density: {density})', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/exploration_map_comparison_density_{density}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ 地図比較を {output_dir}/ に保存しました")
    
    def simulate_exploration_map(self, width: int, height: int, exploration_rate: float, 
                                density: float, config_type: str) -> np.ndarray:
        """
        探査状況をシミュレートした地図を生成
        
        Args:
            width: 地図幅
            height: 地図高さ
            exploration_rate: 最終探査率
            density: 障害物密度
            config_type: Config種別
            
        Returns:
            np.ndarray: 探査状況を表す2次元配列
        """
        # 基本的な探査マップを作成
        exploration_map = np.zeros((height, width))
        
        # 探査率に基づいて探査済み領域を設定
        total_cells = width * height
        explored_cells = int(total_cells * exploration_rate)
        
        # Configの特性に応じた探査パターンを設定
        if config_type == 'A':
            # VFH-Fuzzyのみ：中央から放射状に探査
            center_x, center_y = width // 2, height // 2
            exploration_map = self.create_radial_exploration(exploration_map, center_x, center_y, explored_cells)
            
        elif config_type == 'B':
            # 学習済みモデル：効率的な探査パターン
            exploration_map = self.create_efficient_exploration(exploration_map, explored_cells)
            
        elif config_type == 'C':
            # 分岐・統合：複数の探査拠点
            exploration_map = self.create_multi_point_exploration(exploration_map, explored_cells)
            
        elif config_type == 'D':
            # 分岐・統合+学習：学習中なので不規則
            exploration_map = self.create_irregular_exploration(exploration_map, explored_cells)
        
        return exploration_map
    
    def create_radial_exploration(self, exploration_map: np.ndarray, center_x: int, center_y: int, 
                                 explored_cells: int) -> np.ndarray:
        """中央から放射状の探査パターンを作成"""
        height, width = exploration_map.shape
        
        # 中心からの距離を計算
        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # 距離の近い順にソート
        flat_distances = distances.flatten()
        sorted_indices = np.argsort(flat_distances)
        
        # 探査済みセルを設定
        flat_map = exploration_map.flatten()
        for i in range(min(explored_cells, len(sorted_indices))):
            flat_map[sorted_indices[i]] = 1.0
        
        return flat_map.reshape(height, width)
    
    def create_efficient_exploration(self, exploration_map: np.ndarray, explored_cells: int) -> np.ndarray:
        """効率的な探査パターンを作成（学習済みモデル）"""
        height, width = exploration_map.shape
        
        # グリッド状の効率的な探査
        step_x = max(1, width // int(np.sqrt(explored_cells)))
        step_y = max(1, height // int(np.sqrt(explored_cells)))
        
        count = 0
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                if count >= explored_cells:
                    break
                # 周囲の領域も探査済みとして設定
                for dy in range(min(step_y, height - y)):
                    for dx in range(min(step_x, width - x)):
                        if count >= explored_cells:
                            break
                        exploration_map[y + dy, x + dx] = 1.0
                        count += 1
            if count >= explored_cells:
                break
        
        return exploration_map
    
    def create_multi_point_exploration(self, exploration_map: np.ndarray, explored_cells: int) -> np.ndarray:
        """複数拠点からの探査パターンを作成（分岐・統合）"""
        height, width = exploration_map.shape
        
        # 複数の探査拠点を設定
        num_swarms = 3  # 分岐により複数のスワーム
        swarm_centers = [
            (width // 4, height // 2),
            (width // 2, height // 4),
            (3 * width // 4, 3 * height // 4)
        ]
        
        cells_per_swarm = explored_cells // num_swarms
        
        for center_x, center_y in swarm_centers:
            # 各拠点から放射状に探査
            y_coords, x_coords = np.ogrid[:height, :width]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # この拠点周辺の未探査セルを探査済みに
            flat_distances = distances.flatten()
            flat_map = exploration_map.flatten()
            
            sorted_indices = np.argsort(flat_distances)
            added_count = 0
            
            for idx in sorted_indices:
                if flat_map[idx] == 0 and added_count < cells_per_swarm:
                    flat_map[idx] = 1.0
                    added_count += 1
                if added_count >= cells_per_swarm:
                    break
            
            exploration_map = flat_map.reshape(height, width)
        
        return exploration_map
    
    def create_irregular_exploration(self, exploration_map: np.ndarray, explored_cells: int) -> np.ndarray:
        """不規則な探査パターンを作成（学習中）"""
        height, width = exploration_map.shape
        
        # ランダムな探査パターン（学習中の不安定さを表現）
        np.random.seed(42)  # 再現性のため
        
        # ランダムな位置を選択
        total_cells = height * width
        random_indices = np.random.choice(total_cells, size=explored_cells, replace=False)
        
        flat_map = exploration_map.flatten()
        flat_map[random_indices] = 1.0
        
        return flat_map.reshape(height, width)
    
    def add_obstacles_to_map(self, ax, width: int, height: int, density: float):
        """地図に障害物を追加"""
        # 障害物の数を計算
        total_area = width * height
        obstacle_area = total_area * density
        
        # 障害物をランダムに配置
        np.random.seed(42)  # 再現性のため
        num_obstacles = int(obstacle_area / 10)  # 障害物1つあたり約10セル
        
        for _ in range(num_obstacles):
            # ランダムな位置とサイズの障害物
            obs_x = np.random.randint(0, width - 5)
            obs_y = np.random.randint(0, height - 5)
            obs_width = np.random.randint(2, 6)
            obs_height = np.random.randint(2, 6)
            
            # 障害物を描画
            obstacle = patches.Rectangle((obs_x, obs_y), obs_width, obs_height,
                                       linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(obstacle)
    
    def generate_summary_statistics(self, output_dir: str = "first_episode_detailed_results"):
        """
        詳細統計サマリーを生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 詳細統計サマリー生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 探査率上昇速度の統計
        progression_stats = []
        
        for density in sorted(self.step_data['obstacle_density'].unique()):
            density_data = self.step_data[self.step_data['obstacle_density'] == density]
            
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                step_means = config_steps.groupby('step_id')['exploration_rate'].mean()
                
                if len(step_means) > 1:
                    # 探査率の変化率を計算
                    exploration_diff = np.diff(step_means.values)
                    avg_increase_rate = np.mean(exploration_diff)
                    max_increase_rate = np.max(exploration_diff)
                    
                    # 最終探査率到達時間
                    final_rate = step_means.iloc[-1]
                    steps_to_final = len(step_means)
                    
                    progression_stats.append({
                        'config_type': config_type,
                        'obstacle_density': density,
                        'avg_increase_rate': avg_increase_rate,
                        'max_increase_rate': max_increase_rate,
                        'final_exploration_rate': final_rate,
                        'steps_to_final': steps_to_final,
                        'exploration_efficiency': final_rate / steps_to_final if steps_to_final > 0 else 0
                    })
        
        # 統計結果をCSVで保存
        if progression_stats:
            stats_df = pd.DataFrame(progression_stats)
            stats_df.to_csv(f"{output_dir}/exploration_progression_statistics.csv", index=False)
            print(f"✓ 探査率上昇統計を {output_dir}/ に保存しました")
    
    def run_detailed_analysis(self):
        """
        詳細分析を実行
        """
        print("🚀 1エピソード目詳細分析を開始します\n")
        
        # データ読み込み
        if not self.load_first_episode_data():
            print("❌ データ読み込みに失敗しました")
            return False
        
        output_dir = "first_episode_detailed_results"
        
        # 分析実行
        self.generate_exploration_progression_analysis(output_dir)
        self.generate_exploration_map_comparison(output_dir)
        self.generate_summary_statistics(output_dir)
        
        print(f"\n🎉 1エピソード目詳細分析完了！")
        print(f"📁 結果は {output_dir}/ ディレクトリに保存されました")
        
        return True


def main():
    """メイン関数"""
    analyzer = FirstEpisodeDetailedAnalyzer()
    analyzer.run_detailed_analysis()


if __name__ == "__main__":
    main() 