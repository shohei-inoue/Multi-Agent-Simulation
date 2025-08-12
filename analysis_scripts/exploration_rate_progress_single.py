#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploration Rate Progress Single Graph Script
1エピソード目のExploration Rate Progressグラフを環境別に単体で表示

使用方法:
    python exploration_rate_progress_single.py

出力:
    - exploration_rate_progress_single/ ディレクトリに結果を保存
    - 各環境（障害物密度）別のExploration Rate Progressグラフ
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

class ExplorationRateProgressAnalyzer:
    def __init__(self, data_dir: str = "verify_configs/verification_results"):
        """
        初期化
        
        Args:
            data_dir: 検証結果が保存されているディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.step_data = pd.DataFrame()
        
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
        1エピソード目のステップデータを読み込み
        
        Returns:
            bool: データ読み込み成功可否
        """
        print("=== 1エピソード目ステップデータ読み込み中 ===")
        
        if not self.data_dir.exists():
            print(f"❌ データディレクトリが存在しません: {self.data_dir}")
            return False
        
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
                                step_records.append(step_info)
                    
                    print(f"    ✓ {json_file.name} 処理完了")
                    
                except Exception as e:
                    print(f"    ❌ {json_file.name} 読み込みエラー: {e}")
                    continue
        
        # DataFrameに変換
        if step_records:
            self.step_data = pd.DataFrame(step_records)
            print(f"✓ ステップデータ: {len(self.step_data)} 件")
        else:
            print("❌ ステップデータが見つかりませんでした")
        
        return len(step_records) > 0
    
    def generate_exploration_rate_progress_graphs(self, output_dir: str = "exploration_rate_progress_single"):
        """
        各環境別のExploration Rate Progressグラフを単体で生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== Exploration Rate Progress単体グラフ生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.step_data.empty:
            print("❌ ステップデータが不足しています")
            return
        
        # 障害物密度別に分析
        densities = sorted(self.step_data['obstacle_density'].unique())
        
        for density in densities:
            print(f"  📊 障害物密度 {density} のグラフ生成中...")
            
            density_data = self.step_data[self.step_data['obstacle_density'] == density]
            
            # 大きなサイズで単体グラフを作成
            plt.figure(figsize=(14, 10))
            
            # 各Configの探査率変化をプロット
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                
                # 各ステップでの平均と標準偏差を計算
                step_stats = config_steps.groupby('step_id')['exploration_rate'].agg(['mean', 'std', 'count']).reset_index()
                
                # メインライン
                plt.plot(step_stats['step_id'], step_stats['mean'],
                        color=self.config_colors[config_type], 
                        linewidth=3, 
                        marker='o', 
                        markersize=6,
                        label=f'Config {config_type}: {self.config_descriptions[config_type]}',
                        alpha=0.9)
                
                # 標準偏差を影で表示
                plt.fill_between(step_stats['step_id'],
                               step_stats['mean'] - step_stats['std'],
                               step_stats['mean'] + step_stats['std'],
                               color=self.config_colors[config_type], 
                               alpha=0.2,
                               label=f'Config {config_type} ±1σ')
            
            # グラフの詳細設定
            plt.title(f'Exploration Rate Progress\n(Obstacle Density: {density})', 
                     fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Step', fontsize=14, fontweight='bold')
            plt.ylabel('Exploration Rate', fontsize=14, fontweight='bold')
            
            # 凡例の設定
            handles, labels = plt.gca().get_legend_handles_labels()
            # 標準偏差のラベルを除外してメインラインのみ表示
            main_handles = [h for i, h in enumerate(handles) if '±1σ' not in labels[i]]
            main_labels = [l for l in labels if '±1σ' not in l]
            
            plt.legend(main_handles, main_labels, 
                      loc='upper left', 
                      fontsize=12,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      framealpha=0.9)
            
            # グリッドの設定
            plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')
            
            # 軸の設定
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            
            # 軸の目盛りを調整
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            # Y軸を探査率の範囲に合わせて調整
            max_exploration = density_data['exploration_rate'].max()
            plt.ylim(0, min(1.0, max_exploration * 1.1))
            
            # 背景色を設定
            plt.gca().set_facecolor('#f8f9fa')
            
            # 密度情報のテキストボックスを追加
            density_text = f"Environment: Obstacle Density = {density}"
            if density == 0.0:
                density_text += " (No obstacles)"
            elif density == 0.003:
                density_text += " (Low obstacle density)"
            elif density == 0.005:
                density_text += " (Medium obstacle density)"
            
            plt.text(0.02, 0.98, density_text, 
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # レイアウト調整
            plt.tight_layout()
            
            # ファイル名を設定
            density_str = f"{density:.3f}".replace('.', '_')
            filename = f"exploration_rate_progress_density_{density_str}.png"
            
            # 高解像度で保存
            plt.savefig(f"{output_dir}/{filename}", 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
            plt.close()
            
            print(f"    ✓ {filename} を保存しました")
        
        # 統計サマリーも生成
        self.generate_summary_statistics(output_dir)
        
        print(f"✓ 全ての単体グラフを {output_dir}/ に保存しました")
    
    def generate_summary_statistics(self, output_dir: str):
        """
        各環境の統計サマリーを生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        summary_data = []
        
        densities = sorted(self.step_data['obstacle_density'].unique())
        
        for density in densities:
            density_data = self.step_data[self.step_data['obstacle_density'] == density]
            
            for config_type in sorted(density_data['config_type'].unique()):
                config_steps = density_data[density_data['config_type'] == config_type]
                
                # 基本統計
                exploration_stats = config_steps['exploration_rate'].describe()
                
                # 最終探査率
                final_exploration = config_steps.groupby('step_id')['exploration_rate'].mean().iloc[-1] if len(config_steps) > 0 else 0
                
                # 探査率上昇速度
                step_means = config_steps.groupby('step_id')['exploration_rate'].mean()
                if len(step_means) > 1:
                    exploration_diff = np.diff(step_means.values)
                    avg_increase_rate = np.mean(exploration_diff)
                    max_increase_rate = np.max(exploration_diff)
                else:
                    avg_increase_rate = 0
                    max_increase_rate = 0
                
                summary_data.append({
                    'obstacle_density': density,
                    'config_type': config_type,
                    'config_description': self.config_descriptions[config_type],
                    'final_exploration_rate': final_exploration,
                    'avg_increase_rate': avg_increase_rate,
                    'max_increase_rate': max_increase_rate,
                    'exploration_mean': exploration_stats['mean'],
                    'exploration_std': exploration_stats['std'],
                    'exploration_min': exploration_stats['min'],
                    'exploration_max': exploration_stats['max'],
                    'total_steps': len(step_means)
                })
        
        # CSVで保存
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f"{output_dir}/exploration_progress_summary.csv", index=False)
            print(f"✓ 統計サマリーを {output_dir}/exploration_progress_summary.csv に保存しました")
    
    def generate_comparison_table(self, output_dir: str = "exploration_rate_progress_single"):
        """
        環境別比較表を生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 環境別比較表生成中 ===")
        
        # 比較表用のデータを準備
        densities = sorted(self.step_data['obstacle_density'].unique())
        configs = sorted(self.step_data['config_type'].unique())
        
        comparison_data = []
        
        for config in configs:
            row_data = {'Config': config, 'Description': self.config_descriptions[config]}
            
            for density in densities:
                density_config_data = self.step_data[
                    (self.step_data['obstacle_density'] == density) & 
                    (self.step_data['config_type'] == config)
                ]
                
                if len(density_config_data) > 0:
                    final_rate = density_config_data.groupby('step_id')['exploration_rate'].mean().iloc[-1]
                    row_data[f'Density_{density}'] = f"{final_rate:.4f}"
                else:
                    row_data[f'Density_{density}'] = "N/A"
            
            comparison_data.append(row_data)
        
        # 比較表をCSVで保存
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f"{output_dir}/environment_comparison_table.csv", index=False)
            print(f"✓ 環境別比較表を {output_dir}/environment_comparison_table.csv に保存しました")
    
    def run_analysis(self):
        """
        分析を実行
        """
        print("🚀 Exploration Rate Progress単体グラフ分析を開始します\n")
        
        # データ読み込み
        if not self.load_first_episode_data():
            print("❌ データ読み込みに失敗しました")
            return False
        
        output_dir = "exploration_rate_progress_single"
        
        # 分析実行
        self.generate_exploration_rate_progress_graphs(output_dir)
        self.generate_comparison_table(output_dir)
        
        print(f"\n🎉 Exploration Rate Progress単体グラフ分析完了！")
        print(f"📁 結果は {output_dir}/ ディレクトリに保存されました")
        
        return True


def main():
    """メイン関数"""
    analyzer = ExplorationRateProgressAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 