#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Exploration Comparison Script
環境ごとの探査率の伸び方を比較する分析スクリプト

使用方法:
    python environment_exploration_comparison.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnvironmentExplorationAnalyzer:
    def __init__(self, results_dir="verify_configs/verification_results"):
        self.results_dir = results_dir
        self.data = {}
        self.combined_data = pd.DataFrame()
        
    def load_verification_results(self):
        """検証結果を読み込み"""
        print("🔍 検証結果を読み込み中...")
        
        if not os.path.exists(self.results_dir):
            print(f"❌ 結果ディレクトリが見つかりません: {self.results_dir}")
            return False
            
        # 結果ディレクトリ内の全フォルダを探索
        for folder in os.listdir(self.results_dir):
            folder_path = os.path.join(self.results_dir, folder)
            if os.path.isdir(folder_path):
                # フォルダ名からConfigと障害物密度を抽出
                # 例: Config_A_obstacle_0.0
                if folder.startswith("Config_") and "obstacle_" in folder:
                    parts = folder.split("_")
                    if len(parts) >= 4:
                        config_type = parts[1]  # A, B, C, D
                        obstacle_density = float(parts[3])  # 0.0, 0.003, 0.005
                        
                        # JSONファイルを探す
                        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
                        if json_files:
                            json_path = os.path.join(folder_path, json_files[0])
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    result_data = json.load(f)
                                
                                # データを保存
                                key = f"{config_type}_{obstacle_density}"
                                self.data[key] = {
                                    'config_type': config_type,
                                    'obstacle_density': obstacle_density,
                                    'data': result_data
                                }
                                print(f"✓ {key}: {len(result_data.get('episodes', []))} エピソード")
                                
                            except Exception as e:
                                print(f"❌ {json_path} の読み込みエラー: {e}")
        
        print(f"📊 合計 {len(self.data)} 環境のデータを読み込み完了")
        return len(self.data) > 0
    
    def extract_exploration_progress(self):
        """各環境の探査率の進行を抽出"""
        print("📈 探査率の進行を抽出中...")
        
        progress_data = []
        
        for key, env_data in self.data.items():
            config_type = env_data['config_type']
            obstacle_density = env_data['obstacle_density']
            episodes = env_data['data'].get('episodes', [])
            
            for episode in episodes:
                episode_id = episode.get('episode', 0)
                step_details = episode.get('step_details', [])
                
                for step in step_details:
                    step_num = step.get('step', 0)
                    exploration_rate = step.get('exploration_rate', 0.0)
                    
                    progress_data.append({
                        'config_type': config_type,
                        'obstacle_density': obstacle_density,
                        'episode': episode_id,
                        'step': step_num,
                        'exploration_rate': exploration_rate
                    })
        
        self.combined_data = pd.DataFrame(progress_data)
        print(f"📊 {len(self.combined_data)} ステップのデータを抽出完了")
        
        # データの基本統計を表示
        if not self.combined_data.empty:
            print("\n📋 データ概要:")
            print(f"  Config数: {self.combined_data['config_type'].nunique()}")
            print(f"  障害物密度数: {self.combined_data['obstacle_density'].nunique()}")
            print(f"  エピソード数: {self.combined_data['episode'].nunique()}")
            print(f"  平均探査率: {self.combined_data['exploration_rate'].mean():.3f}")
    
    def create_exploration_progress_plots(self, output_dir="analysis_results"):
        """探査率の進行を可視化"""
        print("🎨 探査率の進行を可視化中...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 環境別の探査率進行（全エピソード）
        self._plot_environment_progress(output_dir)
        
        # 2. エピソード別の探査率進行
        self._plot_episode_progress(output_dir)
        
        # 3. 障害物密度別の比較
        self._plot_density_comparison(output_dir)
        
        # 4. Config別の最終探査率比較
        self._plot_config_comparison(output_dir)
        
        print(f"✅ 可視化完了: {output_dir}")
    
    def _plot_environment_progress(self, output_dir):
        """環境別の探査率進行をプロット"""
        plt.figure(figsize=(16, 10))
        
        # 各環境の色を設定
        colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'}
        densities = sorted(self.combined_data['obstacle_density'].unique())
        
        for i, density in enumerate(densities):
            plt.subplot(2, 2, i+1)
            
            density_data = self.combined_data[self.combined_data['obstacle_density'] == density]
            
            for config_type in sorted(density_data['config_type'].unique()):
                config_data = density_data[density_data['config_type'] == config_type]
                
                # ステップごとの平均探査率を計算
                step_avg = config_data.groupby('step')['exploration_rate'].mean().reset_index()
                
                plt.plot(step_avg['step'], step_avg['exploration_rate'], 
                        color=colors.get(config_type, 'gray'),
                        label=f'Config {config_type}',
                        linewidth=2, marker='o', markersize=4, alpha=0.8)
            
            plt.title(f'Exploration Progress (Obstacle Density: {density})', fontsize=14, fontweight='bold')
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Exploration Rate', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/environment_exploration_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_episode_progress(self, output_dir):
        """エピソード別の探査率進行をプロット"""
        plt.figure(figsize=(16, 10))
        
        colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'}
        densities = sorted(self.combined_data['obstacle_density'].unique())
        
        for i, density in enumerate(densities):
            plt.subplot(2, 2, i+1)
            
            density_data = self.combined_data[self.combined_data['obstacle_density'] == density]
            
            for config_type in sorted(density_data['config_type'].unique()):
                config_data = density_data[density_data['config_type'] == config_type]
                
                # エピソードごとの最終探査率を計算
                episode_final = config_data.groupby('episode')['exploration_rate'].max().reset_index()
                
                plt.plot(episode_final['episode'], episode_final['exploration_rate'], 
                        color=colors.get(config_type, 'gray'),
                        label=f'Config {config_type}',
                        linewidth=2, marker='s', markersize=6, alpha=0.8)
            
            plt.title(f'Episode-wise Exploration Progress (Obstacle Density: {density})', fontsize=14, fontweight='bold')
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Final Exploration Rate', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/episode_exploration_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_density_comparison(self, output_dir):
        """障害物密度別の比較をプロット"""
        plt.figure(figsize=(15, 10))
        
        # 1. 密度別の平均探査率
        plt.subplot(2, 3, 1)
        density_means = self.combined_data.groupby('obstacle_density')['exploration_rate'].mean()
        density_means.plot(kind='bar', color=['lightblue', 'orange', 'lightcoral'])
        plt.title('Average Exploration Rate by Obstacle Density', fontsize=14, fontweight='bold')
        plt.xlabel('Obstacle Density')
        plt.ylabel('Average Exploration Rate')
        plt.xticks(rotation=0)
        
        # 2. 密度別の最終探査率分布
        plt.subplot(2, 3, 2)
        final_rates = self.combined_data.groupby(['obstacle_density', 'episode'])['exploration_rate'].max().reset_index()
        final_rates.boxplot(column='exploration_rate', by='obstacle_density', ax=plt.gca())
        plt.title('Final Exploration Rate Distribution by Density', fontsize=14, fontweight='bold')
        plt.xlabel('Obstacle Density')
        plt.ylabel('Final Exploration Rate')
        plt.suptitle('')  # デフォルトタイトルを削除
        
        # 3. 密度別の探査率向上速度
        plt.subplot(2, 3, 3)
        for density in sorted(self.combined_data['obstacle_density'].unique()):
            density_data = self.combined_data[self.combined_data['obstacle_density'] == density]
            step_avg = density_data.groupby('step')['exploration_rate'].mean()
            plt.plot(step_avg.index, step_avg.values, 
                    label=f'Density {density}', linewidth=2, marker='o', markersize=4)
        plt.title('Exploration Rate Progress by Density', fontsize=14, fontweight='bold')
        plt.xlabel('Step')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 密度×Configのヒートマップ
        plt.subplot(2, 3, 4)
        pivot_data = self.combined_data.groupby(['obstacle_density', 'config_type'])['exploration_rate'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Exploration Rate Heatmap (Density × Config)', fontsize=14, fontweight='bold')
        
        # 5. 密度別の学習曲線
        plt.subplot(2, 3, 5)
        for density in sorted(self.combined_data['obstacle_density'].unique()):
            density_data = self.combined_data[self.combined_data['obstacle_density'] == density]
            episode_avg = density_data.groupby('episode')['exploration_rate'].mean()
            plt.plot(episode_avg.index, episode_avg.values, 
                    label=f'Density {density}', linewidth=2, marker='s', markersize=6)
        plt.title('Learning Curves by Density', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 密度別の統計サマリー
        plt.subplot(2, 3, 6)
        density_stats = self.combined_data.groupby('obstacle_density')['exploration_rate'].agg(['mean', 'std', 'min', 'max'])
        plt.axis('off')
        table_data = []
        for density in density_stats.index:
            stats = density_stats.loc[density]
            table_data.append([f'Density {density}', f'{stats["mean"]:.3f}', f'{stats["std"]:.3f}', 
                             f'{stats["min"]:.3f}', f'{stats["max"]:.3f}'])
        
        table = plt.table(cellText=table_data, 
                         colLabels=['Density', 'Mean', 'Std', 'Min', 'Max'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Statistical Summary by Density', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/density_comparison_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_config_comparison(self, output_dir):
        """Config別の最終探査率比較をプロット"""
        plt.figure(figsize=(15, 10))
        
        # 1. Config別の平均探査率
        plt.subplot(2, 3, 1)
        config_means = self.combined_data.groupby('config_type')['exploration_rate'].mean()
        colors = ['blue', 'red', 'green', 'orange']
        config_means.plot(kind='bar', color=colors)
        plt.title('Average Exploration Rate by Config', fontsize=14, fontweight='bold')
        plt.xlabel('Config Type')
        plt.ylabel('Average Exploration Rate')
        plt.xticks(rotation=0)
        
        # 2. Config別の最終探査率分布
        plt.subplot(2, 3, 2)
        final_rates = self.combined_data.groupby(['config_type', 'episode'])['exploration_rate'].max().reset_index()
        final_rates.boxplot(column='exploration_rate', by='config_type', ax=plt.gca())
        plt.title('Final Exploration Rate Distribution by Config', fontsize=14, fontweight='bold')
        plt.xlabel('Config Type')
        plt.ylabel('Final Exploration Rate')
        plt.suptitle('')  # デフォルトタイトルを削除
        
        # 3. Config別の探査率向上速度
        plt.subplot(2, 3, 3)
        for config_type in sorted(self.combined_data['config_type'].unique()):
            config_data = self.combined_data[self.combined_data['config_type'] == config_type]
            step_avg = config_data.groupby('step')['exploration_rate'].mean()
            plt.plot(step_avg.index, step_avg.values, 
                    label=f'Config {config_type}', linewidth=2, marker='o', markersize=4)
        plt.title('Exploration Rate Progress by Config', fontsize=14, fontweight='bold')
        plt.xlabel('Step')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Config×密度のヒートマップ
        plt.subplot(2, 3, 4)
        pivot_data = self.combined_data.groupby(['config_type', 'obstacle_density'])['exploration_rate'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Exploration Rate Heatmap (Config × Density)', fontsize=14, fontweight='bold')
        
        # 5. Config別の学習曲線
        plt.subplot(2, 3, 5)
        for config_type in sorted(self.combined_data['config_type'].unique()):
            config_data = self.combined_data[self.combined_data['config_type'] == config_type]
            episode_avg = config_data.groupby('episode')['exploration_rate'].mean()
            plt.plot(episode_avg.index, episode_avg.values, 
                    label=f'Config {config_type}', linewidth=2, marker='s', markersize=6)
        plt.title('Learning Curves by Config', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Config別の統計サマリー
        plt.subplot(2, 3, 6)
        config_stats = self.combined_data.groupby('config_type')['exploration_rate'].agg(['mean', 'std', 'min', 'max'])
        plt.axis('off')
        table_data = []
        for config_type in config_stats.index:
            stats = config_stats.loc[config_type]
            table_data.append([f'Config {config_type}', f'{stats["mean"]:.3f}', f'{stats["std"]:.3f}', 
                             f'{stats["min"]:.3f}', f'{stats["max"]:.3f}'])
        
        table = plt.table(cellText=table_data, 
                         colLabels=['Config', 'Mean', 'Std', 'Min', 'Max'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Statistical Summary by Config', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/config_comparison_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_dir="analysis_results"):
        """分析結果のサマリーレポートを生成"""
        print("📋 サマリーレポートを生成中...")
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = f"{output_dir}/exploration_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Environment Exploration Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本統計
            f.write("1. Basic Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total environments: {len(self.data)}\n")
            f.write(f"Total steps: {len(self.combined_data)}\n")
            f.write(f"Config types: {sorted(self.combined_data['config_type'].unique())}\n")
            f.write(f"Obstacle densities: {sorted(self.combined_data['obstacle_density'].unique())}\n\n")
            
            # Config別の統計
            f.write("2. Config-wise Statistics\n")
            f.write("-" * 30 + "\n")
            config_stats = self.combined_data.groupby('config_type')['exploration_rate'].agg(['mean', 'std', 'min', 'max'])
            for config_type in config_stats.index:
                stats = config_stats.loc[config_type]
                f.write(f"Config {config_type}:\n")
                f.write(f"  Mean: {stats['mean']:.3f}\n")
                f.write(f"  Std:  {stats['std']:.3f}\n")
                f.write(f"  Min:  {stats['min']:.3f}\n")
                f.write(f"  Max:  {stats['max']:.3f}\n\n")
            
            # 密度別の統計
            f.write("3. Density-wise Statistics\n")
            f.write("-" * 30 + "\n")
            density_stats = self.combined_data.groupby('obstacle_density')['exploration_rate'].agg(['mean', 'std', 'min', 'max'])
            for density in density_stats.index:
                stats = density_stats.loc[density]
                f.write(f"Density {density}:\n")
                f.write(f"  Mean: {stats['mean']:.3f}\n")
                f.write(f"  Std:  {stats['std']:.3f}\n")
                f.write(f"  Min:  {stats['min']:.3f}\n")
                f.write(f"  Max:  {stats['max']:.3f}\n\n")
            
            # 最良パフォーマンス
            f.write("4. Best Performance Analysis\n")
            f.write("-" * 30 + "\n")
            best_config = config_stats['mean'].idxmax()
            best_density = density_stats['mean'].idxmax()
            f.write(f"Best performing config: Config {best_config} (avg: {config_stats.loc[best_config, 'mean']:.3f})\n")
            f.write(f"Best performing density: {best_density} (avg: {density_stats.loc[best_density, 'mean']:.3f})\n\n")
            
            # 学習効果の分析
            f.write("5. Learning Effect Analysis\n")
            f.write("-" * 30 + "\n")
            for config_type in sorted(self.combined_data['config_type'].unique()):
                config_data = self.combined_data[self.combined_data['config_type'] == config_type]
                episode_final = config_data.groupby('episode')['exploration_rate'].max()
                if len(episode_final) > 1:
                    improvement = episode_final.iloc[-1] - episode_final.iloc[0]
                    f.write(f"Config {config_type}: {improvement:.3f} improvement from episode 1 to {len(episode_final)}\n")
            
        print(f"✅ サマリーレポート生成完了: {report_path}")
    
    def run_analysis(self):
        """分析を実行"""
        print("🚀 環境探査率比較分析を開始")
        print("=" * 50)
        
        # 1. データ読み込み
        if not self.load_verification_results():
            return False
        
        # 2. 探査率の進行を抽出
        self.extract_exploration_progress()
        
        if self.combined_data.empty:
            print("❌ 分析可能なデータが見つかりません")
            return False
        
        # 3. 可視化
        self.create_exploration_progress_plots()
        
        # 4. サマリーレポート生成
        self.generate_summary_report()
        
        print("\n🎉 環境探査率比較分析完了！")
        print("📁 結果は 'analysis_results' ディレクトリに保存されました")
        
        return True

def main():
    """メイン関数"""
    analyzer = EnvironmentExplorationAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\n✅ 分析が正常に完了しました")
    else:
        print("\n❌ 分析中にエラーが発生しました")

if __name__ == "__main__":
    main() 