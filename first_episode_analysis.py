#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First Episode Analysis Script
各Configの1エピソード目のみを抽出して比較分析する

使用方法:
    python first_episode_analysis.py

出力:
    - first_episode_analysis_results/ ディレクトリに結果を保存
    - PNG形式のグラフ
    - CSV形式の統計データ
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

class FirstEpisodeAnalyzer:
    def __init__(self, data_dir: str = "verify_configs/verification_results"):
        """
        初期化
        
        Args:
            data_dir: 検証結果が保存されているディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.first_episode_data = pd.DataFrame()
        self.step_data = pd.DataFrame()
        
    def load_first_episode_data(self) -> bool:
        """
        各Configの1エピソード目のデータを読み込み
        
        Returns:
            bool: データ読み込み成功可否
        """
        print("=== 1エピソード目データ読み込み中 ===")
        
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
                        
                        # 基本情報を抽出
                        episode_info = {
                            'config_type': config_name.split('_')[2].upper(),  # A, B, C, D
                            'obstacle_density': float(config_name.split('_')[3]),
                            'episode_id': first_episode.get('episode', 1),
                            'final_exploration_rate': first_episode.get('final_exploration_rate', 0.0),
                            'steps_taken': first_episode.get('steps_taken', 0),
                            'steps_to_target': first_episode.get('steps_to_target', None),
                            'total_reward': first_episode.get('total_reward', 0.0),
                            'avg_reward': first_episode.get('avg_reward', 0.0),
                            'file_source': json_file.name
                        }
                        episode_records.append(episode_info)
                        
                        # ステップ詳細データがある場合
                        if 'step_details' in first_episode:
                            for step_detail in first_episode['step_details']:
                                step_info = {
                                    'config_type': config_name.split('_')[2].upper(),
                                    'obstacle_density': float(config_name.split('_')[3]),
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
    
    def generate_comparison_plots(self, output_dir: str = "first_episode_analysis_results"):
        """
        1エピソード目の比較グラフを生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 1エピソード目比較グラフ生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.first_episode_data.empty:
            print("❌ データが空のため、グラフ生成をスキップします")
            return
        
        # 1. 基本比較グラフ
        plt.figure(figsize=(16, 12))
        
        # 1-1. Config別最終探査率
        plt.subplot(2, 3, 1)
        config_exploration = self.first_episode_data.groupby('config_type')['final_exploration_rate'].mean()
        config_exploration.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Final Exploration Rate by Config (Episode 1)')
        plt.xlabel('Config Type')
        plt.ylabel('Final Exploration Rate')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # 1-2. Config別ステップ数
        plt.subplot(2, 3, 2)
        config_steps = self.first_episode_data.groupby('config_type')['steps_taken'].mean()
        config_steps.plot(kind='bar', color=['lightsteelblue', 'lightpink', 'lightseagreen', 'lightyellow'])
        plt.title('Steps Taken by Config (Episode 1)')
        plt.xlabel('Config Type')
        plt.ylabel('Steps Taken')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # 1-3. 障害物密度別の影響
        plt.subplot(2, 3, 3)
        density_exploration = self.first_episode_data.groupby('obstacle_density')['final_exploration_rate'].mean()
        density_exploration.plot(kind='bar', color=['lightblue', 'orange', 'lightcoral'])
        plt.title('Final Exploration Rate by Obstacle Density (Episode 1)')
        plt.xlabel('Obstacle Density')
        plt.ylabel('Final Exploration Rate')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # 1-4. Config×障害物密度のヒートマップ
        plt.subplot(2, 3, 4)
        if len(self.first_episode_data) > 1:
            pivot_data = self.first_episode_data.pivot_table(
                values='final_exploration_rate', 
                index='config_type', 
                columns='obstacle_density', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title('Exploration Rate Heatmap (Episode 1)')
        
        # 1-5. 探査率分布（箱ひげ図）
        plt.subplot(2, 3, 5)
        config_types = sorted(self.first_episode_data['config_type'].unique())
        exploration_data = []
        labels = []
        
        for config_type in config_types:
            config_data = self.first_episode_data[
                self.first_episode_data['config_type'] == config_type
            ]['final_exploration_rate'].values
            exploration_data.append(config_data)
            labels.append(f'Config {config_type}')
        
        if exploration_data:
            plt.boxplot(exploration_data, labels=labels)
            plt.title('Exploration Rate Distribution by Config (Episode 1)')
            plt.ylabel('Final Exploration Rate')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 1-6. 探査率 vs ステップ数の散布図
        plt.subplot(2, 3, 6)
        colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'}
        for config_type in self.first_episode_data['config_type'].unique():
            config_df = self.first_episode_data[self.first_episode_data['config_type'] == config_type]
            plt.scatter(config_df['steps_taken'], 
                       config_df['final_exploration_rate'],
                       c=colors.get(config_type, 'gray'),
                       label=f'Config {config_type}',
                       alpha=0.7, s=100)
        
        plt.xlabel('Steps Taken')
        plt.ylabel('Final Exploration Rate')
        plt.title('Exploration Rate vs Steps (Episode 1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/first_episode_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ステップ詳細分析（データがある場合）
        if not self.step_data.empty:
            self.generate_step_analysis(output_dir)
        
        print(f"✓ 比較グラフを {output_dir}/ に保存しました")
    
    def generate_step_analysis(self, output_dir: str):
        """
        1エピソード目のステップ詳細分析
        
        Args:
            output_dir: 出力ディレクトリ
        """
        plt.figure(figsize=(16, 10))
        
        # 2-1. ステップごとの探査率変化
        plt.subplot(2, 3, 1)
        colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'}
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_exploration_by_step = config_steps.groupby('step_id')['exploration_rate'].mean()
            plt.plot(avg_exploration_by_step.index, avg_exploration_by_step.values, 
                    color=colors.get(config_type, 'gray'),
                    label=f'Config {config_type}', linewidth=2, marker='o', markersize=3)
        
        plt.title('Exploration Rate Progress (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-2. ステップごとの報酬変化
        plt.subplot(2, 3, 2)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_reward_by_step = config_steps.groupby('step_id')['reward'].mean()
            plt.plot(avg_reward_by_step.index, avg_reward_by_step.values, 
                    color=colors.get(config_type, 'gray'),
                    label=f'Config {config_type}', linewidth=2, marker='s', markersize=3)
        
        plt.title('Reward Progress (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-3. スワーム数の変化
        plt.subplot(2, 3, 3)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_swarms_by_step = config_steps.groupby('step_id')['swarm_count'].mean()
            plt.plot(avg_swarms_by_step.index, avg_swarms_by_step.values, 
                    color=colors.get(config_type, 'gray'),
                    label=f'Config {config_type}', linewidth=2, marker='^', markersize=3)
        
        plt.title('Swarm Count Progress (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Average Swarm Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-4. 衝突発生率
        plt.subplot(2, 3, 4)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            collision_rate_by_step = config_steps.groupby('step_id')['agent_collision_flag'].mean()
            plt.plot(collision_rate_by_step.index, collision_rate_by_step.values, 
                    color=colors.get(config_type, 'gray'),
                    label=f'Config {config_type}', linewidth=2, marker='x', markersize=3)
        
        plt.title('Collision Rate Progress (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Collision Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-5. 探査効率（探査率/ステップ）
        plt.subplot(2, 3, 5)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            # 探査効率を計算
            efficiency_data = []
            step_ids = []
            for step_id in sorted(config_steps['step_id'].unique()):
                step_data = config_steps[config_steps['step_id'] == step_id]
                if step_id > 0:  # ゼロ除算を避ける
                    efficiency = step_data['exploration_rate'].mean() / step_id
                    efficiency_data.append(efficiency)
                    step_ids.append(step_id)
            
            if efficiency_data:
                plt.plot(step_ids, efficiency_data, 
                        color=colors.get(config_type, 'gray'),
                        label=f'Config {config_type}', linewidth=2, marker='d', markersize=3)
        
        plt.title('Exploration Efficiency (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Exploration Rate / Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-6. 累積報酬
        plt.subplot(2, 3, 6)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            cumulative_reward = config_steps.groupby('step_id')['reward'].sum().cumsum()
            plt.plot(cumulative_reward.index, cumulative_reward.values, 
                    color=colors.get(config_type, 'gray'),
                    label=f'Config {config_type}', linewidth=2)
        
        plt.title('Cumulative Reward (Episode 1)')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/first_episode_step_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_statistical_analysis(self, output_dir: str = "first_episode_analysis_results"):
        """
        1エピソード目の統計分析
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 1エピソード目統計分析実行中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.first_episode_data.empty:
            print("❌ データが不足しています")
            return
        
        # 基本統計量
        stats_summary = []
        
        for config_type in sorted(self.first_episode_data['config_type'].unique()):
            config_data = self.first_episode_data[self.first_episode_data['config_type'] == config_type]
            
            stats_row = {
                'Config': config_type,
                'Sample_Count': len(config_data),
                'Exploration_Rate_Mean': config_data['final_exploration_rate'].mean(),
                'Exploration_Rate_Std': config_data['final_exploration_rate'].std(),
                'Exploration_Rate_Min': config_data['final_exploration_rate'].min(),
                'Exploration_Rate_Max': config_data['final_exploration_rate'].max(),
                'Steps_Mean': config_data['steps_taken'].mean(),
                'Steps_Std': config_data['steps_taken'].std(),
                'Steps_Min': config_data['steps_taken'].min(),
                'Steps_Max': config_data['steps_taken'].max(),
            }
            
            if 'total_reward' in config_data.columns:
                stats_row.update({
                    'Total_Reward_Mean': config_data['total_reward'].mean(),
                    'Total_Reward_Std': config_data['total_reward'].std(),
                })
            
            stats_summary.append(stats_row)
        
        # 統計結果をCSVで保存
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv(f"{output_dir}/first_episode_statistics.csv", index=False)
        
        # ANOVA分析（Config間の差）
        config_groups = []
        config_names = []
        
        for config_type in sorted(self.first_episode_data['config_type'].unique()):
            config_data = self.first_episode_data[
                self.first_episode_data['config_type'] == config_type
            ]['final_exploration_rate'].values
            
            if len(config_data) > 0:
                config_groups.append(config_data)
                config_names.append(config_type)
        
        if len(config_groups) > 1:
            try:
                f_stat, p_value = stats.f_oneway(*config_groups)
                anova_result = {
                    'F_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # ANOVA結果を保存
                with open(f"{output_dir}/first_episode_anova.json", 'w') as f:
                    json.dump(anova_result, f, indent=2)
                
                print(f"✓ ANOVA分析結果: F={f_stat:.4f}, p={p_value:.4f}")
                
            except Exception as e:
                print(f"❌ ANOVA分析エラー: {e}")
        
        print(f"✓ 統計分析結果を {output_dir}/ に保存しました")
    
    def generate_summary_report(self, output_dir: str = "first_episode_analysis_results"):
        """
        1エピソード目のサマリーレポート生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 1エピソード目サマリーレポート生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.first_episode_data.empty:
            print("❌ データが不足しています")
            return
        
        report_lines = []
        report_lines.append("# First Episode Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # データ概要
        report_lines.append("## Data Overview")
        report_lines.append(f"- Total configurations analyzed: {len(self.first_episode_data['config_type'].unique())}")
        report_lines.append(f"- Total simulations: {len(self.first_episode_data)}")
        report_lines.append(f"- Obstacle densities: {sorted(self.first_episode_data['obstacle_density'].unique())}")
        report_lines.append("")
        
        # Config別結果
        report_lines.append("## Results by Configuration")
        for config_type in sorted(self.first_episode_data['config_type'].unique()):
            config_data = self.first_episode_data[self.first_episode_data['config_type'] == config_type]
            
            report_lines.append(f"### Config {config_type}")
            report_lines.append(f"- Sample count: {len(config_data)}")
            report_lines.append(f"- Average exploration rate: {config_data['final_exploration_rate'].mean():.4f} ± {config_data['final_exploration_rate'].std():.4f}")
            report_lines.append(f"- Average steps taken: {config_data['steps_taken'].mean():.1f} ± {config_data['steps_taken'].std():.1f}")
            
            if 'total_reward' in config_data.columns:
                report_lines.append(f"- Average total reward: {config_data['total_reward'].mean():.4f} ± {config_data['total_reward'].std():.4f}")
            
            report_lines.append("")
        
        # 最高性能
        best_exploration = self.first_episode_data.loc[self.first_episode_data['final_exploration_rate'].idxmax()]
        report_lines.append("## Best Performance")
        report_lines.append(f"- Highest exploration rate: {best_exploration['final_exploration_rate']:.4f} (Config {best_exploration['config_type']}, Density {best_exploration['obstacle_density']})")
        report_lines.append(f"- Steps taken: {best_exploration['steps_taken']}")
        report_lines.append("")
        
        # レポート保存
        with open(f"{output_dir}/first_episode_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ サマリーレポートを {output_dir}/ に保存しました")
    
    def run_complete_analysis(self):
        """
        完全な1エピソード目分析を実行
        """
        print("🚀 1エピソード目完全分析を開始します\n")
        
        # データ読み込み
        if not self.load_first_episode_data():
            print("❌ データ読み込みに失敗しました")
            return False
        
        output_dir = "first_episode_analysis_results"
        
        # 分析実行
        self.generate_comparison_plots(output_dir)
        self.generate_statistical_analysis(output_dir)
        self.generate_summary_report(output_dir)
        
        print(f"\n🎉 1エピソード目分析完了！")
        print(f"📁 結果は {output_dir}/ ディレクトリに保存されました")
        
        return True


def main():
    """メイン関数"""
    analyzer = FirstEpisodeAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 