#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config Comparison Analysis Script
Config A, B, C, Dの差を詳細に分析する

Config設定:
- Config A: VFH-Fuzzy のみ（分岐・統合なし、学習なし）
- Config B: 学習済みモデル使用（分岐・統合なし）
- Config C: 分岐・統合あり（学習なし）
- Config D: 分岐・統合あり + 学習あり

使用方法:
    python config_comparison_analysis.py

出力:
    - config_comparison_results/ ディレクトリに結果を保存
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
from scipy.stats import ttest_ind
import itertools

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

class ConfigComparisonAnalyzer:
    def __init__(self, data_dir: str = "verify_configs/verification_results"):
        """
        初期化
        
        Args:
            data_dir: 検証結果が保存されているディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.episode_data = pd.DataFrame()
        self.step_data = pd.DataFrame()
        
        # Config設定の説明
        self.config_descriptions = {
            'A': 'VFH-Fuzzy only (No branching/integration, No learning)',
            'B': 'Pre-trained model (No branching/integration)',
            'C': 'Branching/Integration enabled (No learning)',
            'D': 'Branching/Integration + Learning'
        }
        
        # Config色設定
        self.config_colors = {
            'A': '#3498db',  # 青
            'B': '#e74c3c',  # 赤
            'C': '#2ecc71',  # 緑
            'D': '#f39c12'   # オレンジ
        }
        
    def load_all_data(self) -> bool:
        """
        全てのConfigのデータを読み込み
        
        Returns:
            bool: データ読み込み成功可否
        """
        print("=== Config比較データ読み込み中 ===")
        
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
                    
                    # エピソードデータを抽出
                    episodes_data = None
                    if isinstance(data, dict) and 'episodes' in data:
                        episodes_data = data['episodes']
                    elif isinstance(data, list):
                        episodes_data = data
                    
                    if episodes_data:
                        for episode in episodes_data:
                            # 基本情報を抽出
                            episode_info = {
                                'config_type': config_name.split('_')[1].upper(),  # A, B, C, D
                                'obstacle_density': float(config_name.split('_')[3]),
                                'episode_id': episode.get('episode', 1),
                                'final_exploration_rate': episode.get('final_exploration_rate', 0.0),
                                'steps_taken': episode.get('steps_taken', 0),
                                'steps_to_target': episode.get('steps_to_target', None),
                                'total_reward': episode.get('total_reward', 0.0),
                                'avg_reward': episode.get('avg_reward', 0.0),
                                'file_source': json_file.name
                            }
                            episode_records.append(episode_info)
                            
                            # ステップ詳細データがある場合
                            if 'step_details' in episode:
                                for step_detail in episode['step_details']:
                                    step_info = {
                                        'config_type': config_name.split('_')[1].upper(),
                                        'obstacle_density': float(config_name.split('_')[3]),
                                        'episode_id': episode.get('episode', 1),
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
            self.episode_data = pd.DataFrame(episode_records)
            print(f"✓ エピソードデータ: {len(self.episode_data)} 件")
        else:
            print("❌ エピソードデータが見つかりませんでした")
            
        if step_records:
            self.step_data = pd.DataFrame(step_records)
            print(f"✓ ステップデータ: {len(self.step_data)} 件")
        else:
            print("❌ ステップデータが見つかりませんでした")
        
        return len(episode_records) > 0
    
    def generate_config_overview(self, output_dir: str = "config_comparison_results"):
        """
        Config概要比較グラフを生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== Config概要比較グラフ生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.episode_data.empty:
            print("❌ データが空のため、グラフ生成をスキップします")
            return
        
        plt.figure(figsize=(18, 12))
        
        # 1. Config別平均探査率
        plt.subplot(2, 4, 1)
        config_exploration = self.episode_data.groupby('config_type')['final_exploration_rate'].mean()
        config_exploration_std = self.episode_data.groupby('config_type')['final_exploration_rate'].std()
        
        bars = plt.bar(config_exploration.index, config_exploration.values, 
                      color=[self.config_colors[c] for c in config_exploration.index],
                      alpha=0.8, capsize=5)
        plt.errorbar(config_exploration.index, config_exploration.values, 
                    yerr=config_exploration_std.values, fmt='none', color='black', capsize=5)
        
        plt.title('Average Final Exploration Rate by Config')
        plt.xlabel('Config Type')
        plt.ylabel('Final Exploration Rate')
        plt.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, val in zip(bars, config_exploration.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Config別平均ステップ数
        plt.subplot(2, 4, 2)
        config_steps = self.episode_data.groupby('config_type')['steps_taken'].mean()
        config_steps_std = self.episode_data.groupby('config_type')['steps_taken'].std()
        
        bars = plt.bar(config_steps.index, config_steps.values,
                      color=[self.config_colors[c] for c in config_steps.index],
                      alpha=0.8, capsize=5)
        plt.errorbar(config_steps.index, config_steps.values,
                    yerr=config_steps_std.values, fmt='none', color='black', capsize=5)
        
        plt.title('Average Steps Taken by Config')
        plt.xlabel('Config Type')
        plt.ylabel('Steps Taken')
        plt.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, val in zip(bars, config_steps.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 探査効率（探査率/ステップ）
        plt.subplot(2, 4, 3)
        self.episode_data['exploration_efficiency'] = self.episode_data['final_exploration_rate'] / self.episode_data['steps_taken']
        config_efficiency = self.episode_data.groupby('config_type')['exploration_efficiency'].mean()
        config_efficiency_std = self.episode_data.groupby('config_type')['exploration_efficiency'].std()
        
        bars = plt.bar(config_efficiency.index, config_efficiency.values,
                      color=[self.config_colors[c] for c in config_efficiency.index],
                      alpha=0.8, capsize=5)
        plt.errorbar(config_efficiency.index, config_efficiency.values,
                    yerr=config_efficiency_std.values, fmt='none', color='black', capsize=5)
        
        plt.title('Exploration Efficiency by Config')
        plt.xlabel('Config Type')
        plt.ylabel('Exploration Rate / Steps')
        plt.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, val in zip(bars, config_efficiency.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Config別探査率分布（バイオリンプロット）
        plt.subplot(2, 4, 4)
        config_types = sorted(self.episode_data['config_type'].unique())
        exploration_data = [
            self.episode_data[self.episode_data['config_type'] == config]['final_exploration_rate'].values
            for config in config_types
        ]
        
        parts = plt.violinplot(exploration_data, positions=range(len(config_types)), showmeans=True)
        for i, (part, config) in enumerate(zip(parts['bodies'], config_types)):
            part.set_facecolor(self.config_colors[config])
            part.set_alpha(0.7)
        
        plt.xticks(range(len(config_types)), [f'Config {c}' for c in config_types])
        plt.title('Exploration Rate Distribution by Config')
        plt.ylabel('Final Exploration Rate')
        plt.grid(True, alpha=0.3)
        
        # 5. 障害物密度の影響
        plt.subplot(2, 4, 5)
        for config_type in sorted(self.episode_data['config_type'].unique()):
            config_data = self.episode_data[self.episode_data['config_type'] == config_type]
            density_means = config_data.groupby('obstacle_density')['final_exploration_rate'].mean()
            plt.plot(density_means.index, density_means.values, 
                    color=self.config_colors[config_type], marker='o', linewidth=2, markersize=8,
                    label=f'Config {config_type}')
        
        plt.title('Impact of Obstacle Density by Config')
        plt.xlabel('Obstacle Density')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. エピソード進捗（Config別）
        plt.subplot(2, 4, 6)
        for config_type in sorted(self.episode_data['config_type'].unique()):
            config_data = self.episode_data[self.episode_data['config_type'] == config_type]
            episode_means = config_data.groupby('episode_id')['final_exploration_rate'].mean()
            plt.plot(episode_means.index, episode_means.values,
                    color=self.config_colors[config_type], marker='s', linewidth=2, markersize=6,
                    label=f'Config {config_type}')
        
        plt.title('Learning Progress by Config')
        plt.xlabel('Episode')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 探査率 vs ステップ数の関係
        plt.subplot(2, 4, 7)
        for config_type in sorted(self.episode_data['config_type'].unique()):
            config_data = self.episode_data[self.episode_data['config_type'] == config_type]
            plt.scatter(config_data['steps_taken'], config_data['final_exploration_rate'],
                       c=self.config_colors[config_type], alpha=0.6, s=50,
                       label=f'Config {config_type}')
        
        plt.xlabel('Steps Taken')
        plt.ylabel('Final Exploration Rate')
        plt.title('Exploration Rate vs Steps by Config')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Config説明テキスト
        plt.subplot(2, 4, 8)
        plt.axis('off')
        y_pos = 0.9
        for config, desc in self.config_descriptions.items():
            plt.text(0.05, y_pos, f'Config {config}:', fontweight='bold', fontsize=12,
                    color=self.config_colors[config], transform=plt.gca().transAxes)
            plt.text(0.05, y_pos-0.05, desc, fontsize=10, wrap=True,
                    transform=plt.gca().transAxes)
            y_pos -= 0.2
        
        plt.title('Configuration Descriptions')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/config_overview_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Config概要比較グラフを {output_dir}/ に保存しました")
    
    def generate_step_analysis(self, output_dir: str = "config_comparison_results"):
        """
        ステップレベルの詳細分析
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== ステップレベル詳細分析中 ===")
        
        if self.step_data.empty:
            print("❌ ステップデータが不足しています")
            return
        
        plt.figure(figsize=(16, 12))
        
        # 1. ステップごとの探査率変化（全エピソード平均）
        plt.subplot(2, 3, 1)
        for config_type in sorted(self.step_data['config_type'].unique()):
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_exploration_by_step = config_steps.groupby('step_id')['exploration_rate'].mean()
            plt.plot(avg_exploration_by_step.index, avg_exploration_by_step.values,
                    color=self.config_colors[config_type], linewidth=2, marker='o', markersize=4,
                    label=f'Config {config_type}')
        
        plt.title('Exploration Rate Progress by Config')
        plt.xlabel('Step')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. スワーム数の変化
        plt.subplot(2, 3, 2)
        for config_type in sorted(self.step_data['config_type'].unique()):
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_swarms_by_step = config_steps.groupby('step_id')['swarm_count'].mean()
            plt.plot(avg_swarms_by_step.index, avg_swarms_by_step.values,
                    color=self.config_colors[config_type], linewidth=2, marker='^', markersize=4,
                    label=f'Config {config_type}')
        
        plt.title('Swarm Count Progress by Config')
        plt.xlabel('Step')
        plt.ylabel('Average Swarm Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 衝突率の変化
        plt.subplot(2, 3, 3)
        for config_type in sorted(self.step_data['config_type'].unique()):
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            collision_rate_by_step = config_steps.groupby('step_id')['agent_collision_flag'].mean()
            plt.plot(collision_rate_by_step.index, collision_rate_by_step.values,
                    color=self.config_colors[config_type], linewidth=2, marker='x', markersize=4,
                    label=f'Config {config_type}')
        
        plt.title('Collision Rate Progress by Config')
        plt.xlabel('Step')
        plt.ylabel('Collision Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 報酬の変化
        plt.subplot(2, 3, 4)
        for config_type in sorted(self.step_data['config_type'].unique()):
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_reward_by_step = config_steps.groupby('step_id')['reward'].mean()
            plt.plot(avg_reward_by_step.index, avg_reward_by_step.values,
                    color=self.config_colors[config_type], linewidth=2, marker='s', markersize=4,
                    label=f'Config {config_type}')
        
        plt.title('Reward Progress by Config')
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 探査効率の変化
        plt.subplot(2, 3, 5)
        for config_type in sorted(self.step_data['config_type'].unique()):
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            efficiency_data = []
            step_ids = []
            for step_id in sorted(config_steps['step_id'].unique()):
                step_data = config_steps[config_steps['step_id'] == step_id]
                if step_id > 0:
                    efficiency = step_data['exploration_rate'].mean() / step_id
                    efficiency_data.append(efficiency)
                    step_ids.append(step_id)
            
            if efficiency_data:
                plt.plot(step_ids, efficiency_data,
                        color=self.config_colors[config_type], linewidth=2, marker='d', markersize=4,
                        label=f'Config {config_type}')
        
        plt.title('Exploration Efficiency Progress by Config')
        plt.xlabel('Step')
        plt.ylabel('Exploration Rate / Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 累積報酬
        plt.subplot(2, 3, 6)
        for config_type in sorted(self.step_data['config_type'].unique()):
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            cumulative_reward = config_steps.groupby('step_id')['reward'].sum().cumsum()
            plt.plot(cumulative_reward.index, cumulative_reward.values,
                    color=self.config_colors[config_type], linewidth=2,
                    label=f'Config {config_type}')
        
        plt.title('Cumulative Reward by Config')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/config_step_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ステップレベル分析グラフを {output_dir}/ に保存しました")
    
    def generate_statistical_comparison(self, output_dir: str = "config_comparison_results"):
        """
        統計的比較分析
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 統計的比較分析実行中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.episode_data.empty:
            print("❌ データが不足しています")
            return
        
        # 基本統計量
        stats_summary = []
        config_types = sorted(self.episode_data['config_type'].unique())
        
        for config_type in config_types:
            config_data = self.episode_data[self.episode_data['config_type'] == config_type]
            
            stats_row = {
                'Config': config_type,
                'Description': self.config_descriptions[config_type],
                'Sample_Count': len(config_data),
                'Exploration_Rate_Mean': config_data['final_exploration_rate'].mean(),
                'Exploration_Rate_Std': config_data['final_exploration_rate'].std(),
                'Exploration_Rate_Min': config_data['final_exploration_rate'].min(),
                'Exploration_Rate_Max': config_data['final_exploration_rate'].max(),
                'Steps_Mean': config_data['steps_taken'].mean(),
                'Steps_Std': config_data['steps_taken'].std(),
                'Steps_Min': config_data['steps_taken'].min(),
                'Steps_Max': config_data['steps_taken'].max(),
                'Exploration_Efficiency_Mean': (config_data['final_exploration_rate'] / config_data['steps_taken']).mean(),
                'Exploration_Efficiency_Std': (config_data['final_exploration_rate'] / config_data['steps_taken']).std(),
            }
            
            if 'total_reward' in config_data.columns:
                stats_row.update({
                    'Total_Reward_Mean': config_data['total_reward'].mean(),
                    'Total_Reward_Std': config_data['total_reward'].std(),
                })
            
            stats_summary.append(stats_row)
        
        # 統計結果をCSVで保存
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv(f"{output_dir}/config_comparison_statistics.csv", index=False)
        
        # ペアワイズt検定（探査率）
        t_test_results = []
        
        for config1, config2 in itertools.combinations(config_types, 2):
            data1 = self.episode_data[self.episode_data['config_type'] == config1]['final_exploration_rate']
            data2 = self.episode_data[self.episode_data['config_type'] == config2]['final_exploration_rate']
            
            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = ttest_ind(data1, data2)
                
                t_test_results.append({
                    'Config_1': config1,
                    'Config_2': config2,
                    'Mean_1': data1.mean(),
                    'Mean_2': data2.mean(),
                    'Mean_Diff': data1.mean() - data2.mean(),
                    'T_statistic': t_stat,
                    'P_value': p_value,
                    'Significant': p_value < 0.05,
                    'Effect_Size': abs(data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                })
        
        # t検定結果をCSVで保存
        if t_test_results:
            t_test_df = pd.DataFrame(t_test_results)
            t_test_df.to_csv(f"{output_dir}/config_pairwise_ttest.csv", index=False)
            
            print("✓ ペアワイズt検定結果:")
            for result in t_test_results:
                significance = "有意" if result['Significant'] else "非有意"
                print(f"  Config {result['Config_1']} vs {result['Config_2']}: "
                      f"p={result['P_value']:.4f} ({significance}), "
                      f"効果サイズ={result['Effect_Size']:.4f}")
        
        # ANOVA分析
        config_groups = []
        for config_type in config_types:
            config_data = self.episode_data[
                self.episode_data['config_type'] == config_type
            ]['final_exploration_rate'].values
            
            if len(config_data) > 0:
                config_groups.append(config_data)
        
        if len(config_groups) > 1:
            try:
                f_stat, p_value = stats.f_oneway(*config_groups)
                anova_result = {
                    'F_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'configs_tested': config_types
                }
                
                # ANOVA結果を保存
                with open(f"{output_dir}/config_anova_result.json", 'w') as f:
                    json.dump(anova_result, f, indent=2)
                
                print(f"✓ ANOVA分析結果: F={f_stat:.4f}, p={p_value:.4f}")
                
            except Exception as e:
                print(f"❌ ANOVA分析エラー: {e}")
        
        print(f"✓ 統計分析結果を {output_dir}/ に保存しました")
    
    def generate_performance_ranking(self, output_dir: str = "config_comparison_results"):
        """
        性能ランキング分析
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 性能ランキング分析実行中 ===")
        
        # 各指標での順位を計算
        config_performance = self.episode_data.groupby('config_type').agg({
            'final_exploration_rate': ['mean', 'std'],
            'steps_taken': ['mean', 'std'],
        }).round(4)
        
        # 探査効率を追加
        config_performance[('exploration_efficiency', 'mean')] = (
            self.episode_data.groupby('config_type')['final_exploration_rate'].mean() /
            self.episode_data.groupby('config_type')['steps_taken'].mean()
        )
        
        # ランキング表を作成
        ranking_data = []
        
        # 探査率ランキング
        exploration_ranking = config_performance[('final_exploration_rate', 'mean')].sort_values(ascending=False)
        for rank, (config, value) in enumerate(exploration_ranking.items(), 1):
            ranking_data.append({
                'Metric': 'Final Exploration Rate',
                'Rank': rank,
                'Config': config,
                'Value': f"{value:.4f}",
                'Description': self.config_descriptions[config]
            })
        
        # 探査効率ランキング
        efficiency_ranking = config_performance[('exploration_efficiency', 'mean')].sort_values(ascending=False)
        for rank, (config, value) in enumerate(efficiency_ranking.items(), 1):
            ranking_data.append({
                'Metric': 'Exploration Efficiency',
                'Rank': rank,
                'Config': config,
                'Value': f"{value:.6f}",
                'Description': self.config_descriptions[config]
            })
        
        # ステップ数ランキング（少ない方が良い）
        steps_ranking = config_performance[('steps_taken', 'mean')].sort_values(ascending=True)
        for rank, (config, value) in enumerate(steps_ranking.items(), 1):
            ranking_data.append({
                'Metric': 'Steps Taken (Lower is Better)',
                'Rank': rank,
                'Config': config,
                'Value': f"{value:.1f}",
                'Description': self.config_descriptions[config]
            })
        
        # ランキング結果をCSVで保存
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_csv(f"{output_dir}/config_performance_ranking.csv", index=False)
        
        print(f"✓ 性能ランキングを {output_dir}/ に保存しました")
        
        # ランキング結果を表示
        print("\n📊 性能ランキング結果:")
        for metric in ['Final Exploration Rate', 'Exploration Efficiency', 'Steps Taken (Lower is Better)']:
            print(f"\n{metric}:")
            metric_data = ranking_df[ranking_df['Metric'] == metric].sort_values('Rank')
            for _, row in metric_data.iterrows():
                print(f"  {row['Rank']}位: Config {row['Config']} - {row['Value']}")
    
    def generate_summary_report(self, output_dir: str = "config_comparison_results"):
        """
        総合サマリーレポート生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 総合サマリーレポート生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.episode_data.empty:
            print("❌ データが不足しています")
            return
        
        report_lines = []
        report_lines.append("# Config Comparison Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # データ概要
        report_lines.append("## Data Overview")
        report_lines.append(f"- Total configurations analyzed: {len(self.episode_data['config_type'].unique())}")
        report_lines.append(f"- Total episodes: {len(self.episode_data)}")
        report_lines.append(f"- Obstacle densities: {sorted(self.episode_data['obstacle_density'].unique())}")
        report_lines.append("")
        
        # Config説明
        report_lines.append("## Configuration Descriptions")
        for config, desc in self.config_descriptions.items():
            report_lines.append(f"- **Config {config}**: {desc}")
        report_lines.append("")
        
        # 性能サマリー
        report_lines.append("## Performance Summary")
        config_summary = self.episode_data.groupby('config_type').agg({
            'final_exploration_rate': ['mean', 'std', 'count'],
            'steps_taken': ['mean', 'std'],
        }).round(4)
        
        for config in sorted(self.episode_data['config_type'].unique()):
            exploration_mean = config_summary.loc[config, ('final_exploration_rate', 'mean')]
            exploration_std = config_summary.loc[config, ('final_exploration_rate', 'std')]
            steps_mean = config_summary.loc[config, ('steps_taken', 'mean')]
            count = config_summary.loc[config, ('final_exploration_rate', 'count')]
            
            report_lines.append(f"### Config {config}")
            report_lines.append(f"- Episodes analyzed: {count}")
            report_lines.append(f"- Average exploration rate: {exploration_mean:.4f} ± {exploration_std:.4f}")
            report_lines.append(f"- Average steps taken: {steps_mean:.1f}")
            report_lines.append(f"- Exploration efficiency: {exploration_mean/steps_mean:.6f}")
            report_lines.append("")
        
        # 主要な発見
        report_lines.append("## Key Findings")
        
        # 最高性能のConfig
        best_exploration = self.episode_data.groupby('config_type')['final_exploration_rate'].mean()
        best_config = best_exploration.idxmax()
        best_value = best_exploration.max()
        
        report_lines.append(f"- **Highest exploration rate**: Config {best_config} ({best_value:.4f})")
        
        # 最高効率のConfig
        efficiency = (self.episode_data.groupby('config_type')['final_exploration_rate'].mean() /
                     self.episode_data.groupby('config_type')['steps_taken'].mean())
        most_efficient = efficiency.idxmax()
        efficiency_value = efficiency.max()
        
        report_lines.append(f"- **Most efficient**: Config {most_efficient} ({efficiency_value:.6f} exploration/step)")
        
        # 最少ステップのConfig
        min_steps = self.episode_data.groupby('config_type')['steps_taken'].mean()
        fastest_config = min_steps.idxmin()
        fastest_value = min_steps.min()
        
        report_lines.append(f"- **Fastest completion**: Config {fastest_config} ({fastest_value:.1f} steps)")
        
        report_lines.append("")
        
        # 統計的有意差
        report_lines.append("## Statistical Significance")
        report_lines.append("Based on pairwise t-tests (p < 0.05):")
        
        # t検定結果を読み込み（既に計算済みの場合）
        t_test_file = Path(f"{output_dir}/config_pairwise_ttest.csv")
        if t_test_file.exists():
            t_test_df = pd.read_csv(t_test_file)
            significant_pairs = t_test_df[t_test_df['Significant'] == True]
            
            if len(significant_pairs) > 0:
                for _, row in significant_pairs.iterrows():
                    report_lines.append(f"- Config {row['Config_1']} vs {row['Config_2']}: "
                                      f"p={row['P_value']:.4f}, effect size={row['Effect_Size']:.4f}")
            else:
                report_lines.append("- No statistically significant differences found")
        
        report_lines.append("")
        
        # レポート保存
        with open(f"{output_dir}/config_comparison_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 総合サマリーレポートを {output_dir}/ に保存しました")
    
    def run_complete_analysis(self):
        """
        完全なConfig比較分析を実行
        """
        print("🚀 Config比較分析を開始します\n")
        
        # データ読み込み
        if not self.load_all_data():
            print("❌ データ読み込みに失敗しました")
            return False
        
        output_dir = "config_comparison_results"
        
        # 分析実行
        self.generate_config_overview(output_dir)
        self.generate_step_analysis(output_dir)
        self.generate_statistical_comparison(output_dir)
        self.generate_performance_ranking(output_dir)
        self.generate_summary_report(output_dir)
        
        print(f"\n🎉 Config比較分析完了！")
        print(f"📁 結果は {output_dir}/ ディレクトリに保存されました")
        
        return True


def main():
    """メイン関数"""
    analyzer = ConfigComparisonAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 