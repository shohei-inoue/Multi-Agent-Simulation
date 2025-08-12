#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Performance Comparison Script
各Configで最終探査率が最も良かった環境を抽出し、ステップごとの探査率上昇を比較

使用方法:
    python best_performance_comparison.py

出力:
    - best_performance_comparison/ ディレクトリに結果を保存
    - 各Configの最高性能時の探査率上昇比較グラフ
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

class BestPerformanceAnalyzer:
    def __init__(self, data_dir: str = "verify_configs/verification_results"):
        """
        初期化
        
        Args:
            data_dir: 検証結果が保存されているディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.step_data = pd.DataFrame()
        self.episode_data = pd.DataFrame()
        self.best_configs = {}  # 各Configの最高性能環境を格納
        
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
        1エピソード目のデータを読み込み
        
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
                        
                        config_type = config_name.split('_')[1].upper()
                        obstacle_density = float(config_name.split('_')[3])
                        
                        # エピソード基本情報を抽出
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
    
    def identify_best_configs(self) -> Dict[str, Tuple[float, float]]:
        """
        各Configで最終探査率が最も良かった環境（障害物密度）を特定
        
        Returns:
            Dict[str, Tuple[float, float]]: {config_type: (obstacle_density, final_exploration_rate)}
        """
        print("=== 各Config最高性能環境の特定中 ===")
        
        best_configs = {}
        
        for config_type in sorted(self.episode_data['config_type'].unique()):
            config_data = self.episode_data[self.episode_data['config_type'] == config_type]
            
            # 最高の最終探査率を持つ環境を特定
            best_row = config_data.loc[config_data['final_exploration_rate'].idxmax()]
            
            best_density = best_row['obstacle_density']
            best_exploration_rate = best_row['final_exploration_rate']
            
            best_configs[config_type] = (best_density, best_exploration_rate)
            
            print(f"  Config {config_type}: 障害物密度 {best_density} で最高探査率 {best_exploration_rate:.4f}")
        
        self.best_configs = best_configs
        return best_configs
    
    def generate_best_performance_comparison(self, output_dir: str = "best_performance_comparison"):
        """
        各Configの最高性能時の探査率上昇を比較
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 最高性能比較グラフ生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.step_data.empty or not self.best_configs:
            print("❌ データが不足しています")
            return
        
        # 大きなサイズでグラフを作成
        plt.figure(figsize=(16, 12))
        
        # メインの比較グラフ
        plt.subplot(2, 2, 1)
        
        best_performance_data = []
        
        for config_type, (best_density, best_exploration_rate) in self.best_configs.items():
            # 該当するConfigと環境のステップデータを抽出
            config_step_data = self.step_data[
                (self.step_data['config_type'] == config_type) & 
                (self.step_data['obstacle_density'] == best_density)
            ]
            
            if len(config_step_data) > 0:
                # ステップごとの平均探査率を計算
                step_stats = config_step_data.groupby('step_id')['exploration_rate'].agg(['mean', 'std']).reset_index()
                
                # メインライン
                plt.plot(step_stats['step_id'], step_stats['mean'],
                        color=self.config_colors[config_type], 
                        linewidth=4, 
                        marker='o', 
                        markersize=8,
                        label=f'Config {config_type}: {self.config_descriptions[config_type]}\n(Density: {best_density}, Final: {best_exploration_rate:.4f})',
                        alpha=0.9)
                
                # 標準偏差を影で表示
                plt.fill_between(step_stats['step_id'],
                               step_stats['mean'] - step_stats['std'],
                               step_stats['mean'] + step_stats['std'],
                               color=self.config_colors[config_type], 
                               alpha=0.2)
                
                # 分析用データを保存
                best_performance_data.append({
                    'config_type': config_type,
                    'best_density': best_density,
                    'final_exploration_rate': best_exploration_rate,
                    'step_data': step_stats
                })
        
        plt.title('Best Performance Comparison - Exploration Rate Progress', 
                 fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Step', fontsize=14, fontweight='bold')
        plt.ylabel('Exploration Rate', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.4)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        
        # 探査率上昇速度の比較
        plt.subplot(2, 2, 2)
        
        for data in best_performance_data:
            step_means = data['step_data']['mean'].values
            if len(step_means) > 1:
                # 探査率の変化率を計算（微分の近似）
                exploration_diff = np.diff(step_means)
                step_ids = data['step_data']['step_id'].values[1:]
                
                plt.plot(step_ids, exploration_diff,
                        color=self.config_colors[data['config_type']], 
                        linewidth=3, 
                        marker='s', 
                        markersize=6,
                        label=f'Config {data["config_type"]}',
                        alpha=0.8)
        
        plt.title('Exploration Rate Increase Speed Comparison', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Step', fontsize=12, fontweight='bold')
        plt.ylabel('Exploration Rate Increase per Step', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.4)
        
        # 累積探査率の比較
        plt.subplot(2, 2, 3)
        
        for data in best_performance_data:
            step_stats = data['step_data']
            plt.plot(step_stats['step_id'], step_stats['mean'],
                    color=self.config_colors[data['config_type']], 
                    linewidth=3,
                    label=f'Config {data["config_type"]} (Final: {data["final_exploration_rate"]:.4f})',
                    alpha=0.8)
        
        plt.title('Cumulative Exploration Rate Comparison', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Step', fontsize=12, fontweight='bold')
        plt.ylabel('Exploration Rate', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.4)
        
        # 最終探査率のバー比較
        plt.subplot(2, 2, 4)
        
        configs = list(self.best_configs.keys())
        final_rates = [self.best_configs[config][1] for config in configs]
        densities = [self.best_configs[config][0] for config in configs]
        
        bars = plt.bar(configs, final_rates, 
                      color=[self.config_colors[config] for config in configs],
                      alpha=0.8, capsize=5)
        
        # バーの上に数値と環境情報を表示
        for bar, rate, density in zip(bars, final_rates, densities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.4f}\n(Density: {density})', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.title('Final Exploration Rate Comparison\n(Best Performance Environment)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Config Type', fontsize=12, fontweight='bold')
        plt.ylabel('Final Exploration Rate', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.4, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/best_performance_comparison.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 詳細な単体グラフも生成
        self.generate_detailed_single_graph(output_dir, best_performance_data)
        
        print(f"✓ 最高性能比較グラフを {output_dir}/ に保存しました")
    
    def generate_detailed_single_graph(self, output_dir: str, best_performance_data: List[Dict]):
        """
        詳細な単体グラフを生成
        
        Args:
            output_dir: 出力ディレクトリ
            best_performance_data: 最高性能データのリスト
        """
        # 大きな単体グラフ
        plt.figure(figsize=(16, 10))
        
        for data in best_performance_data:
            step_stats = data['step_data']
            config_type = data['config_type']
            
            plt.plot(step_stats['step_id'], step_stats['mean'],
                    color=self.config_colors[config_type], 
                    linewidth=4, 
                    marker='o', 
                    markersize=8,
                    label=f'Config {config_type}: {self.config_descriptions[config_type]}\n'
                          f'Best Environment - Density: {data["best_density"]}, Final Rate: {data["final_exploration_rate"]:.4f}',
                    alpha=0.9)
            
            # 標準偏差を影で表示
            plt.fill_between(step_stats['step_id'],
                           step_stats['mean'] - step_stats['std'],
                           step_stats['mean'] + step_stats['std'],
                           color=self.config_colors[config_type], 
                           alpha=0.2)
        
        plt.title('Best Performance Exploration Rate Progress Comparison\nEach Config at Their Optimal Environment', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Step', fontsize=16, fontweight='bold')
        plt.ylabel('Exploration Rate', fontsize=16, fontweight='bold')
        
        plt.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')
        
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # 背景色を設定
        plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/best_performance_detailed.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_summary_statistics(self, output_dir: str = "best_performance_comparison"):
        """
        最高性能の統計サマリーを生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        print("=== 最高性能統計サマリー生成中 ===")
        
        summary_data = []
        
        for config_type, (best_density, best_exploration_rate) in self.best_configs.items():
            # 該当するステップデータを取得
            config_step_data = self.step_data[
                (self.step_data['config_type'] == config_type) & 
                (self.step_data['obstacle_density'] == best_density)
            ]
            
            if len(config_step_data) > 0:
                step_means = config_step_data.groupby('step_id')['exploration_rate'].mean()
                
                # 探査率上昇速度を計算
                if len(step_means) > 1:
                    exploration_diff = np.diff(step_means.values)
                    avg_increase_rate = np.mean(exploration_diff)
                    max_increase_rate = np.max(exploration_diff)
                    min_increase_rate = np.min(exploration_diff)
                else:
                    avg_increase_rate = max_increase_rate = min_increase_rate = 0
                
                # 目標探査率到達時間を計算
                target_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
                achievement_times = {}
                
                for target_rate in target_rates:
                    achieved_steps = step_means[step_means >= target_rate]
                    if len(achieved_steps) > 0:
                        achievement_times[f'time_to_{target_rate}'] = achieved_steps.index[0]
                    else:
                        achievement_times[f'time_to_{target_rate}'] = None
                
                summary_row = {
                    'config_type': config_type,
                    'config_description': self.config_descriptions[config_type],
                    'best_obstacle_density': best_density,
                    'final_exploration_rate': best_exploration_rate,
                    'avg_increase_rate': avg_increase_rate,
                    'max_increase_rate': max_increase_rate,
                    'min_increase_rate': min_increase_rate,
                    'total_steps': len(step_means),
                    'exploration_efficiency': best_exploration_rate / len(step_means) if len(step_means) > 0 else 0,
                    **achievement_times
                }
                
                summary_data.append(summary_row)
        
        # CSVで保存
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f"{output_dir}/best_performance_statistics.csv", index=False)
            print(f"✓ 最高性能統計を {output_dir}/best_performance_statistics.csv に保存しました")
        
        # 比較表も生成
        self.generate_comparison_table(output_dir, summary_data)
    
    def generate_comparison_table(self, output_dir: str, summary_data: List[Dict]):
        """
        比較表を生成
        
        Args:
            output_dir: 出力ディレクトリ
            summary_data: 統計データ
        """
        comparison_data = []
        
        for data in summary_data:
            comparison_row = {
                'Config': data['config_type'],
                'Description': data['config_description'],
                'Best_Environment': f"Density {data['best_obstacle_density']}",
                'Final_Exploration_Rate': f"{data['final_exploration_rate']:.4f}",
                'Avg_Increase_Rate': f"{data['avg_increase_rate']:.6f}",
                'Max_Increase_Rate': f"{data['max_increase_rate']:.6f}",
                'Exploration_Efficiency': f"{data['exploration_efficiency']:.6f}",
                'Time_to_30%': data.get('time_to_0.3', 'N/A')
            }
            comparison_data.append(comparison_row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f"{output_dir}/best_performance_comparison_table.csv", index=False)
            print(f"✓ 比較表を {output_dir}/best_performance_comparison_table.csv に保存しました")
    
    def run_analysis(self):
        """
        分析を実行
        """
        print("🚀 各Config最高性能比較分析を開始します\n")
        
        # データ読み込み
        if not self.load_first_episode_data():
            print("❌ データ読み込みに失敗しました")
            return False
        
        # 最高性能環境を特定
        best_configs = self.identify_best_configs()
        if not best_configs:
            print("❌ 最高性能環境の特定に失敗しました")
            return False
        
        output_dir = "best_performance_comparison"
        
        # 分析実行
        self.generate_best_performance_comparison(output_dir)
        self.generate_summary_statistics(output_dir)
        
        print(f"\n🎉 各Config最高性能比較分析完了！")
        print(f"📁 結果は {output_dir}/ ディレクトリに保存されました")
        
        # 結果サマリーを表示
        print("\n📊 最高性能サマリー:")
        for config_type, (density, rate) in best_configs.items():
            print(f"  Config {config_type}: 障害物密度 {density} で最終探査率 {rate:.4f}")
        
        return True


def main():
    """メイン関数"""
    analyzer = BestPerformanceAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 