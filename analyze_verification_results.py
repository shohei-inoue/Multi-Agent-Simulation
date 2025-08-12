#!/usr/bin/env python3
"""
シミュレーション結果解析スクリプト
verify_configs/verification_resultsディレクトリ内の結果を解析し、
各Configの性能を比較・可視化します。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class VerificationResultAnalyzer:
    """シミュレーション結果解析クラス"""
    
    def __init__(self, results_dir: str = "verify_configs/verification_results"):
        """
        初期化
        Args:
            results_dir: 結果ディレクトリのパス
        """
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.summary_stats = {}
        
    def load_results(self):
        """すべての結果ファイルを読み込み"""
        print("=== シミュレーション結果読み込み中 ===")
        
        for config_dir in self.results_dir.iterdir():
            if config_dir.is_dir():
                config_name = config_dir.name
                json_file = config_dir / "verification_result.json"
                
                if json_file.exists():
                    print(f"読み込み中: {config_name}")
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        self.results_data[config_name] = data
                        print(f"  ✓ {config_name}: {len(data.get('episodes', []))} エピソード")
                    except Exception as e:
                        print(f"  ❌ {config_name}: エラー - {e}")
                else:
                    print(f"  ⚠️ {config_name}: JSONファイルが見つかりません")
        
        print(f"✓ 合計 {len(self.results_data)} 個の結果を読み込み完了\n")
    
    def extract_config_info(self, config_name: str) -> Tuple[str, float]:
        """Config名から設定情報を抽出"""
        parts = config_name.replace('Config_', '').split('_obstacle_')
        config_type = parts[0]  # A, B, C, D
        obstacle_density = float(parts[1]) if len(parts) > 1 else 0.0
        return config_type, obstacle_density
    
    def calculate_summary_statistics(self):
        """各Configの統計情報を計算"""
        print("=== 統計情報計算中 ===")
        
        for config_name, data in self.results_data.items():
            config_type, obstacle_density = self.extract_config_info(config_name)
            episodes = data.get('episodes', [])
            
            if not episodes:
                continue
            
            # 基本統計
            exploration_rates = [ep.get('final_exploration_rate', 0) for ep in episodes]
            steps_taken = [ep.get('steps_taken', 0) for ep in episodes]
            
            # 各エピソードの詳細統計
            episode_stats = []
            for ep in episodes:
                step_details = ep.get('step_details', [])
                if step_details:
                    rewards = [step.get('reward', 0) for step in step_details]
                    episode_stats.append({
                        'total_reward': sum(rewards),
                        'avg_reward': np.mean(rewards),
                        'max_reward': max(rewards),
                        'min_reward': min(rewards)
                    })
            
            # 統計サマリー
            stats = {
                'config_type': config_type,
                'obstacle_density': obstacle_density,
                'total_episodes': len(episodes),
                'exploration_rate': {
                    'mean': np.mean(exploration_rates),
                    'std': np.std(exploration_rates),
                    'min': np.min(exploration_rates),
                    'max': np.max(exploration_rates),
                    'median': np.median(exploration_rates)
                },
                'steps_taken': {
                    'mean': np.mean(steps_taken),
                    'std': np.std(steps_taken),
                    'min': np.min(steps_taken),
                    'max': np.max(steps_taken),
                    'median': np.median(steps_taken)
                }
            }
            
            if episode_stats:
                total_rewards = [es['total_reward'] for es in episode_stats]
                avg_rewards = [es['avg_reward'] for es in episode_stats]
                
                stats['total_reward'] = {
                    'mean': np.mean(total_rewards),
                    'std': np.std(total_rewards),
                    'min': np.min(total_rewards),
                    'max': np.max(total_rewards),
                    'median': np.median(total_rewards)
                }
                
                stats['avg_reward'] = {
                    'mean': np.mean(avg_rewards),
                    'std': np.std(avg_rewards),
                    'min': np.min(avg_rewards),
                    'max': np.max(avg_rewards),
                    'median': np.median(avg_rewards)
                }
            
            self.summary_stats[config_name] = stats
            print(f"  ✓ {config_name}: 探査率平均 {stats['exploration_rate']['mean']:.3f}")
        
        print("✓ 統計情報計算完了\n")
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """比較用のDataFrameを作成"""
        rows = []
        
        for config_name, stats in self.summary_stats.items():
            row = {
                'Config': config_name,
                'ConfigType': stats['config_type'],
                'ObstacleDensity': stats['obstacle_density'],
                'Episodes': stats['total_episodes'],
                'ExplorationRate_Mean': stats['exploration_rate']['mean'],
                'ExplorationRate_Std': stats['exploration_rate']['std'],
                'ExplorationRate_Min': stats['exploration_rate']['min'],
                'ExplorationRate_Max': stats['exploration_rate']['max'],
                'StepsTaken_Mean': stats['steps_taken']['mean'],
                'StepsTaken_Std': stats['steps_taken']['std']
            }
            
            if 'total_reward' in stats:
                row.update({
                    'TotalReward_Mean': stats['total_reward']['mean'],
                    'TotalReward_Std': stats['total_reward']['std'],
                    'AvgReward_Mean': stats['avg_reward']['mean'],
                    'AvgReward_Std': stats['avg_reward']['std']
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_visualizations(self, output_dir: str = "analysis_results"):
        """可視化グラフを生成"""
        print("=== 可視化グラフ生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        df = self.create_comparison_dataframe()
        
        if df.empty:
            print("❌ データが空のため、可視化をスキップします")
            return
        
        # 1. Exploration Rate Comparison (by Config)
        plt.figure(figsize=(15, 10))
        
        # 1-1. Average Exploration Rate by Config
        plt.subplot(2, 3, 1)
        config_means = df.groupby('ConfigType')['ExplorationRate_Mean'].mean()
        config_means.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Average Exploration Rate by Config')
        plt.xlabel('Config Type')
        plt.ylabel('Exploration Rate')
        plt.xticks(rotation=0)
        
        # 1-2. Average Exploration Rate by Obstacle Density
        plt.subplot(2, 3, 2)
        density_means = df.groupby('ObstacleDensity')['ExplorationRate_Mean'].mean()
        density_means.plot(kind='bar', color=['lightblue', 'orange', 'lightcoral'])
        plt.title('Average Exploration Rate by Obstacle Density')
        plt.xlabel('Obstacle Density')
        plt.ylabel('Exploration Rate')
        plt.xticks(rotation=0)
        
        # 1-3. Config×障害物密度のヒートマップ
        plt.subplot(2, 3, 3)
        pivot_data = df.pivot_table(values='ExplorationRate_Mean', 
                                   index='ConfigType', 
                                   columns='ObstacleDensity', 
                                   aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Exploration Rate Heatmap')
        
        # 1-4. Exploration Rate Distribution (Box Plot)
        plt.subplot(2, 3, 4)
        config_types = df['ConfigType'].unique()
        exploration_data = []
        labels = []
        
        for config_type in sorted(config_types):
            config_data = df[df['ConfigType'] == config_type]['ExplorationRate_Mean'].values
            exploration_data.append(config_data)
            labels.append(f'Config {config_type}')
        
        plt.boxplot(exploration_data, labels=labels)
        plt.title('Exploration Rate Distribution by Config')
        plt.ylabel('Exploration Rate')
        plt.xticks(rotation=45)
        
        # 1-5. Steps Comparison
        plt.subplot(2, 3, 5)
        step_means = df.groupby('ConfigType')['StepsTaken_Mean'].mean()
        step_means.plot(kind='bar', color=['lightsteelblue', 'lightpink', 'lightseagreen', 'lightyellow'])
        plt.title('Average Steps by Config')
        plt.xlabel('Config Type')
        plt.ylabel('Steps')
        plt.xticks(rotation=0)
        
        # 1-6. Exploration Rate vs Steps Scatter Plot
        plt.subplot(2, 3, 6)
        colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'}
        for config_type in df['ConfigType'].unique():
            config_df = df[df['ConfigType'] == config_type]
            plt.scatter(config_df['StepsTaken_Mean'], 
                       config_df['ExplorationRate_Mean'],
                       c=colors.get(config_type, 'gray'),
                       label=f'Config {config_type}',
                       alpha=0.7, s=100)
        
        plt.xlabel('Average Steps')
        plt.ylabel('Average Exploration Rate')
        plt.title('Exploration Rate vs Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/exploration_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Comparison Table
        if 'TotalReward_Mean' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # 2-1. Reward Comparison
            plt.subplot(2, 2, 1)
            reward_means = df.groupby('ConfigType')['TotalReward_Mean'].mean()
            reward_means.plot(kind='bar', color=['lightcyan', 'mistyrose', 'honeydew', 'lemonchiffon'])
            plt.title('Average Total Reward by Config')
            plt.xlabel('Config Type')
            plt.ylabel('Total Reward')
            plt.xticks(rotation=0)
            
            # 2-2. Reward Efficiency (Reward/Step)
            plt.subplot(2, 2, 2)
            df['RewardEfficiency'] = df['TotalReward_Mean'] / df['StepsTaken_Mean']
            efficiency_means = df.groupby('ConfigType')['RewardEfficiency'].mean()
            efficiency_means.plot(kind='bar', color=['lightblue', 'lightpink', 'lightgreen', 'lightyellow'])
            plt.title('Reward Efficiency by Config')
            plt.xlabel('Config Type')
            plt.ylabel('Reward/Step')
            plt.xticks(rotation=0)
            
            # 2-3. Exploration Rate vs Total Reward Scatter Plot
            plt.subplot(2, 2, 3)
            for config_type in df['ConfigType'].unique():
                config_df = df[df['ConfigType'] == config_type]
                plt.scatter(config_df['TotalReward_Mean'], 
                           config_df['ExplorationRate_Mean'],
                           c=colors.get(config_type, 'gray'),
                           label=f'Config {config_type}',
                           alpha=0.7, s=100)
            
            plt.xlabel('Average Total Reward')
            plt.ylabel('Average Exploration Rate')
            plt.title('Exploration Rate vs Total Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2-4. Impact of Obstacle Density
            plt.subplot(2, 2, 4)
            for density in sorted(df['ObstacleDensity'].unique()):
                density_df = df[df['ObstacleDensity'] == density]
                plt.plot(density_df['ConfigType'], 
                        density_df['ExplorationRate_Mean'], 
                        marker='o', label=f'Density {density}', linewidth=2, markersize=8)
            
            plt.xlabel('Config Type')
            plt.ylabel('Average Exploration Rate')
            plt.title('Impact of Obstacle Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/reward_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ 可視化グラフを {output_dir}/ に保存しました")
    
    def generate_summary_report(self, output_dir: str = "analysis_results"):
        """サマリーレポートを生成"""
        print("=== サマリーレポート生成中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        df = self.create_comparison_dataframe()
        
        report_path = f"{output_dir}/verification_summary_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# シミュレーション結果解析レポート\n\n")
            f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 概要
            f.write("## 1. 解析概要\n\n")
            f.write(f"- 解析対象Config数: {len(df)}\n")
            f.write(f"- Config種類: {', '.join(sorted(df['ConfigType'].unique()))}\n")
            f.write(f"- 障害物密度: {', '.join(map(str, sorted(df['ObstacleDensity'].unique())))}\n")
            f.write(f"- 総エピソード数: {df['Episodes'].sum()}\n\n")
            
            # 2. Config別性能ランキング
            f.write("## 2. Config別性能ランキング\n\n")
            
            # 探査率ランキング
            f.write("### 2.1 探査率ランキング（高い順）\n\n")
            exploration_ranking = df.groupby('ConfigType')['ExplorationRate_Mean'].mean().sort_values(ascending=False)
            for i, (config, rate) in enumerate(exploration_ranking.items(), 1):
                f.write(f"{i}. Config {config}: {rate:.4f}\n")
            f.write("\n")
            
            # 効率性ランキング（探査率/ステップ）
            f.write("### 2.2 効率性ランキング（探査率/ステップ）\n\n")
            df['Efficiency'] = df['ExplorationRate_Mean'] / df['StepsTaken_Mean']
            efficiency_ranking = df.groupby('ConfigType')['Efficiency'].mean().sort_values(ascending=False)
            for i, (config, eff) in enumerate(efficiency_ranking.items(), 1):
                f.write(f"{i}. Config {config}: {eff:.6f}\n")
            f.write("\n")
            
            # 3. 詳細統計表
            f.write("## 3. 詳細統計表\n\n")
            f.write("| Config | 障害物密度 | エピソード数 | 探査率(平均±標準偏差) | ステップ数(平均±標準偏差) |\n")
            f.write("|--------|------------|--------------|----------------------|------------------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['ConfigType']} | {row['ObstacleDensity']:.3f} | {row['Episodes']} | "
                       f"{row['ExplorationRate_Mean']:.4f}±{row['ExplorationRate_Std']:.4f} | "
                       f"{row['StepsTaken_Mean']:.1f}±{row['StepsTaken_Std']:.1f} |\n")
            f.write("\n")
            
            # 4. 統計的分析
            f.write("## 4. 統計的分析\n\n")
            
            # Config間の比較
            f.write("### 4.1 Config間の探査率比較\n\n")
            config_stats = df.groupby('ConfigType')['ExplorationRate_Mean'].agg(['mean', 'std', 'min', 'max'])
            for config in config_stats.index:
                stats = config_stats.loc[config]
                f.write(f"**Config {config}**\n")
                f.write(f"- 平均: {stats['mean']:.4f}\n")
                f.write(f"- 標準偏差: {stats['std']:.4f}\n")
                f.write(f"- 範囲: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
            
            # 障害物密度の影響
            f.write("### 4.2 障害物密度の影響\n\n")
            density_stats = df.groupby('ObstacleDensity')['ExplorationRate_Mean'].agg(['mean', 'std'])
            for density in density_stats.index:
                stats = density_stats.loc[density]
                f.write(f"**障害物密度 {density:.3f}**\n")
                f.write(f"- 平均探査率: {stats['mean']:.4f}±{stats['std']:.4f}\n\n")
            
            # 5. 推奨事項
            f.write("## 5. 推奨事項\n\n")
            
            best_config = exploration_ranking.index[0]
            best_rate = exploration_ranking.iloc[0]
            f.write(f"- **最高性能**: Config {best_config} (探査率: {best_rate:.4f})\n")
            
            most_efficient = efficiency_ranking.index[0]
            best_efficiency = efficiency_ranking.iloc[0]
            f.write(f"- **最高効率**: Config {most_efficient} (効率性: {best_efficiency:.6f})\n")
            
            # 障害物密度の影響
            if len(df['ObstacleDensity'].unique()) > 1:
                density_impact = df.groupby('ObstacleDensity')['ExplorationRate_Mean'].mean()
                best_density = density_impact.idxmax()
                f.write(f"- **最適障害物密度**: {best_density:.3f} (平均探査率: {density_impact[best_density]:.4f})\n")
            
            f.write("\n")
            
            # 6. 改善提案
            f.write("## 6. 改善提案\n\n")
            
            worst_config = exploration_ranking.index[-1]
            worst_rate = exploration_ranking.iloc[-1]
            improvement_potential = best_rate - worst_rate
            
            f.write(f"- Config {worst_config}の探査率は{worst_rate:.4f}で、最高性能のConfig {best_config}と比べて{improvement_potential:.4f}の改善余地があります。\n")
            f.write(f"- 学習パラメータの調整や分岐・統合戦略の見直しを検討してください。\n")
            f.write(f"- 障害物密度が性能に与える影響を詳しく分析し、環境適応性を向上させてください。\n\n")
        
        print(f"✓ サマリーレポートを {report_path} に保存しました")
        
        # CSVファイルも出力
        csv_path = f"{output_dir}/verification_results_summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ 詳細データを {csv_path} に保存しました")
    
    def run_analysis(self, output_dir: str = "analysis_results"):
        """完全な解析を実行"""
        print("🔍 シミュレーション結果解析を開始します\n")
        
        # 1. データ読み込み
        self.load_results()
        
        if not self.results_data:
            print("❌ 解析対象のデータが見つかりません")
            return
        
        # 2. 統計計算
        self.calculate_summary_statistics()
        
        # 3. 可視化
        self.generate_visualizations(output_dir)
        
        # 4. レポート生成
        self.generate_summary_report(output_dir)
        
        print(f"\n🎉 解析完了！結果は {output_dir}/ ディレクトリに保存されました")
        print(f"📊 グラフ: {output_dir}/exploration_analysis.png")
        print(f"📈 報酬分析: {output_dir}/reward_analysis.png")
        print(f"📝 レポート: {output_dir}/verification_summary_report.md")
        print(f"📄 データ: {output_dir}/verification_results_summary.csv")


def main():
    """メイン関数"""
    analyzer = VerificationResultAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 