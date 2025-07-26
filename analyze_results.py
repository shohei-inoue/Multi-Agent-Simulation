"""
結果解析スクリプト
学習結果と検証結果を比較・可視化する
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any
import glob

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'


class ResultAnalyzer:
    """結果解析クラス"""
    
    def __init__(self):
        self.training_results = {}
        self.verification_results = {}
        self.analysis_results = {}
        
    def load_training_results(self, results_dir: str = "training_results"):
        """学習結果を読み込み"""
        if not os.path.exists(results_dir):
            print(f"学習結果ディレクトリが存在しません: {results_dir}")
            return
        
        # 最新の学習結果ファイルを探す
        result_files = glob.glob(os.path.join(results_dir, "training_results_*.json"))
        if not result_files:
            print("学習結果ファイルが見つかりません")
            return
        
        latest_file = max(result_files, key=os.path.getctime)
        print(f"学習結果読み込み: {latest_file}")
        
        with open(latest_file, 'r') as f:
            self.training_results = json.load(f)
    
    def load_verification_results(self, results_dir: str = "verification_results"):
        """検証結果を読み込み"""
        if not os.path.exists(results_dir):
            print(f"検証結果ディレクトリが存在しません: {results_dir}")
            return
        
        # 最新の検証結果ファイルを探す
        result_files = glob.glob(os.path.join(results_dir, "verification_results_*.json"))
        if not result_files:
            print("検証結果ファイルが見つかりません")
            return
        
        latest_file = max(result_files, key=os.path.getctime)
        print(f"検証結果読み込み: {latest_file}")
        
        with open(latest_file, 'r') as f:
            self.verification_results = json.load(f)
    
    def analyze_performance_comparison(self) -> Dict[str, Any]:
        """性能比較分析"""
        if not self.verification_results:
            print("検証結果が読み込まれていません")
            return {}
        
        # 基本環境での性能比較（障害物なし）
        base_env = "Map100x200_Obs0.0_Robot20"
        performance_data = {}
        
        for exp_name, exp_results in self.verification_results.items():
            if base_env in exp_results:
                base_result = exp_results[base_env]
                performance_data[exp_name] = {
                    'target_reach_rate': base_result['target_reach_rate'],
                    'avg_steps_to_target': base_result['avg_steps_to_target'],
                    'avg_exploration_speed': base_result['avg_exploration_speed'],
                    'avg_final_exploration_rate': base_result['avg_final_exploration_rate']
                }
        
        return performance_data
    
    def analyze_robustness(self) -> Dict[str, Any]:
        """ロバスト性分析"""
        if not self.verification_results:
            return {}
        
        robustness_data = {}
        
        for exp_name, exp_results in self.verification_results.items():
            env_performances = {}
            
            for env_name, result in exp_results.items():
                env_performances[env_name] = result['target_reach_rate']
            
            # 環境変化に対する性能の安定性
            performance_std = np.std(list(env_performances.values()))
            robustness_data[exp_name] = {
                'performance_std': performance_std,
                'env_performances': env_performances,
                'stability_score': 1.0 / (1.0 + performance_std)
            }
        
        return robustness_data
    
    def analyze_obstacle_density_impact(self) -> Dict[str, Any]:
        """障害物密度の影響分析"""
        if not self.verification_results:
            return {}
        
        density_impact = {}
        
        for exp_name, exp_results in self.verification_results.items():
            density_performances = {}
            for env_name, result in exp_results.items():
                # 環境名から障害物密度を抽出
                if "Obs0.0" in env_name:
                    density = 0.0
                elif "Obs0.003" in env_name:
                    density = 0.003
                elif "Obs0.005" in env_name:
                    density = 0.005
                else:
                    continue
                
                density_performances[density] = result['target_reach_rate']
            
            # 障害物密度による性能変化を分析
            if len(density_performances) >= 2:
                densities = sorted(density_performances.keys())
                performances = [density_performances[d] for d in densities]
                
                # 性能劣化率（障害物密度0.0を基準）
                if 0.0 in density_performances:
                    baseline = density_performances[0.0]
                    degradation_rates = []
                    for density, perf in density_performances.items():
                        if density > 0.0:
                            degradation = (baseline - perf) / baseline if baseline > 0 else 0
                            degradation_rates.append(degradation)
                    
                    avg_degradation = np.mean(degradation_rates) if degradation_rates else 0.0
                else:
                    avg_degradation = 0.0
                
                density_impact[exp_name] = {
                    'density_performances': density_performances,
                    'avg_degradation_rate': avg_degradation,
                    'performance_trend': performances
                }
        
        return density_impact
    
    def create_performance_comparison_plot(self, output_dir: str = "analysis_results"):
        """性能比較グラフ作成"""
        performance_data = self.analyze_performance_comparison()
        if not performance_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('性能比較結果', fontsize=16)
        
        configs = list(performance_data.keys())
        
        # 目標達成率
        target_reach_rates = [performance_data[config]['target_reach_rate'] for config in configs]
        axes[0, 0].bar(configs, target_reach_rates, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('目標達成率（80%探査）')
        axes[0, 0].set_ylabel('達成率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 平均ステップ数
        avg_steps = [performance_data[config]['avg_steps_to_target'] or 0 for config in configs]
        axes[0, 1].bar(configs, avg_steps, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('平均ステップ数（目標達成時）')
        axes[0, 1].set_ylabel('ステップ数')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 探査速度
        exploration_speeds = [performance_data[config]['avg_exploration_speed'] for config in configs]
        axes[1, 0].bar(configs, exploration_speeds, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('探査進捗速度')
        axes[1, 0].set_ylabel('速度')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 最終探査率
        final_rates = [performance_data[config]['avg_final_exploration_rate'] for config in configs]
        axes[1, 1].bar(configs, final_rates, alpha=0.7, color='gold')
        axes[1, 1].set_title('最終探査率')
        axes[1, 1].set_ylabel('探査率')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'performance_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_robustness_plot(self, output_dir: str = "analysis_results"):
        """ロバスト性比較グラフ作成"""
        robustness_data = self.analyze_robustness()
        if not robustness_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        configs = list(robustness_data.keys())
        stds = [robustness_data[config]['performance_std'] for config in configs]
        stability_scores = [robustness_data[config]['stability_score'] for config in configs]
        
        # 性能標準偏差
        ax1.bar(configs, stds, alpha=0.7, color='orange')
        ax1.set_title('ロバスト性比較（性能標準偏差）')
        ax1.set_ylabel('標準偏差')
        ax1.set_xlabel('構成')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 安定性スコア
        ax2.bar(configs, stability_scores, alpha=0.7, color='purple')
        ax2.set_title('安定性スコア')
        ax2.set_ylabel('安定性スコア')
        ax2.set_xlabel('構成')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'robustness_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_obstacle_density_impact_plot(self, output_dir: str = "analysis_results"):
        """障害物密度影響グラフ作成"""
        density_data = self.analyze_obstacle_density_impact()
        if not density_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 各構成の障害物密度による性能変化をプロット
        densities = [0.0, 0.003, 0.005]
        
        for config_name, config_data in density_data.items():
            performances = []
            for density in densities:
                if density in config_data['density_performances']:
                    performances.append(config_data['density_performances'][density])
                else:
                    performances.append(0.0)
            
            ax.plot(densities, performances, marker='o', label=config_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('障害物密度')
        ax.set_ylabel('目標達成率')
        ax.set_title('障害物密度による性能への影響')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f'obstacle_density_impact_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self, output_dir: str = "analysis_results"):
        """結果サマリーテーブル作成"""
        performance_data = self.analyze_performance_comparison()
        robustness_data = self.analyze_robustness()
        density_data = self.analyze_obstacle_density_impact()
        
        if not performance_data:
            return
        
        # データフレーム作成
        summary_data = []
        for config in performance_data.keys():
            row = {
                '構成': config,
                '目標達成率': f"{performance_data[config]['target_reach_rate']:.3f}",
                '平均ステップ数': f"{performance_data[config]['avg_steps_to_target']:.1f}" if performance_data[config]['avg_steps_to_target'] else "N/A",
                '探査速度': f"{performance_data[config]['avg_exploration_speed']:.4f}",
                '最終探査率': f"{performance_data[config]['avg_final_exploration_rate']:.3f}"
            }
            
            if config in robustness_data:
                row['性能標準偏差'] = f"{robustness_data[config]['performance_std']:.3f}"
                row['安定性スコア'] = f"{robustness_data[config]['stability_score']:.3f}"
            
            if config in density_data:
                row['平均性能劣化率'] = f"{density_data[config]['avg_degradation_rate']:.3f}"
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # テーブル保存
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV保存
        csv_path = os.path.join(output_dir, f'summary_table_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # HTML保存（見やすい形式）
        html_path = os.path.join(output_dir, f'summary_table_{timestamp}.html')
        html_content = f"""
        <html>
        <head>
            <title>実験結果サマリー</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>実験結果サマリー</h1>
            <p>生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"サマリーテーブル保存: {csv_path}, {html_path}")
        return df
    
    def run_analysis(self):
        """全解析実行"""
        print("=== 結果解析開始 ===")
        
        # 結果読み込み
        self.load_training_results()
        self.load_verification_results()
        
        # 解析実行
        performance_data = self.analyze_performance_comparison()
        robustness_data = self.analyze_robustness()
        density_data = self.analyze_obstacle_density_impact()
        
        # 結果表示
        print("\n=== 性能比較 ===")
        for config, perf in performance_data.items():
            print(f"{config}:")
            print(f"  目標達成率: {perf['target_reach_rate']:.3f}")
            print(f"  平均ステップ数: {perf['avg_steps_to_target']:.1f}")
            print(f"  探査速度: {perf['avg_exploration_speed']:.4f}")
            print(f"  最終探査率: {perf['avg_final_exploration_rate']:.3f}")
        
        print("\n=== ロバスト性分析 ===")
        for config, robust in robustness_data.items():
            print(f"{config}: 性能標準偏差 = {robust['performance_std']:.3f}, 安定性スコア = {robust['stability_score']:.3f}")
        
        print("\n=== 障害物密度影響 ===")
        for config, density_impact in density_data.items():
            print(f"{config}: 平均性能劣化率 = {density_impact['avg_degradation_rate']:.3f}")
        
        # グラフ作成
        print("\n=== グラフ作成 ===")
        self.create_performance_comparison_plot()
        self.create_robustness_plot()
        self.create_obstacle_density_impact_plot()
        
        # サマリーテーブル作成
        print("\n=== サマリーテーブル作成 ===")
        self.create_summary_table()
        
        print("\n=== 解析完了 ===")


def main():
    """メイン実行関数"""
    analyzer = ResultAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 