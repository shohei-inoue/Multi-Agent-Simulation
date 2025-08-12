#!/usr/bin/env python3
"""
探査率向上スピード分析スクリプト
Config A、B、C、Dの探査効率を測定・比較する
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse
from scipy import stats
from scipy.interpolate import interp1d

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Hiragino Sans', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

class ExplorationSpeedAnalyzer:
    """探査速度分析クラス"""
    
    def __init__(self, results_dir: str = "verify_configs/verification_results"):
        self.results_dir = Path(results_dir)
        self.configs = ['A', 'B', 'C', 'D']
        self.obstacle_densities = [0.0, 0.003, 0.005]
        self.output_dir = Path("exploration_speed_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_config_data(self, config: str, obstacle_density: float) -> Optional[Dict]:
        """指定ConfigとObstacle密度のデータを読み込み"""
        config_dir = self.results_dir / f"Config_{config}_obstacle_{obstacle_density}"
        
        if not config_dir.exists():
            print(f"⚠️ ディレクトリが見つかりません: {config_dir}")
            return None
            
        # 結果JSONファイルを探す
        json_files = list(config_dir.glob("*.json"))
        if not json_files:
            print(f"⚠️ JSONファイルが見つかりません: {config_dir}")
            return None
            
        # 最新のファイルを使用
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ データ読み込み: {latest_file.name}")
            return data
        except Exception as e:
            print(f"❌ データ読み込みエラー: {latest_file} - {e}")
            return None
    
    def extract_exploration_progress(self, episode_data: Dict) -> Tuple[List[int], List[float]]:
        """エピソードデータから探査進捗を抽出"""
        steps = []
        exploration_rates = []
        
        if 'step_data' in episode_data:
            for step_info in episode_data['step_data']:
                if isinstance(step_info, dict):
                    step = step_info.get('step', 0)
                    exploration_rate = step_info.get('exploration_rate', 0.0)
                    steps.append(step)
                    exploration_rates.append(exploration_rate * 100)  # パーセント表示
                    
        return steps, exploration_rates
    
    def calculate_exploration_speed_metrics(self, steps: List[int], exploration_rates: List[float]) -> Dict[str, float]:
        """探査速度メトリクスを計算"""
        if len(steps) < 2 or len(exploration_rates) < 2:
            return {
                'avg_speed': 0.0,
                'max_speed': 0.0,
                'time_to_50': float('inf'),
                'time_to_80': float('inf'),
                'final_rate': 0.0,
                'acceleration': 0.0
            }
        
        # 探査速度計算（探査率の変化率）
        speeds = []
        for i in range(1, len(exploration_rates)):
            dt = steps[i] - steps[i-1]
            if dt > 0:
                speed = (exploration_rates[i] - exploration_rates[i-1]) / dt
                speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0
        
        # 目標探査率到達時間
        time_to_50 = self._find_time_to_target(steps, exploration_rates, 50.0)
        time_to_80 = self._find_time_to_target(steps, exploration_rates, 80.0)
        
        final_rate = exploration_rates[-1] if exploration_rates else 0.0
        
        # 加速度計算（速度の変化率）
        acceleration = 0.0
        if len(speeds) > 1:
            mid_point = len(speeds) // 2
            early_speed = np.mean(speeds[:mid_point])
            late_speed = np.mean(speeds[mid_point:])
            acceleration = late_speed - early_speed
        
        return {
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'time_to_50': time_to_50,
            'time_to_80': time_to_80,
            'final_rate': final_rate,
            'acceleration': acceleration
        }
    
    def _find_time_to_target(self, steps: List[int], exploration_rates: List[float], target: float) -> float:
        """目標探査率に到達する時間を計算"""
        for i, rate in enumerate(exploration_rates):
            if rate >= target:
                return steps[i]
        return float('inf')
    
    def analyze_all_configs(self) -> pd.DataFrame:
        """全Config・全Obstacle密度のデータを分析"""
        results = []
        
        for config in self.configs:
            for obstacle_density in self.obstacle_densities:
                print(f"\n📊 分析中: Config {config}, Obstacle {obstacle_density}")
                
                data = self.load_config_data(config, obstacle_density)
                if not data:
                    continue
                
                # 各エピソードの分析
                episode_metrics = []
                for episode_key, episode_data in data.items():
                    if episode_key.startswith('episode_'):
                        steps, exploration_rates = self.extract_exploration_progress(episode_data)
                        if steps and exploration_rates:
                            metrics = self.calculate_exploration_speed_metrics(steps, exploration_rates)
                            episode_metrics.append(metrics)
                
                if episode_metrics:
                    # 統計値計算
                    avg_metrics = {}
                    for key in episode_metrics[0].keys():
                        values = [m[key] for m in episode_metrics if m[key] != float('inf')]
                        if values:
                            avg_metrics[f'{key}_mean'] = np.mean(values)
                            avg_metrics[f'{key}_std'] = np.std(values)
                            avg_metrics[f'{key}_median'] = np.median(values)
                        else:
                            avg_metrics[f'{key}_mean'] = 0.0
                            avg_metrics[f'{key}_std'] = 0.0
                            avg_metrics[f'{key}_median'] = 0.0
                    
                    result = {
                        'config': config,
                        'obstacle_density': obstacle_density,
                        'episode_count': len(episode_metrics),
                        **avg_metrics
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def create_speed_comparison_plots(self, df: pd.DataFrame):
        """探査速度比較プロットを作成"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('探査速度分析 - Config比較', fontsize=16, fontweight='bold')
        
        metrics = [
            ('avg_speed_mean', '平均探査速度 (%/step)'),
            ('max_speed_mean', '最大探査速度 (%/step)'),
            ('time_to_50_mean', '50%到達時間 (steps)'),
            ('time_to_80_mean', '80%到達時間 (steps)'),
            ('final_rate_mean', '最終探査率 (%)'),
            ('acceleration_mean', '探査加速度 (%/step²)')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            # 各Obstacle密度でのConfig比較
            for obstacle_density in self.obstacle_densities:
                subset = df[df['obstacle_density'] == obstacle_density]
                if not subset.empty:
                    x_pos = np.arange(len(subset))
                    values = subset[metric].values
                    errors = subset[f"{metric.replace('_mean', '_std')}"].values
                    
                    ax.bar(x_pos + obstacle_density * 0.25, values, 
                          width=0.25, label=f'Obstacle {obstacle_density}',
                          alpha=0.8, yerr=errors, capsize=5)
                    
                    # Config名をx軸に設定
                    ax.set_xticks(x_pos + 0.25)
                    ax.set_xticklabels(subset['config'])
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Config')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 無限値の処理
            if 'time_to' in metric:
                ax.set_ylim(0, min(1000, ax.get_ylim()[1]))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exploration_speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_analysis_plots(self, df: pd.DataFrame):
        """詳細分析プロット"""
        # 1. 探査効率ヒートマップ
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 平均探査速度ヒートマップ
        pivot_speed = df.pivot(index='config', columns='obstacle_density', values='avg_speed_mean')
        sns.heatmap(pivot_speed, annot=True, fmt='.4f', cmap='YlOrRd', 
                   ax=axes[0], cbar_kws={'label': '平均探査速度 (%/step)'})
        axes[0].set_title('平均探査速度ヒートマップ', fontweight='bold')
        
        # 最終探査率ヒートマップ
        pivot_final = df.pivot(index='config', columns='obstacle_density', values='final_rate_mean')
        sns.heatmap(pivot_final, annot=True, fmt='.1f', cmap='YlGnBu', 
                   ax=axes[1], cbar_kws={'label': '最終探査率 (%)'})
        axes[1].set_title('最終探査率ヒートマップ', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exploration_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 統計的有意差検定
        self.perform_statistical_tests(df)
    
    def perform_statistical_tests(self, df: pd.DataFrame):
        """統計的有意差検定"""
        print("\n📈 統計的有意差検定")
        print("=" * 50)
        
        # Config間の平均探査速度比較（ANOVA）
        config_groups = []
        for config in self.configs:
            config_data = df[df['config'] == config]['avg_speed_mean'].values
            if len(config_data) > 0:
                config_groups.append(config_data)
        
        if len(config_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*config_groups)
            print(f"Config間の平均探査速度 ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print("✓ Config間に有意差あり (p < 0.05)")
                
                # ペアワイズt検定
                from itertools import combinations
                for i, j in combinations(range(len(self.configs)), 2):
                    if i < len(config_groups) and j < len(config_groups):
                        t_stat, t_p = stats.ttest_ind(config_groups[i], config_groups[j])
                        print(f"  {self.configs[i]} vs {self.configs[j]}: t={t_stat:.4f}, p={t_p:.4f}")
            else:
                print("✗ Config間に有意差なし (p >= 0.05)")
    
    def create_time_series_analysis(self):
        """時系列分析（詳細な探査進捗）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('探査進捗の時系列分析', fontsize=16, fontweight='bold')
        
        for idx, config in enumerate(self.configs):
            ax = axes[idx // 2, idx % 2]
            
            for obstacle_density in self.obstacle_densities:
                data = self.load_config_data(config, obstacle_density)
                if not data:
                    continue
                
                # 全エピソードの平均進捗を計算
                all_progress = []
                max_steps = 0
                
                for episode_key, episode_data in data.items():
                    if episode_key.startswith('episode_'):
                        steps, exploration_rates = self.extract_exploration_progress(episode_data)
                        if steps and exploration_rates:
                            all_progress.append((steps, exploration_rates))
                            max_steps = max(max_steps, max(steps))
                
                if all_progress:
                    # 共通のステップ軸で補間
                    common_steps = np.linspace(0, min(200, max_steps), 100)
                    interpolated_rates = []
                    
                    for steps, rates in all_progress:
                        if len(steps) > 1 and len(rates) > 1:
                            interp_func = interp1d(steps, rates, kind='linear', 
                                                 bounds_error=False, fill_value='extrapolate')
                            interpolated_rates.append(interp_func(common_steps))
                    
                    if interpolated_rates:
                        mean_rates = np.mean(interpolated_rates, axis=0)
                        std_rates = np.std(interpolated_rates, axis=0)
                        
                        ax.plot(common_steps, mean_rates, 
                               label=f'Obstacle {obstacle_density}', linewidth=2)
                        ax.fill_between(common_steps, 
                                       mean_rates - std_rates,
                                       mean_rates + std_rates,
                                       alpha=0.2)
            
            ax.set_title(f'Config {config}', fontweight='bold')
            ax.set_xlabel('Steps')
            ax.set_ylabel('探査率 (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exploration_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, df: pd.DataFrame):
        """分析レポートを生成"""
        report_path = self.output_dir / 'exploration_speed_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 探査速度分析レポート\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 概要\n\n")
            f.write("Config A、B、C、Dの探査効率を比較分析した結果です。\n\n")
            
            f.write("## Config設定\n\n")
            f.write("| Config | SystemAgent学習 | SwarmAgent学習 | 分岐・統合 |\n")
            f.write("|--------|----------------|----------------|------------|\n")
            f.write("| A      | なし           | なし           | なし       |\n")
            f.write("| B      | なし           | あり           | なし       |\n")
            f.write("| C      | あり           | なし           | あり       |\n")
            f.write("| D      | あり           | あり           | あり       |\n\n")
            
            f.write("## 主要メトリクス\n\n")
            
            # 最高性能Config
            best_speed_config = df.loc[df['avg_speed_mean'].idxmax(), 'config']
            best_final_config = df.loc[df['final_rate_mean'].idxmax(), 'config']
            
            f.write(f"- **最高平均探査速度**: Config {best_speed_config}\n")
            f.write(f"- **最高最終探査率**: Config {best_final_config}\n\n")
            
            f.write("## 詳細結果\n\n")
            f.write(df.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n")
            
            f.write("## 結論\n\n")
            f.write("1. **探査速度**: 学習の有無とアルゴリズムの組み合わせによる効果\n")
            f.write("2. **環境適応性**: 障害物密度に対する各Configの頑健性\n")
            f.write("3. **学習効果**: 学習ありConfigの探査効率向上\n")
            f.write("4. **分岐・統合効果**: 複数群による探査範囲拡大の影響\n\n")
            
            f.write("## 生成ファイル\n\n")
            f.write("- `exploration_speed_comparison.png`: 基本比較グラフ\n")
            f.write("- `exploration_heatmaps.png`: ヒートマップ分析\n")
            f.write("- `exploration_time_series.png`: 時系列進捗分析\n")
            f.write("- `exploration_speed_data.csv`: 生データ\n")
        
        # CSVファイルも保存
        df.to_csv(self.output_dir / 'exploration_speed_data.csv', index=False)
        
        print(f"\n📄 レポート生成完了: {report_path}")
        print(f"📊 データ保存完了: {self.output_dir}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='探査速度分析スクリプト')
    parser.add_argument('--results-dir', default='verify_configs/verification_results',
                       help='結果ディレクトリのパス')
    parser.add_argument('--output-dir', default='exploration_speed_analysis',
                       help='出力ディレクトリのパス')
    
    args = parser.parse_args()
    
    print("🚀 探査速度分析開始")
    print("=" * 50)
    
    analyzer = ExplorationSpeedAnalyzer(args.results_dir)
    analyzer.output_dir = Path(args.output_dir)
    analyzer.output_dir.mkdir(exist_ok=True)
    
    # データ分析
    print("📊 データ分析中...")
    df = analyzer.analyze_all_configs()
    
    if df.empty:
        print("❌ 分析可能なデータが見つかりません")
        return
    
    print(f"✓ {len(df)} 件のデータを分析")
    
    # プロット生成
    print("📈 グラフ生成中...")
    analyzer.create_speed_comparison_plots(df)
    analyzer.create_detailed_analysis_plots(df)
    analyzer.create_time_series_analysis()
    
    # レポート生成
    print("📝 レポート生成中...")
    analyzer.generate_report(df)
    
    print("\n🎉 分析完了!")
    print(f"結果は {analyzer.output_dir} に保存されました")

if __name__ == "__main__":
    main() 