#!/usr/bin/env python3
"""
高度なシミュレーション結果解析スクリプト
統計的検定、時系列分析、詳細な性能比較を含む包括的な解析を行います。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class AdvancedAnalyzer:
    """高度な解析クラス"""
    
    def __init__(self, results_dir: str = "verify_configs/verification_results"):
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.episode_data = pd.DataFrame()
        self.step_data = pd.DataFrame()
        
    def load_and_process_data(self):
        """データを読み込み、詳細な処理を行う"""
        print("=== 詳細データ処理中 ===")
        
        all_episodes = []
        all_steps = []
        
        for config_dir in self.results_dir.iterdir():
            if not config_dir.is_dir():
                continue
                
            config_name = config_dir.name
            json_file = config_dir / "verification_result.json"
            
            if not json_file.exists():
                continue
                
            print(f"処理中: {config_name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Config情報を抽出
                config_type, obstacle_density = self._extract_config_info(config_name)
                
                # エピソードデータを処理
                for episode in data.get('episodes', []):
                    episode_record = {
                        'config_name': config_name,
                        'config_type': config_type,
                        'obstacle_density': obstacle_density,
                        'episode_id': episode.get('episode', 0),
                        'final_exploration_rate': episode.get('final_exploration_rate', 0),
                        'steps_taken': episode.get('steps_taken', 0),
                        'steps_to_target': episode.get('steps_to_target'),
                    }
                    
                    # ステップ詳細データを処理
                    step_details = episode.get('step_details', [])
                    if step_details:
                        rewards = [step.get('reward', 0) for step in step_details]
                        exploration_rates = [step.get('exploration_rate', 0) for step in step_details]
                        
                        episode_record.update({
                            'total_reward': sum(rewards),
                            'avg_reward': np.mean(rewards),
                            'reward_std': np.std(rewards),
                            'final_reward': rewards[-1] if rewards else 0,
                            'exploration_progression': np.mean(np.diff(exploration_rates)) if len(exploration_rates) > 1 else 0,
                            'exploration_variance': np.var(exploration_rates),
                        })
                        
                        # 各ステップのデータ
                        for step_idx, step in enumerate(step_details):
                            step_record = {
                                'config_name': config_name,
                                'config_type': config_type,
                                'obstacle_density': obstacle_density,
                                'episode_id': episode.get('episode', 0),
                                'step_id': step_idx,
                                'exploration_rate': step.get('exploration_rate', 0),
                                'reward': step.get('reward', 0),
                                'done': step.get('done', False),
                                'truncated': step.get('truncated', False)
                            }
                            all_steps.append(step_record)
                    
                    all_episodes.append(episode_record)
                    
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                continue
        
        self.episode_data = pd.DataFrame(all_episodes)
        self.step_data = pd.DataFrame(all_steps)
        
        print(f"✓ エピソードデータ: {len(self.episode_data)} 件")
        print(f"✓ ステップデータ: {len(self.step_data)} 件\n")
    
    def _extract_config_info(self, config_name: str) -> Tuple[str, float]:
        """Config名から設定情報を抽出"""
        parts = config_name.replace('Config_', '').split('_obstacle_')
        config_type = parts[0]
        obstacle_density = float(parts[1]) if len(parts) > 1 else 0.0
        return config_type, obstacle_density
    
    def statistical_analysis(self, output_dir: str = "analysis_results"):
        """統計的分析を実行"""
        print("=== 統計的分析実行中 ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. ANOVA分析（Config間の有意差検定）
        print("1. ANOVA分析")
        config_groups = []
        config_names = []
        
        for config_type in self.episode_data['config_type'].unique():
            config_data = self.episode_data[self.episode_data['config_type'] == config_type]['final_exploration_rate']
            config_groups.append(config_data.values)
            config_names.append(config_type)
        
        if len(config_groups) > 1:
            f_stat, p_value = stats.f_oneway(*config_groups)
            print(f"  F統計量: {f_stat:.4f}")
            print(f"  p値: {p_value:.6f}")
            print(f"  有意差: {'あり' if p_value < 0.05 else 'なし'}")
        
        # 2. 対応のないt検定（ペアワイズ比較）
        print("\n2. ペアワイズt検定")
        t_test_results = []
        
        for i, config1 in enumerate(config_names):
            for config2 in config_names[i+1:]:
                data1 = self.episode_data[self.episode_data['config_type'] == config1]['final_exploration_rate']
                data2 = self.episode_data[self.episode_data['config_type'] == config2]['final_exploration_rate']
                
                t_stat, p_val = stats.ttest_ind(data1, data2)
                effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                
                t_test_results.append({
                    'config1': config1,
                    'config2': config2,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'significant': p_val < 0.05
                })
                
                print(f"  {config1} vs {config2}: t={t_stat:.3f}, p={p_val:.4f}, effect_size={effect_size:.3f}")
        
        # 結果をCSVで保存
        t_test_df = pd.DataFrame(t_test_results)
        t_test_df.to_csv(f"{output_dir}/statistical_tests.csv", index=False, encoding='utf-8')
        
        print("✓ 統計的分析完了\n")
        return t_test_results
    
    def time_series_analysis(self, output_dir: str = "analysis_results"):
        """時系列分析"""
        print("=== 時系列分析実行中 ===")
        
        if self.step_data.empty:
            print("❌ ステップデータが不足しています")
            return
        
        plt.figure(figsize=(16, 12))
        
        # 1. 探査率の時系列変化
        plt.subplot(3, 2, 1)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_exploration_by_step = config_steps.groupby('step_id')['exploration_rate'].mean()
            plt.plot(avg_exploration_by_step.index, avg_exploration_by_step.values, 
                    label=f'Config {config_type}', linewidth=2, marker='o', markersize=4)
        
        plt.title('Exploration Rate Change by Step')
        plt.xlabel('Step')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Reward Time Series Change
        plt.subplot(3, 2, 2)
        for config_type in self.step_data['config_type'].unique():
            config_steps = self.step_data[self.step_data['config_type'] == config_type]
            avg_reward_by_step = config_steps.groupby('step_id')['reward'].mean()
            plt.plot(avg_reward_by_step.index, avg_reward_by_step.values, 
                    label=f'Config {config_type}', linewidth=2, marker='s', markersize=4)
        
        plt.title('Reward Change by Step')
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Learning Curves Across Episodes
        plt.subplot(3, 2, 3)
        for config_type in self.episode_data['config_type'].unique():
            config_episodes = self.episode_data[self.episode_data['config_type'] == config_type]
            avg_exploration_by_episode = config_episodes.groupby('episode_id')['final_exploration_rate'].mean()
            plt.plot(avg_exploration_by_episode.index, avg_exploration_by_episode.values, 
                    label=f'Config {config_type}', linewidth=2, alpha=0.8)
        
        plt.title('Learning Curves Across Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Final Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Exploration Rate Variance Analysis
        plt.subplot(3, 2, 4)
        for config_type in self.episode_data['config_type'].unique():
            config_episodes = self.episode_data[self.episode_data['config_type'] == config_type]
            if 'exploration_variance' in config_episodes.columns:
                avg_variance_by_episode = config_episodes.groupby('episode_id')['exploration_variance'].mean()
                plt.plot(avg_variance_by_episode.index, avg_variance_by_episode.values, 
                        label=f'Config {config_type}', linewidth=2, linestyle='--')
        
        plt.title('Exploration Rate Variance Change')
        plt.xlabel('Episode')
        plt.ylabel('Exploration Rate Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Convergence Analysis
        plt.subplot(3, 2, 5)
        for config_type in self.episode_data['config_type'].unique():
            config_episodes = self.episode_data[self.episode_data['config_type'] == config_type]
            # Calculate moving average
            window_size = min(10, len(config_episodes) // 4)
            if window_size > 1:
                moving_avg = config_episodes['final_exploration_rate'].rolling(window=window_size).mean()
                plt.plot(config_episodes['episode_id'], moving_avg, 
                        label=f'Config {config_type} (Moving Avg)', linewidth=3, alpha=0.7)
        
        plt.title('Convergence Analysis (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Exploration Rate Moving Average')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Impact of Obstacle Density
        plt.subplot(3, 2, 6)
        density_colors = {0.0: 'blue', 0.003: 'orange', 0.005: 'red'}
        for density in self.episode_data['obstacle_density'].unique():
            density_data = self.episode_data[self.episode_data['obstacle_density'] == density]
            avg_exploration_by_config = density_data.groupby('config_type')['final_exploration_rate'].mean()
            plt.plot(avg_exploration_by_config.index, avg_exploration_by_config.values, 
                    color=density_colors.get(density, 'gray'), 
                    label=f'Density {density}', linewidth=2, marker='D', markersize=6)
        
        plt.title('Impact of Obstacle Density')
        plt.xlabel('Config Type')
        plt.ylabel('Average Exploration Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_series_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 時系列分析完了")
    
    def clustering_analysis(self, output_dir: str = "analysis_results"):
        """クラスタリング分析"""
        print("=== クラスタリング分析実行中 ===")
        
        # 特徴量を準備
        features = ['final_exploration_rate', 'steps_taken', 'obstacle_density']
        if 'total_reward' in self.episode_data.columns:
            features.extend(['total_reward', 'avg_reward'])
        
        # 欠損値を除去
        analysis_data = self.episode_data[features + ['config_type']].dropna()
        
        if len(analysis_data) < 10:
            print("❌ 分析に十分なデータがありません")
            return
        
        # 特徴量を標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(analysis_data[features])
        
        # PCA分析
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # K-meansクラスタリング
        optimal_k = min(4, len(analysis_data['config_type'].unique()))
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        plt.figure(figsize=(15, 5))
        
        # 1. PCAによる次元削減結果
        plt.subplot(1, 3, 1)
        config_colors = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'}
        for config_type in analysis_data['config_type'].unique():
            mask = analysis_data['config_type'] == config_type
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=config_colors.get(config_type, 'gray'),
                       label=f'Config {config_type}', alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA Analysis Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Clustering Results
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=50)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('K-means Clustering')
        plt.colorbar(scatter)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        plt.subplot(1, 3, 3)
        feature_importance = np.abs(pca.components_[0])
        plt.barh(features, feature_importance)
        plt.xlabel('Importance in PC1')
        plt.title('Feature Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # クラスタリング結果をCSVで保存
        analysis_data['cluster'] = clusters
        analysis_data['pc1'] = X_pca[:, 0]
        analysis_data['pc2'] = X_pca[:, 1]
        analysis_data.to_csv(f"{output_dir}/clustering_results.csv", index=False, encoding='utf-8')
        
        print("✓ クラスタリング分析完了")
    
    def generate_comprehensive_report(self, output_dir: str = "analysis_results"):
        """包括的なレポートを生成"""
        print("=== 包括的レポート生成中 ===")
        
        report_path = f"{output_dir}/comprehensive_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 包括的シミュレーション結果解析レポート\n\n")
            f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # データサマリー
            f.write("## データサマリー\n\n")
            f.write(f"- 総エピソード数: {len(self.episode_data)}\n")
            f.write(f"- 総ステップ数: {len(self.step_data)}\n")
            f.write(f"- Config種類: {', '.join(sorted(self.episode_data['config_type'].unique()))}\n")
            f.write(f"- 障害物密度: {', '.join(map(str, sorted(self.episode_data['obstacle_density'].unique())))}\n\n")
            
            # 基本統計
            f.write("## 基本統計\n\n")
            basic_stats = self.episode_data.groupby('config_type')['final_exploration_rate'].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(4)
            
            f.write("### Config別探査率統計\n\n")
            f.write("| Config | サンプル数 | 平均 | 標準偏差 | 最小値 | 最大値 | 中央値 |\n")
            f.write("|--------|-----------|------|----------|--------|--------|--------|\n")
            
            for config in basic_stats.index:
                stats = basic_stats.loc[config]
                f.write(f"| {config} | {stats['count']} | {stats['mean']:.4f} | "
                       f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |\n")
            
            f.write("\n")
            
            # 性能ランキング
            f.write("## 性能ランキング\n\n")
            ranking = self.episode_data.groupby('config_type')['final_exploration_rate'].mean().sort_values(ascending=False)
            
            f.write("### 探査率ランキング\n\n")
            for i, (config, rate) in enumerate(ranking.items(), 1):
                f.write(f"{i}. **Config {config}**: {rate:.4f}\n")
            
            f.write("\n")
            
            # 改善提案
            f.write("## 改善提案\n\n")
            best_config = ranking.index[0]
            worst_config = ranking.index[-1]
            improvement_gap = ranking.iloc[0] - ranking.iloc[-1]
            
            f.write(f"- **最高性能**: Config {best_config} ({ranking.iloc[0]:.4f})\n")
            f.write(f"- **最低性能**: Config {worst_config} ({ranking.iloc[-1]:.4f})\n")
            f.write(f"- **改善余地**: {improvement_gap:.4f} ({improvement_gap/ranking.iloc[-1]*100:.1f}%の向上可能性)\n\n")
            
            # 障害物密度の影響
            if len(self.episode_data['obstacle_density'].unique()) > 1:
                f.write("## 障害物密度の影響\n\n")
                density_impact = self.episode_data.groupby(['config_type', 'obstacle_density'])['final_exploration_rate'].mean().unstack()
                
                f.write("### Config×障害物密度の探査率\n\n")
                f.write("| Config |")
                for density in sorted(density_impact.columns):
                    f.write(f" 密度{density:.3f} |")
                f.write("\n|--------|")
                for _ in density_impact.columns:
                    f.write("----------|")
                f.write("\n")
                
                for config in density_impact.index:
                    f.write(f"| {config} |")
                    for density in sorted(density_impact.columns):
                        value = density_impact.loc[config, density]
                        if pd.notna(value):
                            f.write(f" {value:.4f} |")
                        else:
                            f.write(" N/A |")
                    f.write("\n")
                
                f.write("\n")
        
        print(f"✓ 包括的レポートを {report_path} に保存しました")
    
    def run_advanced_analysis(self, output_dir: str = "analysis_results"):
        """高度な解析を実行"""
        print("🔬 高度なシミュレーション結果解析を開始します\n")
        
        # データ読み込み・処理
        self.load_and_process_data()
        
        if self.episode_data.empty:
            print("❌ 解析対象のデータが見つかりません")
            return
        
        # 各種分析を実行
        self.statistical_analysis(output_dir)
        self.time_series_analysis(output_dir)
        self.clustering_analysis(output_dir)
        self.generate_comprehensive_report(output_dir)
        
        print(f"\n🎉 高度な解析完了！結果は {output_dir}/ ディレクトリに保存されました")
        print(f"📊 時系列分析: {output_dir}/time_series_analysis.png")
        print(f"🔍 クラスタリング分析: {output_dir}/clustering_analysis.png")
        print(f"📈 統計検定結果: {output_dir}/statistical_tests.csv")
        print(f"📝 包括的レポート: {output_dir}/comprehensive_analysis_report.md")


def main():
    """メイン関数"""
    analyzer = AdvancedAnalyzer()
    analyzer.run_advanced_analysis()


if __name__ == "__main__":
    main() 