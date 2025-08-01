#!/usr/bin/env python3
"""
Config_A、B、C、Dの比較分析スクリプト
障害物密度0.003の結果を比較
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import os

# 日本語フォント設定
import matplotlib.font_manager as fm

# 利用可能な日本語フォントを検索
japanese_fonts = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
available_font = None

for font in japanese_fonts:
    try:
        fm.findfont(font)
        available_font = font
        break
    except:
        continue

if available_font:
    plt.rcParams['font.family'] = available_font
    print(f"使用フォント: {available_font}")
else:
    # フォールバック設定
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    print("警告: 日本語フォントが見つかりませんでした。フォールバック設定を使用します。")

plt.rcParams['axes.unicode_minus'] = False

def load_verification_result(file_path: str) -> Dict[str, Any]:
    """検証結果ファイルを読み込み"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_step_data(config_name: str, episode: int, log_dir: str) -> pd.DataFrame:
    """エピソードごとのstepデータを読み込み"""
    csv_path = os.path.join(log_dir, "csvs", f"episode_{episode:04d}_exploration.csv")
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"警告: {csv_path} が見つかりません")
        return pd.DataFrame()

def calculate_exploration_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """探査率の詳細メトリクスを計算"""
    if df.empty:
        return {}
    
    # 探査率の変化率を計算
    df['exploration_rate_change'] = df['exploration_rate'].diff()
    df['exploration_rate_change_rate'] = df['exploration_rate_change'] / df['exploration_rate'].shift(1)
    df['exploration_rate_change_rate'] = df['exploration_rate_change_rate'].fillna(0)
    
    # 新しく探査されたエリア数を計算
    df['new_explored_area'] = df['explored_area'].diff()
    df['new_explored_area'] = df['new_explored_area'].fillna(0)
    
    # 探査効率（新しく探査されたエリア数/ステップ）
    df['exploration_efficiency'] = df['new_explored_area'] / df['total_area']
    
    # 目標探査率（例：0.8）への到達速度
    target_rate = 0.8
    target_reached_steps = df[df['exploration_rate'] >= target_rate]
    target_reaching_speed = len(target_reached_steps) if not target_reached_steps.empty else None
    
    # 探査率の一貫性（標準偏差）
    exploration_consistency = df['exploration_rate'].std()
    
    # 平均探査効率
    avg_exploration_efficiency = df['exploration_efficiency'].mean()
    
    # 探査率の最大増加率
    max_exploration_increase = df['exploration_rate_change'].max()
    
    return {
        'step_data': df,
        'target_reaching_speed': target_reaching_speed,
        'exploration_consistency': exploration_consistency,
        'avg_exploration_efficiency': avg_exploration_efficiency,
        'max_exploration_increase': max_exploration_increase,
        'total_steps': len(df),
        'final_exploration_rate': df['exploration_rate'].iloc[-1] if not df.empty else 0.0
    }

def calculate_summary_stats(episodes: List[Dict]) -> Dict[str, Any]:
    """エピソードデータから統計を計算"""
    exploration_rates = [ep['final_exploration_rate'] for ep in episodes]
    steps_taken = [ep['steps_taken'] for ep in episodes]
    target_reached = [ep['steps_to_target'] is not None for ep in episodes]
    
    return {
        'total_episodes': len(episodes),
        'target_reached_episodes': sum(target_reached),
        'average_exploration_rate': np.mean(exploration_rates),
        'average_steps_taken': np.mean(steps_taken),
        'std_exploration_rate': np.std(exploration_rates),
        'std_steps_taken': np.std(steps_taken),
        'min_exploration_rate': np.min(exploration_rates),
        'max_exploration_rate': np.max(exploration_rates)
    }

def analyze_configs():
    """Config_A、B、C、Dの比較分析"""
    
    # 結果ファイルのパス
    config_files = {
        'Config_A': 'verification_results/Config_A_obstacle_0.003/verification_result.json',
        'Config_B': 'verification_results/Config_B_obstacle_0.003/verification_result.json',
        'Config_C': 'verification_results/Config_E_obstacle_0.003/verification_result.json',  # Config_CはEに保存されている
        'Config_D': 'verification_results/Config_D_obstacle_0.003/verification_result.json'
    }
    
    # 出力ディレクトリを作成
    output_dir = "config_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 各Configの結果を読み込み
    results = {}
    step_analysis_results = {}
    
    for config_name, file_path in config_files.items():
        try:
            data = load_verification_result(file_path)
            episodes = data['episodes']
            
            # summaryが存在しない場合は計算
            if 'summary' in data:
                summary = data['summary']
                # 不足している統計値を追加
                if 'min_exploration_rate' not in summary:
                    exploration_rates = [ep['final_exploration_rate'] for ep in episodes]
                    summary['min_exploration_rate'] = np.min(exploration_rates)
                    summary['max_exploration_rate'] = np.max(exploration_rates)
            else:
                summary = calculate_summary_stats(episodes)
            
            # Config_Cの場合は、実際のConfig名をEからCに変更
            display_name = config_name
            if config_name == 'Config_C':
                display_name = 'Config_C'  # 表示用の名前をCに統一
            
            results[display_name] = {
                'episodes': episodes,
                'summary': summary,
                'environment': data['environment']
            }
            
            # Stepごとの詳細分析
            step_analysis = analyze_step_data(display_name, episodes, data['environment'])
            step_analysis_results[display_name] = step_analysis
            
            print(f"✓ {display_name} データ読み込み完了")
            
        except Exception as e:
            print(f"❌ {config_name} データ読み込みエラー: {e}")
    
    # 比較表を作成
    print("\n=== Config比較結果 (障害物密度: 0.003) ===")
    print(f"{'Config':<10} {'平均探査率':<12} {'標準偏差':<12} {'目標達成率':<12} {'最小値':<10} {'最大値':<10}")
    print("-" * 80)
    
    comparison_data = []
    for config_name, data in results.items():
        summary = data['summary']
        avg_rate = summary['average_exploration_rate']
        std_rate = summary['std_exploration_rate']
        target_rate = summary['target_reached_episodes'] / summary['total_episodes'] * 100
        min_rate = summary['min_exploration_rate']
        max_rate = summary['max_exploration_rate']
        
        print(f"{config_name:<10} {avg_rate:.3f}±{std_rate:.3f} {'':<4} {target_rate:.1f}% {'':<8} {min_rate:.3f} {'':<7} {max_rate:.3f}")
        
        comparison_data.append({
            'Config': config_name,
            'Average_Exploration_Rate': avg_rate,
            'Std_Exploration_Rate': std_rate,
            'Target_Achievement_Rate': target_rate,
            'Min_Exploration_Rate': min_rate,
            'Max_Exploration_Rate': max_rate
        })
    
    # CSVファイルとして保存
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, "config_comparison_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✓ 比較結果を保存: {csv_path}")
    
    # 可視化
    create_comparison_plots(results, output_dir)
    
    # Stepごとの詳細分析結果を可視化
    create_step_analysis_plots(step_analysis_results, output_dir)
    
    return results, step_analysis_results

def analyze_step_data(config_name: str, episodes: List[Dict], environment: Dict) -> Dict[str, Any]:
    """Stepごとの詳細分析"""
    step_metrics = {
        'config_name': config_name,
        'episodes': [],
        'avg_exploration_curves': [],
        'exploration_efficiency': [],
        'target_reaching_speeds': []
    }
    
    # 各エピソードのstepデータを分析
    for episode in episodes:
        episode_num = episode.get('episode', 0)
        
        # ログディレクトリを特定
        log_dir = f"verification_results/{config_name}_obstacle_0.003"
        
        # Stepデータを読み込み
        step_df = load_step_data(config_name, episode_num, log_dir)
        
        if not step_df.empty:
            # 探査メトリクスを計算
            metrics = calculate_exploration_metrics(step_df)
            
            step_metrics['episodes'].append({
                'episode': episode_num,
                'step_data': step_df,
                'metrics': metrics
            })
            
            # 平均値の計算用データを収集
            if 'step_data' in metrics and not metrics['step_data'].empty:
                step_metrics['avg_exploration_curves'].append(metrics['step_data']['exploration_rate'].values)
                step_metrics['exploration_efficiency'].append(metrics['avg_exploration_efficiency'])
                if metrics['target_reaching_speed'] is not None:
                    step_metrics['target_reaching_speeds'].append(metrics['target_reaching_speed'])
    
    # 平均値を計算
    if step_metrics['avg_exploration_curves']:
        # 最大ステップ数に合わせて正規化
        max_steps = max(len(curve) for curve in step_metrics['avg_exploration_curves'])
        normalized_curves = []
        
        for curve in step_metrics['avg_exploration_curves']:
            if len(curve) < max_steps:
                # 最後の値を繰り返して最大ステップ数に合わせる
                extended_curve = np.pad(curve, (0, max_steps - len(curve)), mode='edge')
                normalized_curves.append(extended_curve)
            else:
                normalized_curves.append(curve)
        
        step_metrics['avg_exploration_curve'] = np.mean(normalized_curves, axis=0)
        step_metrics['std_exploration_curve'] = np.std(normalized_curves, axis=0)
    
    step_metrics['avg_exploration_efficiency'] = np.mean(step_metrics['exploration_efficiency']) if step_metrics['exploration_efficiency'] else 0.0
    step_metrics['avg_target_reaching_speed'] = np.mean(step_metrics['target_reaching_speeds']) if step_metrics['target_reaching_speeds'] else None
    
    return step_metrics

def create_comparison_plots(results: Dict[str, Any], output_dir: str):
    """比較グラフを作成"""
    
    # 1. 平均探査率の比較
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 平均探査率と標準偏差
    configs = list(results.keys())
    avg_rates = [results[config]['summary']['average_exploration_rate'] for config in configs]
    std_rates = [results[config]['summary']['std_exploration_rate'] for config in configs]
    
    bars = ax1.bar(configs, avg_rates, yerr=std_rates, capsize=5, alpha=0.7)
    ax1.set_title('平均探査率比較')
    ax1.set_ylabel('探査率')
    ax1.set_ylim(0, 1)
    
    # バーの色を設定
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 2. 探査率分布の箱ひげ図
    exploration_data = []
    for config in configs:
        rates = [ep['final_exploration_rate'] for ep in results[config]['episodes']]
        exploration_data.append(rates)
    
    bp = ax2.boxplot(exploration_data, labels=configs, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('探査率分布')
    ax2.set_ylabel('探査率')
    ax2.set_ylim(0, 1)
    
    # 3. エピソード進行による探査率変化
    for i, config in enumerate(configs):
        episodes = results[config]['episodes']
        rates = [ep['final_exploration_rate'] for ep in episodes]
        ax3.plot(range(1, len(rates) + 1), rates, 
                label=config, color=colors[i], alpha=0.7, linewidth=1)
    
    ax3.set_title('エピソード進行による探査率変化')
    ax3.set_xlabel('エピソード')
    ax3.set_ylabel('探査率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 目標達成率の比較
    target_rates = []
    for config in configs:
        summary = results[config]['summary']
        target_rate = summary['target_reached_episodes'] / summary['total_episodes'] * 100
        target_rates.append(target_rate)
    
    bars = ax4.bar(configs, target_rates, color=colors, alpha=0.7)
    ax4.set_title('目標達成率比較')
    ax4.set_ylabel('達成率 (%)')
    ax4.set_ylim(0, 100)
    
    # バーに値を表示
    for bar, rate in zip(bars, target_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 画像ファイルとして保存
    plot_path = os.path.join(output_dir, "config_comparison_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 比較グラフを保存: {plot_path}")
    
    # 統計的検定
    perform_statistical_tests(results, output_dir)

def create_step_analysis_plots(step_analysis_results: Dict[str, Any], output_dir: str):
    """Stepごとの詳細分析結果を可視化"""
    
    # 1. 探査率の時間変化（平均）
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    configs = list(step_analysis_results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 探査率の時間変化
    for i, config in enumerate(configs):
        if 'avg_exploration_curve' in step_analysis_results[config]:
            curve = step_analysis_results[config]['avg_exploration_curve']
            std_curve = step_analysis_results[config]['std_exploration_curve']
            steps = range(1, len(curve) + 1)
            
            ax1.plot(steps, curve, label=config, color=colors[i], linewidth=2)
            ax1.fill_between(steps, curve - std_curve, curve + std_curve, 
                           alpha=0.3, color=colors[i])
    
    ax1.set_title('探査率の時間変化（平均）')
    ax1.set_xlabel('ステップ')
    ax1.set_ylabel('探査率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 探査効率の比較
    efficiency_data = []
    efficiency_labels = []
    for config in configs:
        if 'avg_exploration_efficiency' in step_analysis_results[config]:
            efficiency = step_analysis_results[config]['avg_exploration_efficiency']
            efficiency_data.append(efficiency)
            efficiency_labels.append(config)
    
    if efficiency_data:
        bars = ax2.bar(efficiency_labels, efficiency_data, color=colors[:len(efficiency_data)], alpha=0.7)
        ax2.set_title('平均探査効率比較')
        ax2.set_ylabel('探査効率')
        
        # バーに値を表示
        for bar, value in zip(bars, efficiency_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
    
    # 3. 目標到達速度の比較
    speed_data = []
    speed_labels = []
    for config in configs:
        if 'avg_target_reaching_speed' in step_analysis_results[config]:
            speed = step_analysis_results[config]['avg_target_reaching_speed']
            if speed is not None:
                speed_data.append(speed)
                speed_labels.append(config)
    
    if speed_data:
        bars = ax3.bar(speed_labels, speed_data, color=colors[:len(speed_data)], alpha=0.7)
        ax3.set_title('目標到達速度比較')
        ax3.set_ylabel('到達ステップ数')
        
        # バーに値を表示
        for bar, value in zip(bars, speed_data):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.0f}', ha='center', va='bottom')
    
    # 4. 探査率の変化率分布
    for i, config in enumerate(configs):
        if 'episodes' in step_analysis_results[config]:
            all_changes = []
            for episode_data in step_analysis_results[config]['episodes']:
                if 'step_data' in episode_data['metrics']:
                    df = episode_data['metrics']['step_data']
                    if 'exploration_rate_change' in df.columns:
                        changes = df['exploration_rate_change'].dropna()
                        all_changes.extend(changes)
            
            if all_changes:
                ax4.hist(all_changes, bins=20, alpha=0.7, label=config, color=colors[i])
    
    ax4.set_title('探査率変化率の分布')
    ax4.set_xlabel('探査率変化量')
    ax4.set_ylabel('頻度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 画像ファイルとして保存
    plot_path = os.path.join(output_dir, "step_analysis_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Step分析グラフを保存: {plot_path}")
    
    # Step分析結果をCSVとして保存
    step_summary_data = []
    for config in configs:
        if 'avg_exploration_efficiency' in step_analysis_results[config]:
            summary = {
                'Config': config,
                'Avg_Exploration_Efficiency': step_analysis_results[config]['avg_exploration_efficiency'],
                'Avg_Target_Reaching_Speed': step_analysis_results[config].get('avg_target_reaching_speed', None),
                'Total_Episodes_Analyzed': len(step_analysis_results[config]['episodes'])
            }
            step_summary_data.append(summary)
    
    if step_summary_data:
        step_df = pd.DataFrame(step_summary_data)
        step_csv_path = os.path.join(output_dir, "step_analysis_summary.csv")
        step_df.to_csv(step_csv_path, index=False, encoding='utf-8')
        print(f"✓ Step分析サマリーを保存: {step_csv_path}")

def perform_statistical_tests(results: Dict[str, Any], output_dir: str):
    """統計的検定を実行"""
    from scipy import stats
    
    print(f"\n=== 統計的検定結果 ===")
    
    # 各Configの探査率データを取得
    config_data = {}
    for config_name, data in results.items():
        rates = [ep['final_exploration_rate'] for ep in data['episodes']]
        config_data[config_name] = rates
    
    # 一元配置分散分析（ANOVA）
    configs = list(config_data.keys())
    f_stat, p_value = stats.f_oneway(*[config_data[config] for config in configs])
    
    print(f"一元配置分散分析（ANOVA）:")
    print(f"  F統計量: {f_stat:.4f}")
    print(f"  p値: {p_value:.4f}")
    print(f"  有意差: {'あり' if p_value < 0.05 else 'なし'}")
    
    # ペアワイズ比較（Tukey検定）
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        all_rates = []
        all_configs = []
        for config in configs:
            all_rates.extend(config_data[config])
            all_configs.extend([config] * len(config_data[config]))
        
        tukey = pairwise_tukeyhsd(all_rates, all_configs)
        print(f"\nTukey検定結果:")
        print(tukey)
        
        # Tukey検定結果をCSVとして保存
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_path = os.path.join(output_dir, "tukey_test_results.csv")
        tukey_df.to_csv(tukey_path, index=False, encoding='utf-8')
        print(f"✓ Tukey検定結果を保存: {tukey_path}")
        
    except ImportError:
        print("statsmodelsが利用できないため、Tukey検定をスキップ")
    
    # 効果量（Cohen's d）の計算
    print(f"\n効果量（Cohen's d）:")
    effect_sizes = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            config1, config2 = configs[i], configs[j]
            d = (np.mean(config_data[config1]) - np.mean(config_data[config2])) / np.sqrt(
                ((len(config_data[config1]) - 1) * np.var(config_data[config1], ddof=1) + 
                 (len(config_data[config2]) - 1) * np.var(config_data[config2], ddof=1)) / 
                (len(config_data[config1]) + len(config_data[config2]) - 2)
            )
            print(f"  {config1} vs {config2}: {d:.3f}")
            effect_sizes.append({
                'Config1': config1,
                'Config2': config2,
                'Cohen_d': d
            })
    
    # 効果量をCSVとして保存
    effect_df = pd.DataFrame(effect_sizes)
    effect_path = os.path.join(output_dir, "effect_sizes.csv")
    effect_df.to_csv(effect_path, index=False, encoding='utf-8')
    print(f"✓ 効果量を保存: {effect_path}")

def main():
    """メイン関数"""
    print("=== Config_A、B、C、D比較分析開始 ===")
    
    try:
        results, step_analysis_results = analyze_configs()
        print(f"\n🎉 分析が正常に完了しました！")
        print(f"結果は 'config_results' ディレクトリに保存されました。")
        
    except Exception as e:
        print(f"❌ 分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 