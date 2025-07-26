#!/usr/bin/env python3
"""
4つの構成での比較実験実行スクリプト
SystemAgentとSwarmAgentの学習有無による性能比較
"""

import sys
import os
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.comparison_experiment import ComparisonExperiment
from params.simulation import SimulationParam


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='4つの構成での比較実験')
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=50,
        help='エピソード数 (default: 50)'
    )
    
    parser.add_argument(
        '--steps', 
        type=int, 
        default=200,
        help='エピソードあたりの最大ステップ数 (default: 200)'
    )
    
    parser.add_argument(
        '--target-rate', 
        type=float, 
        default=0.8,
        help='目標探査率 (default: 0.8)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='experiment_results',
        help='結果出力ディレクトリ (default: experiment_results)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='クイックモード（エピソード数とステップ数を減らす）'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='詳細なログ出力'
    )
    
    parser.add_argument(
        '--no-pretrained', 
        action='store_true',
        help='事前学習済みモデルを使用しない'
    )
    
    parser.add_argument(
        '--num-runs', 
        type=int, 
        default=5,
        help='統計的信頼性のための複数回実行数 (default: 5)'
    )
    
    return parser.parse_args()


def setup_experiment_config(args):
    """実験設定を作成"""
    base_config = SimulationParam()
    
    if args.quick:
        # クイックモード
        base_config.episodeNum = 20
        base_config.maxStepsPerEpisode = 50  # クイックモードでも最低50ステップ
        print("Quick mode: Reduced episodes and steps for faster execution")
    else:
        # 通常モード
        base_config.episodeNum = args.episodes
        base_config.maxStepsPerEpisode = args.steps
    
    # その他の設定
    base_config.map_width = 100
    base_config.map_height = 200
    base_config.robot_num = 20
    base_config.obstacle_density = 0.0
    
    return base_config


def print_experiment_summary():
    """実験概要を表示"""
    print("=" * 60)
    print("4つの構成での比較実験")
    print("=" * 60)
    print()
    print("実験構成:")
    print("1. Config_A: SystemAgent(学習なし, 分岐なし) + SwarmAgent(学習なし)")
    print("2. Config_B: SystemAgent(学習なし, 分岐なし) + SwarmAgent(学習済み)")
    print("3. Config_C: SystemAgent(学習なし, 分岐あり) + SwarmAgent(学習なし)")
    print("4. Config_D: SystemAgent(学習済み, 分岐あり) + SwarmAgent(学習済み)")
    print()
    print("評価指標:")
    print("- 探査率80%達成までの速度")
    print("- 探査進捗の早さ")
    print("- 環境変化に対するロバスト性")
    print()
    print("環境設定:")
    print("- マップサイズ: 100×200")
    print("- ロボット数: 20台")
    print("- 障害物密度: 0.0, 0.003, 0.005")
    print()
    print("ステップ数設定:")
    print("- 1領域あたりのfollowerの各探査ステップ: 30")
    print("- 1エピソードあたりの最大ステップ数: 200")
    print("- ロボット20台での探査効率を考慮")
    print()
    print("統計的信頼性:")
    print("- 複数回実行による統計分析")
    print("- 平均値、標準偏差、信頼区間の計算")
    print("- 各構成・環境で5回実行（デフォルト）")
    print("=" * 60)
    print()


def check_pretrained_models():
    """事前学習済みモデルの存在確認"""
    pretrained_dir = "pretrained_models"
    system_model_path = os.path.join(pretrained_dir, "system_agent_model.keras")
    swarm_model_path = os.path.join(pretrained_dir, "swarm_agent_model.keras")
    
    system_exists = os.path.exists(system_model_path)
    swarm_exists = os.path.exists(swarm_model_path)
    
    if not system_exists or not swarm_exists:
        print("Warning: 事前学習済みモデルが見つかりません")
        print(f"SystemAgent model: {'✓' if system_exists else '✗'}")
        print(f"SwarmAgent model: {'✓' if swarm_exists else '✗'}")
        print()
        print("事前学習済みモデルを作成するには:")
        print("python train_pretrained_models.py --train-both")
        print()
        return False
    
    print("✓ 事前学習済みモデルが見つかりました")
    return True


def main():
    """メイン実行関数"""
    # 引数解析
    args = parse_arguments()
    
    # 実験概要表示
    print_experiment_summary()
    
    # 事前学習済みモデルの確認
    if not args.no_pretrained:
        if not check_pretrained_models():
            print("事前学習済みモデルなしで実験を続行します...")
            print("（学習ありの構成はランダム初期化で実行されます）")
    
    # 実験設定作成
    base_config = setup_experiment_config(args)
    
    print(f"実験設定:")
    print(f"  エピソード数: {base_config.episodeNum}")
    print(f"  最大ステップ数: {base_config.maxStepsPerEpisode}")
    print(f"  目標探査率: {args.target_rate}")
    print(f"  複数回実行数: {args.num_runs}")
    print(f"  出力ディレクトリ: {args.output_dir}")
    print(f"  事前学習済みモデル使用: {not args.no_pretrained}")
    print()
    
    # 実験実行
    print("実験を開始します...")
    experiment = ComparisonExperiment(base_config)
    
    try:
        # 比較実験実行
        results = experiment.run_comparison_experiments()
        
        # 結果保存
        print("結果を保存中...")
        results_file, analysis_file = experiment.save_results(args.output_dir)
        
        # 結果可視化
        print("結果を可視化中...")
        experiment.plot_results(args.output_dir)
        
        # 分析結果表示
        print("\n" + "=" * 60)
        print("実験結果")
        print("=" * 60)
        
        analysis = experiment.analyze_results()
        
        print("\n=== 性能比較 (障害物なし環境) ===")
        for config, perf in analysis['performance_comparison'].items():
            print(f"\n{config}:")
            print(f"  目標達成率: {perf['target_reach_rate']:.3f}")
            if perf['avg_steps_to_target'] is not None:
                print(f"  平均ステップ数: {perf['avg_steps_to_target']:.1f}")
            else:
                print(f"  平均ステップ数: 目標未達成")
            print(f"  探査速度: {perf['avg_exploration_speed']:.4f}")
        
        print("\n=== ロバスト性分析 ===")
        for config, robust in analysis['robustness_analysis'].items():
            print(f"{config}: 性能標準偏差 = {robust['performance_std']:.3f}")
            print(f"  環境別性能: {robust['env_performances']}")
        
        print("\n=== 速度分析 ===")
        for config, speed in analysis['speed_analysis'].items():
            if speed['avg_speed'] is not None:
                print(f"{config}: 平均速度 = {speed['avg_speed']:.1f} ± {speed['speed_std']:.1f} ステップ")
            else:
                print(f"{config}: 目標未達成")
        
        print("\n=== 障害物密度影響分析 ===")
        for config, density_impact in analysis['obstacle_density_analysis'].items():
            print(f"{config}: 平均性能劣化率 = {density_impact['avg_degradation_rate']:.3f}")
            print(f"  密度別性能: {density_impact['density_performances']}")
        
        print(f"\n結果ファイル:")
        print(f"  詳細結果: {results_file}")
        print(f"  分析結果: {analysis_file}")
        print(f"  グラフ: {args.output_dir}/")
        
        print("\n実験完了!")
        
    except KeyboardInterrupt:
        print("\n実験が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 