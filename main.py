"""
Main entry point for the Multi-Agent Simulation project.
Uses the new optimized architecture with centralized configuration and logging.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.application import Application, ExperimentConfig, create_application
from core.config import config
from core.logging import get_logger
from params.simulation import SimulationParam
from agents.agent_factory import create_initial_agents
from envs.env import Env
from envs.observation_space import create_swarm_observation_space, create_system_observation_space
from envs.action_space import create_swarm_action_space, create_system_action_space
from models.model import create_swarm_model, create_system_model
from utils.logger import create_experiment_logger


def create_simulation_environment(simulation_param: SimulationParam):
    """
    シミュレーション環境を作成
    """
    # 環境を作成
    env = Env(simulation_param)
    
    # 初期エージェントを作成（Noneチェック付き）
    if simulation_param.agent is None:
        raise ValueError("AgentParam is required")
    system_agent, swarm_agents = create_initial_agents(env, simulation_param.agent)
    
    # 環境にSystemAgentを設定
    env.set_system_agent(system_agent)
    
    return env, system_agent, swarm_agents


def main():
    """Main application entry point"""
    logger = get_logger("main")
    logger.info("Starting Multi-Agent Simulation application...")
    
    try:
        # シミュレーションパラメータを初期化
        simulation_param = SimulationParam()
        logger.info("Simulation parameters initialized")
        
        # シミュレーション環境を作成
        env, system_agent, swarm_agents = create_simulation_environment(simulation_param)
        logger.info(f"Simulation environment created with {len(swarm_agents)} swarm agents")
        
        # 実験設定を作成（Noneチェック付き）
        if simulation_param.agent is None:
            raise ValueError("AgentParam is required")
        if simulation_param.explore is None:
            raise ValueError("ExploreParam is required")
        if simulation_param.environment is None:
            raise ValueError("EnvironmentParam is required")
        if simulation_param.environment.map is None:
            raise ValueError("MapParam is required")
        if simulation_param.agent.system_agent_param is None:
            raise ValueError("SystemAgentParam is required")
        if simulation_param.agent.system_agent_param.branch_condition is None:
            raise ValueError("BranchConditionParam is required")
        if simulation_param.agent.system_agent_param.integration_condition is None:
            raise ValueError("IntegrationConditionParam is required")
        
        experiment_config = ExperimentConfig(
            name="multi_agent_simulation",
            description="Multi-agent swarm exploration with dynamic branching/integration",
            parameters={
                "swarm_algorithm": "vfh_fuzzy",
                "system_algorithm": "actor_critic",
                "robot_count": len(simulation_param.robot_params),
                "initial_swarm_count": simulation_param.explore.initialSwarmNum,
                "map_size": f"{simulation_param.environment.map.width}x{simulation_param.environment.map.height}",
                "branch_algorithm": simulation_param.agent.system_agent_param.branch_condition.branch_algorithm,
                "integration_algorithm": simulation_param.agent.system_agent_param.integration_condition.integration_algorithm
            },
            num_episodes=simulation_param.agent.episodeNum,
            max_steps_per_episode=simulation_param.agent.maxStepsPerEpisode,
            save_models=True,
            save_logs=True,
            save_visualizations=config.system.enable_gif_save
        )
        
        # 実験を実行
        logger.info(f"Starting experiment: {experiment_config.name}")
        
        # ログディレクトリを作成
        log_dir = config.simulation.log_dir or "./logs/default"
        run_simulation(env, system_agent, swarm_agents, experiment_config, log_dir)
        
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise
    
    logger.info("Application finished successfully")


def run_simulation(env, system_agent, swarm_agents, experiment_config, log_dir: str):
    """
    シミュレーションを実行（ログ機能付き）
    """
    logger = get_logger("simulation")
    
    # 実験ロガーを作成
    experiment_logger = create_experiment_logger(log_dir, experiment_config.name)
    experiment_logger.set_experiment_config(experiment_config.parameters)
    
    # 初期状態を保存
    env.save_initial_state(log_dir)
    
    for episode in range(experiment_config.num_episodes):
        logger.info(f"Starting episode {episode + 1}/{experiment_config.num_episodes}")
        
        # エピソード開始時の処理
        env.start_episode(episode)
        
        # 環境をリセット
        state = env.reset()
        
        episode_reward = 0
        episode_steps = 0
        
        for step in range(experiment_config.max_steps_per_episode):
            # 各SwarmAgentのアクションを取得
            swarm_actions = {}
            
            # デバッグ情報をログ出力
            if step == 0:
                logger.info(f"Debug: swarm_agents keys: {list(swarm_agents.keys())}")
                logger.info(f"Debug: env.swarms: {[swarm.swarm_id for swarm in env.swarms]}")
            
            for swarm_id, swarm_agent in swarm_agents.items():
                # env.swarmsから該当する群を探す
                swarm_exists = any(swarm.swarm_id == swarm_id for swarm in env.swarms)
                if swarm_exists:  # 存在する群のみ
                    # 適切な状態を取得
                    swarm_state = env.get_swarm_agent_observation(swarm_id)
                    action, action_info = swarm_agent.get_action(swarm_state, episode)
                    swarm_actions[swarm_id] = action  # action_infoではなくactionを渡す
                    if step == 0:
                        logger.info(f"Debug: Swarm {swarm_id} action: {action}")
                else:
                    if step == 0:
                        logger.info(f"Debug: Swarm {swarm_id} not found in env.swarms")
            
            # SystemAgentのアクションを取得
            # 簡易的な状態取得（実際の実装では適切なメソッドを使用）
            system_state = {"episode": episode, "step": step, "swarm_count": len(env.swarms)}
            system_action, system_action_info = system_agent.get_action(system_state, episode)
            
            # 環境をステップ実行
            next_state, rewards, done, truncated, info = env.step(swarm_actions)
            
            # 新しい群が作成された場合、対応するSwarmAgentを作成
            for swarm in env.swarms:
                if swarm.swarm_id not in swarm_agents:
                    logger.info(f"Creating new SwarmAgent for swarm {swarm.swarm_id}")
                    from agents.agent_factory import create_swarm_agent
                    from params.swarm_agent import SwarmAgentParam
                    
                    # 新しいSwarmAgentParamを作成（元の群のパラメータをコピー）
                    new_param = SwarmAgentParam()
                    if swarm_agents:  # 既存の群がある場合、そのパラメータをコピー
                        existing_param = next(iter(swarm_agents.values())).param
                        new_param = existing_param
                    
                    # 新しいSwarmAgentを作成
                    new_swarm_agent = create_swarm_agent(env, new_param, system_agent, swarm.swarm_id)
                    swarm_agents[swarm.swarm_id] = new_swarm_agent
                    
                    # SystemAgentに新しいSwarmAgentを登録
                    system_agent.register_swarm_agent(new_swarm_agent, swarm.swarm_id)
            
            # フレームをキャプチャ（GIF作成用）
            env.capture_frame()
            
            # 報酬を累積
            if isinstance(rewards, dict):
                episode_reward += sum(rewards.values())
            else:
                episode_reward += rewards
            
            episode_steps += 1
            
            # デバッグ情報をログ出力
            if step % 100 == 0:
                logger.info(f"Episode {episode + 1}, Step {step}: {len(env.swarms)} swarms, {len(swarm_actions)} actions")
            
            if done:
                logger.info(f"Episode {episode + 1} completed at step {step}")
                break
        
        # エピソード終了時の処理
        exploration_rate = env.get_exploration_rate()
        
        # エピソードデータを記録
        episode_data = {
            'episode': episode,
            'total_reward': episode_reward,
            'exploration_rate': exploration_rate,
            'swarm_count': len(env.swarms),
            'steps': episode_steps,
            'duration': 0.0  # 必要に応じて計測
        }
        experiment_logger.log_episode(episode, episode_data)
        
        # GIFを保存
        env.end_episode(log_dir)
        
        # 定期的にチェックポイントを保存（10エピソードごと）
        if (episode + 1) % 10 == 0:
            try:
                from agents.agent_factory import save_agents_checkpoint
                save_agents_checkpoint(system_agent, swarm_agents, log_dir, episode + 1)
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
        logger.info(f"Episode {episode + 1} summary: {len(env.swarms)} swarms, exploration rate: {exploration_rate:.2f}")
    
    # 最終レポートを保存
    experiment_logger.save_final_report()
    experiment_logger.close()
    
    logger.info("Simulation completed with logging")


if __name__ == "__main__":
    main()