"""
学習済みモデル作成スクリプト
各構成のモデルを学習・保存する
"""

import os
import json
from datetime import datetime
from typing import Dict, List

from core.application import Application
from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam, LearningParameter as SystemLearningParameter
from params.swarm_agent import SwarmAgentParam, LearningParameter as SwarmLearningParameter
from agents.agent_factory import create_initial_agents
from envs.env import Env


class ModelTrainer:
    """モデル学習クラス"""
    
    def __init__(self):
        self.base_config = SimulationParam()
        self.results = {}
        
    def create_training_configs(self) -> List[Dict]:
        """学習設定を作成"""
        configs = [
            {
                'name': 'Config_B_System',
                'system_agent_learning': False,
                'swarm_agent_learning': True,
                'system_agent_branching': False,
                'description': 'SystemAgent(学習なし) + SwarmAgent(学習あり)'
            },
            {
                'name': 'Config_D_System',
                'system_agent_learning': True,
                'swarm_agent_learning': True,
                'system_agent_branching': True,
                'description': 'SystemAgent(学習あり) + SwarmAgent(学習あり)'
            }
        ]
        return configs
    
    def setup_training_environment(self) -> SimulationParam:
        """学習用環境設定"""
        sim_param = self.base_config.copy()
        
        # 学習用パラメータ設定
        sim_param.environment.map.width = 200
        sim_param.environment.map.height = 100
        sim_param.environment.obstacle.probability = 0.0  # 学習時は障害物なし
        sim_param.explore.robotNum = 20
        sim_param.explore.coordinate.x = 50.0
        sim_param.explore.coordinate.y = 100.0
        sim_param.explore.boundary.outer = 5.0
        
        # 学習パラメータ
        sim_param.agent.episodeNum = 1500
        sim_param.agent.maxStepsPerEpisode = 50
        
        return sim_param
    
    def setup_agent_config(self, config: Dict) -> AgentParam:
        """エージェント設定を作成"""
        agent_param = AgentParam()
        
        # SystemAgent設定
        system_param = SystemAgentParam()
        if config['system_agent_learning']:
            system_param.learningParameter = SystemLearningParameter()
        else:
            system_param.learningParameter = None
        
        # 分岐・統合設定
        if config['system_agent_branching']:
            system_param.branch_condition.branch_enabled = True
            system_param.integration_condition.integration_enabled = True
        else:
            system_param.branch_condition.branch_enabled = False
            system_param.integration_condition.integration_enabled = False
        
        agent_param.system_agent_param = system_param
        
        # SwarmAgent設定
        swarm_param = SwarmAgentParam()
        if config['swarm_agent_learning']:
            swarm_param.isLearning = True
            swarm_param.learningParameter = SwarmLearningParameter()
        else:
            swarm_param.isLearning = False
            swarm_param.learningParameter = None
        
        agent_param.swarm_agent_params = [swarm_param]
        
        return agent_param
    
    def train_models(self, config: Dict) -> Dict:
        """モデル学習実行"""
        print(f"学習開始: {config['name']} - {config['description']}")
        
        # 環境設定
        sim_param = self.setup_training_environment()
        env = Env(sim_param)
        
        # エージェント設定
        agent_param = self.setup_agent_config(config)
        
        # エージェント作成
        system_agent, swarm_agents = create_initial_agents(env, agent_param)
        
        # 学習実行
        training_results = self._execute_training(env, system_agent, swarm_agents, config)
        
        # モデル保存
        self._save_models(system_agent, swarm_agents, config)
        
        return training_results
    
    def _execute_training(self, env, system_agent, swarm_agents, config: Dict) -> Dict:
        """学習実行"""
        episode_results = []
        
        for episode in range(self.base_config.agent.episodeNum):
            state = env.reset()
            episode_reward = 0.0
            episode_exploration = 0.0
            
            for step in range(self.base_config.agent.maxStepsPerEpisode):
                # エージェント行動決定
                if config['system_agent_learning']:
                    system_action = system_agent.get_action(state, episode)
                else:
                    system_action = self._get_default_system_action()
                
                if config['swarm_agent_learning']:
                    swarm_actions = {}
                    for swarm_id, agent in swarm_agents.items():
                        swarm_state = env.get_swarm_agent_observation(swarm_id)
                        swarm_actions[swarm_id] = agent.get_action(swarm_state, episode)
                else:
                    swarm_actions = {swarm_id: self._get_default_swarm_action() 
                                   for swarm_id in swarm_agents.keys()}
                
                # 環境更新
                next_state, reward, done, truncated, info = env.step(swarm_actions)
                
                episode_reward += reward
                episode_exploration = env.get_exploration_rate()
                
                state = next_state
                
                if done or truncated:
                    break
            
            # 学習更新
            if config['system_agent_learning']:
                system_agent.train()
            if config['swarm_agent_learning']:
                for agent in swarm_agents.values():
                    agent.train()
            
            # 結果記録
            episode_results.append({
                'episode': episode,
                'reward': episode_reward,
                'exploration_rate': episode_exploration,
                'steps': step + 1
            })
            
            if episode % 100 == 0:
                print(f"  Episode {episode}: Reward={episode_reward:.2f}, Exploration={episode_exploration:.3f}")
        
        return {
            'config': config,
            'episode_results': episode_results,
            'final_exploration_rate': episode_exploration,
            'avg_reward': sum(r['reward'] for r in episode_results) / len(episode_results)
        }
    
    def _get_default_system_action(self) -> Dict:
        """デフォルトのSystemAgent行動"""
        return {
            'action_type': 'none',
            'target_swarm_id': 0,
            'branch_threshold': 0.3,
            'integration_threshold': 0.7
        }
    
    def _get_default_swarm_action(self) -> Dict:
        """デフォルトのSwarmAgent行動"""
        import numpy as np
        return {
            'theta': np.random.uniform(0, 2*np.pi),
            'th': 0.5,
            'k_e': 10.0,
            'k_c': 5.0
        }
    
    def _save_models(self, system_agent, swarm_agents, config: Dict):
        """モデル保存"""
        models_dir = "trained_models"
        os.makedirs(models_dir, exist_ok=True)
        
        config_dir = os.path.join(models_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        # SystemAgentモデル保存
        if config['system_agent_learning'] and system_agent and hasattr(system_agent, 'model'):
            system_model_path = os.path.join(config_dir, "system_agent_model.keras")
            system_agent.model.save(system_model_path)
            print(f"  SystemAgentモデル保存: {system_model_path}")
        
        # SwarmAgentモデル保存
        if config['swarm_agent_learning'] and swarm_agents:
            swarm_model_path = os.path.join(config_dir, "swarm_agent_model.keras")
            # 最初のSwarmAgentのモデルを保存（全SwarmAgentは同じモデルを使用）
            first_agent = list(swarm_agents.values())[0]
            if hasattr(first_agent, 'model'):
                first_agent.model.save(swarm_model_path)
                print(f"  SwarmAgentモデル保存: {swarm_model_path}")
        
        # 学習設定保存
        config_path = os.path.join(config_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_all_training(self):
        """全構成の学習実行"""
        configs = self.create_training_configs()
        
        for config in configs:
            try:
                result = self.train_models(config)
                self.results[config['name']] = result
                print(f"学習完了: {config['name']}")
            except Exception as e:
                print(f"学習エラー {config['name']}: {e}")
        
        # 学習結果保存
        self.save_training_results()
    
    def save_training_results(self):
        """学習結果保存"""
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"training_results_{timestamp}.json")
        
        # JSON変換可能な形式に変換
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                'config': result['config'],
                'final_exploration_rate': result['final_exploration_rate'],
                'avg_reward': result['avg_reward'],
                'episode_count': len(result['episode_results'])
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"学習結果保存: {results_file}")


def main():
    """メイン実行関数"""
    trainer = ModelTrainer()
    trainer.run_all_training()


if __name__ == "__main__":
    main() 