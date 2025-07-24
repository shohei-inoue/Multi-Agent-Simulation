"""
エージェント設定クラス
エージェントの設定と作成を管理
"""

from agents.agent_factory import create_agent, create_system_with_swarm_agents
from agents.swarm_agent import SwarmAgent
from agents.system_agent import SystemAgent
from params.agent import AgentParam
import tensorflow as tf
from typing import Dict, Any, List, Tuple


class AgentConfig:
    """エージェント設定クラス"""
    
    def __init__(self, env, agent_param: AgentParam):
        self.env = env
        self.agent_param = agent_param
        self.algorithm = None
        self.model = None
        self.optimizer = None
        
        # エージェント
        self.system_agent = None
        self.swarm_agents = []
        self.current_swarm_agent = None  # 現在アクティブな群エージェント
        
        # 設定を初期化
        self._setup_components()
        self._create_agents()
    
    def _setup_components(self):
        """コンポーネントを設定"""
        # アルゴリズムを作成
        self.algorithm = self._create_algorithm()
            
        # モデルを作成
        self.model = self._create_model()
        
        # オプティマイザーを作成
        self.optimizer = self._create_optimizer()
    
    def _create_algorithm(self):
        """アルゴリズムを作成"""
        algorithm_name = self.agent_param.algorithm
        
        if algorithm_name == "vfh_fuzzy":
            from algorithms.vfh_fuzzy import AlgorithmVfhFuzzy
            return AlgorithmVfhFuzzy()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def _create_model(self):
        """モデルを作成"""
        model_name = self.agent_param.learningParameter.model
        
        if model_name == "actor-critic":
            from models.actor_critic import ModelActorCritic
            return ModelActorCritic()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _create_optimizer(self):
        """オプティマイザーを作成"""
        optimizer_name = self.agent_param.learningParameter.optimizer
        learning_rate = self.agent_param.learningParameter.learningLate
        
        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_agents(self):
        """エージェントを作成"""
        agent_type = self.agent_param.learningParameter.type
        
        if agent_type == "a2c":
            # A2Cエージェントを作成
            self.agent = create_agent(
                "a2c",
                env=self.env,
                algorithm=self.algorithm,
                model=self.model,
                optimizer=self.optimizer,
                gamma=self.agent_param.learningParameter.gamma,
                n_steps=self.agent_param.learningParameter.nStep,
                max_steps_per_episode=self.agent_param.maxStepsPerEpisode,
                action_space=self.env.action_space
            )
        
        elif agent_type == "swarm_agent":
            # 単一の群エージェントを作成
            self.agent = create_agent(
                "swarm_agent",
                env=self.env,
                algorithm=self.algorithm,
                model=self.model,
                optimizer=self.optimizer,
                gamma=self.agent_param.learningParameter.gamma,
                n_steps=self.agent_param.learningParameter.nStep,
                max_steps_per_episode=self.agent_param.maxStepsPerEpisode,
                action_space=self.env.action_space,
                swarm_id="swarm_single"
            )
        
        elif agent_type == "system_agent":
            # システムエージェントを作成
            update_interval = getattr(self.agent_param.learningParameter, 'updateInterval', 1.0)
            
            self.agent = create_agent(
                "system_agent",
                env=self.env,
                algorithm=self.algorithm,
                model=self.model,
                optimizer=self.optimizer,
                gamma=self.agent_param.learningParameter.gamma,
                n_steps=self.agent_param.learningParameter.nStep,
                max_steps_per_episode=self.agent_param.maxStepsPerEpisode,
                action_space=self.env.action_space,
                update_interval=update_interval
            )
        
        elif agent_type == "system_with_swarms":
            # システムエージェントと群エージェントを作成
            update_interval = getattr(self.agent_param.learningParameter, 'updateInterval', 1.0)
            initial_swarm_count = getattr(self.agent_param.learningParameter, 'initial_swarm_count', 1)
            
            self.system_agent, self.swarm_agents = create_system_with_swarm_agents(
                env=self.env,
                algorithm=self.algorithm,
                model=self.model,
                optimizer=self.optimizer,
                gamma=self.agent_param.learningParameter.gamma,
                n_steps=self.agent_param.learningParameter.nStep,
                max_steps_per_episode=self.agent_param.maxStepsPerEpisode,
                action_space=self.env.action_space,
                initial_swarm_count=initial_swarm_count,
                update_interval=update_interval
            )
            
            # 最初の群エージェントを現在のエージェントとして設定
            if self.swarm_agents:
                self.current_swarm_agent = self.swarm_agents[0]
                self.agent = self.current_swarm_agent
            else:
                self.agent = self.system_agent
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_action(self, state: Dict[str, Any], episode: int = 0, log_dir: str = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """アクションを取得"""
        if hasattr(self, 'current_swarm_agent') and self.current_swarm_agent is not None:
            # 群エージェントからアクションを取得
            return self.current_swarm_agent.get_action(state, episode, log_dir)
        else:
            # 通常のエージェントからアクションを取得
            return self.agent.get_action(state, episode, log_dir)
    
    def train(self, log_dir: str = None):
        """学習を実行"""
        if self.agent_param.isLearning:
            if hasattr(self, 'current_swarm_agent') and self.current_swarm_agent is not None:
                # 群エージェントの学習
                self.current_swarm_agent.train(log_dir)
            else:
                # 通常のエージェントの学習
                self.agent.train(log_dir)
    
    def get_system_agent(self) -> SystemAgent:
        """システムエージェントを取得"""
        return self.system_agent
    
    def get_swarm_agents(self) -> List[SwarmAgent]:
        """群エージェントのリストを取得"""
        return self.swarm_agents.copy()
    
    def get_current_swarm_agent(self) -> SwarmAgent:
        """現在の群エージェントを取得"""
        return self.current_swarm_agent
    
    def set_current_swarm_agent(self, swarm_agent: SwarmAgent):
        """現在の群エージェントを設定"""
        self.current_swarm_agent = swarm_agent
        self.agent = swarm_agent
    
    def add_swarm_agent(self, swarm_agent: SwarmAgent):
        """群エージェントを追加"""
        self.swarm_agents.append(swarm_agent)
    
    def remove_swarm_agent(self, swarm_id: str):
        """群エージェントを削除"""
        self.swarm_agents = [s for s in self.swarm_agents if s.get_swarm_id() != swarm_id]
        
        # 削除されたエージェントが現在のエージェントだった場合、別のエージェントを選択
        if self.current_swarm_agent and self.current_swarm_agent.get_swarm_id() == swarm_id:
            if self.swarm_agents:
                self.current_swarm_agent = self.swarm_agents[0]
                self.agent = self.current_swarm_agent
    else:
                self.current_swarm_agent = None
                self.agent = self.system_agent
    
    def get_system_stats(self) -> Dict[str, Any]:
        """システム統計を取得"""
        if self.system_agent:
            return self.system_agent.get_system_stats()
        else:
            return {}
    
    def get_swarm_management_info(self) -> Dict[str, Any]:
        """群管理情報を取得"""
        if self.system_agent:
            return self.system_agent.get_swarm_management_info()
    else:
            return {
                "current_swarm": {
                    "id": self.current_swarm_agent.get_swarm_id() if self.current_swarm_agent else "none",
                    "state": self.current_swarm_agent.get_swarm_state() if self.current_swarm_agent else "none",
                    "type": "SwarmAgent"
                },
                "all_swarms": [
                    {
                        "id": swarm.get_swarm_id(),
                        "type": "SwarmAgent"
                    }
                    for swarm in self.swarm_agents
                ],
                "management_stats": {
                    "total_swarms_created": len(self.swarm_agents),
                    "current_active_swarms": len(self.swarm_agents)
                }
            }
    
    def stop(self):
        """システムエージェントのスレッドを停止"""
        if self.system_agent:
            self.system_agent.stop()