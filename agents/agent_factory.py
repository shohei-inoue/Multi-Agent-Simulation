"""
エージェントファクトリー
エージェントの作成を管理
"""

from agents.swarm_agent import SwarmAgent
from agents.system_agent import SystemAgent
from agents.base_agent import BaseAgent
from algorithms.algorithm_factory import select_algorithm
from params.swarm_agent import SwarmAgentParam
from params.system_agent import SystemAgentParam
from params.agent import AgentParam
from algorithms.vfh_fuzzy import AlgorithmVfhFuzzy
from algorithms.branch_algorithm import create_branch_algorithm
from algorithms.integration_algorithm import create_integration_algorithm
from models.model import create_swarm_model, create_system_model, Model, ModelType
from envs.action_space import create_swarm_action_space, create_system_action_space
from envs.observation_space import create_swarm_observation_space, create_system_observation_space
from typing import Dict, Any, Optional, List, Tuple
import copy


def create_swarm_agent(env, param: SwarmAgentParam, system_agent, swarm_id: int) -> SwarmAgent:
    """
    SwarmAgentを作成
    """
    # アルゴリズムを作成
    algorithm = AlgorithmVfhFuzzy(env=env)
    
    # 観測空間とアクション空間を作成
    observation_space = create_swarm_observation_space()
    action_space = create_swarm_action_space()
    
    # モデルを作成（SwarmActorCritic）
    # Dict型の観測空間を平坦化してinput_dimを計算
    if hasattr(observation_space, 'spaces'):
        # Dict型の場合、各空間の要素数を合計
        input_dim = sum(space.shape[0] if hasattr(space, 'shape') and len(space.shape) > 0 else 1 
                       for space in observation_space.spaces.values())
    else:
        input_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else 32
    model = create_swarm_model(input_dim)
    
    # SwarmAgentを作成
    swarm_agent = SwarmAgent(
        env=env,
        algorithm=algorithm,
        model=model.model if hasattr(model, 'model') else model,
        action_space=action_space,
        param=param,
        system_agent=system_agent,
        swarm_id=swarm_id
    )
    
    return swarm_agent


def create_system_agent(env, param: SystemAgentParam) -> SystemAgent:
    """
    SystemAgentを作成
    """
    # 観測空間とアクション空間を作成
    observation_space = create_system_observation_space()
    action_space = create_system_action_space()
    
    # モデルを作成（SystemActorCritic）
    # Dict型の観測空間を平坦化してinput_dimを計算
    if hasattr(observation_space, 'spaces'):
        # Dict型の場合、各空間の要素数を合計
        input_dim = sum(space.shape[0] if hasattr(space, 'shape') and len(space.shape) > 0 else 1 
                       for space in observation_space.spaces.values())
    else:
        input_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else 32
    max_swarms = getattr(param, 'max_swarms', 10)
    model = create_system_model(input_dim, max_swarms)
    
    # SystemAgentを作成
    system_agent = SystemAgent(
        env=env,
        algorithm=None,
        model=model.model if hasattr(model, 'model') else model,
        action_space=action_space,
        param=param
    )
    
    return system_agent


def create_branched_swarm_agent(source_swarm_agent: SwarmAgent, new_swarm_id: int, 
                               learning_info: Optional[Dict[str, Any]] = None) -> SwarmAgent:
    """
    分岐時に新しいSwarmAgentを作成
    """
    # source_swarm_agentがNoneの場合のチェック
    if source_swarm_agent is None:
        print(f"Error: source_swarm_agent is None for new_swarm_id {new_swarm_id}")
        return None
    
    # paramがNoneの場合のチェック
    if source_swarm_agent.param is None:
        print(f"Error: source_swarm_agent.param is None for new_swarm_id {new_swarm_id}")
        return None
    
    try:
        # パラメータをコピー
        new_param = copy.deepcopy(source_swarm_agent.param)
        
        # 新しいSwarmAgentを作成
        new_swarm_agent = SwarmAgent(
            env=source_swarm_agent.env,
            algorithm=source_swarm_agent.algorithm,
            model=source_swarm_agent.model,  # 同じモデルを共有
            action_space=source_swarm_agent.action_space,
            param=new_param,
            system_agent=source_swarm_agent.system_agent,
            swarm_id=new_swarm_id
        )
        
        # 学習情報を引き継ぐ
        if learning_info and learning_info.get('model_weights') is not None and new_swarm_agent.model is not None:
            new_swarm_agent.model.set_weights(learning_info['model_weights'])
        
        return new_swarm_agent
        
    except Exception as e:
        print(f"Error creating branched swarm agent: {e}")
        return None


def create_initial_agents(env, agent_param: AgentParam) -> Tuple[SystemAgent, Dict[int, SwarmAgent]]:
    """
    初期エージェントを作成
    """
    system_model = create_system_model(input_dim=32)
    
    if agent_param.system_agent_param is None:
        raise ValueError("system_agent_param is required")
    system_agent = SystemAgent(
        env=env,
        algorithm=None,
        model=system_model.model,
        action_space=create_system_action_space(),
        param=agent_param.system_agent_param
    )
    swarm_agents = {}
    for i, swarm_param in enumerate(agent_param.swarm_agent_params):
        swarm_id = i + 1
        swarm_agent = create_swarm_agent(env, swarm_param, system_agent, swarm_id)
        swarm_agents[swarm_id] = swarm_agent
    for swarm_id, swarm_agent in swarm_agents.items():
        system_agent.register_swarm_agent(swarm_agent, swarm_id)
    return system_agent, swarm_agents


def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    従来のエージェント作成関数（後方互換性のため）
    """
    # この関数は非推奨。新しいcreate_swarm_agentやcreate_system_agentを使用してください
    raise DeprecationWarning("create_agent is deprecated. Use create_swarm_agent or create_system_agent instead.")


def create_swarm_agent_with_pretrained_model(env, swarm_id: int, param: SwarmAgentParam, 
                                           pretrained_model_path: str) -> SwarmAgent:
    """
    学習済みモデルを使用してSwarmAgentを作成
    """
    # アルゴリズムを作成
    algorithm = AlgorithmVfhFuzzy(env=env)
    
    # 学習済みモデルを読み込み
    model = create_swarm_model(input_dim=32)  # 仮のinput_dim
    model.load_model(pretrained_model_path, "swarm_model")
    
    # 観測空間とアクション空間を作成
    observation_space = create_swarm_observation_space()
    action_space = create_swarm_action_space()
    
    # SwarmAgentを作成
    swarm_agent = SwarmAgent(
        env=env,
        algorithm=algorithm,
        model=model.model if hasattr(model, 'model') else model,
        action_space=action_space,
        param=param,
        system_agent=None,  # 後で設定
        swarm_id=swarm_id
    )
    
    return swarm_agent


def create_system_agent_with_pretrained_model(env, param: SystemAgentParam, 
                                            pretrained_model_path: str) -> SystemAgent:
    """
    学習済みモデルを使用してSystemAgentを作成
    """
    # 学習済みモデルを読み込み
    model = create_system_model(input_dim=32)  # 仮のinput_dim
    model.load_model(pretrained_model_path, "system_model")
    
    # 観測空間とアクション空間を作成
    observation_space = create_system_observation_space()
    action_space = create_system_action_space()
    
    # SystemAgentを作成
    system_agent = SystemAgent(
        env=env,
        algorithm=None,
        model=model.model,
        action_space=action_space,
        param=param
    )
    
    return system_agent


def create_agents_from_checkpoint(env, agent_param: AgentParam, 
                                checkpoint_path: str) -> Tuple[SystemAgent, Dict[int, SwarmAgent]]:
    """
    チェックポイントからエージェントを作成
    """
    # SystemAgentを作成（最新のチェックポイントを使用）
    system_model = create_system_model(input_dim=32)
    system_model, episode = system_model.load_latest_checkpoint(checkpoint_path, "system_model")
    
    if agent_param.system_agent_param is None:
        raise ValueError("system_agent_param is required")
    system_agent = SystemAgent(
        env=env,
        algorithm=None,
        model=system_model.model,
        action_space=create_system_action_space(),
        param=agent_param.system_agent_param
    )
    
    # SwarmAgentを作成（最新のチェックポイントを使用）
    swarm_agents = {}
    for swarm_id, swarm_param in enumerate(agent_param.swarm_agent_params):
        swarm_model = create_swarm_model(input_dim=32)
        swarm_model, _ = swarm_model.load_latest_checkpoint(checkpoint_path, f"swarm_model_{swarm_id}")
        
        # 観測空間とアクション空間を作成
        observation_space = create_swarm_observation_space()
        action_space = create_swarm_action_space()
        
        swarm_agent = SwarmAgent(
            env=env,
            algorithm=AlgorithmVfhFuzzy(env=env),
            model=swarm_model.model,
            action_space=action_space,
            param=swarm_param,
            system_agent=system_agent,
            swarm_id=swarm_id
        )
        swarm_agents[swarm_id] = swarm_agent
    
    # SystemAgentにSwarmAgentを登録
    for swarm_id, swarm_agent in swarm_agents.items():
        system_agent.register_swarm_agent(swarm_agent, swarm_id)
    
    print(f"Loaded agents from checkpoint at episode {episode}")
    return system_agent, swarm_agents


def save_agents_checkpoint(system_agent: SystemAgent, swarm_agents: Dict[int, SwarmAgent], 
                          save_dir: str, episode: int):
    """
    エージェントのチェックポイントを保存
    """
    # SystemAgentのモデルを保存
    if system_agent.model is not None:
        system_model = Model(ModelType.SYSTEM_ACTOR_CRITIC)
        system_model.model = system_agent.model
        system_model.save_checkpoint(save_dir, episode, "system_model")
    
    # SwarmAgentのモデルを保存
    for swarm_id, swarm_agent in swarm_agents.items():
        if swarm_agent.model is not None:
            swarm_model = Model(ModelType.SWARM_ACTOR_CRITIC)
            swarm_model.model = swarm_agent.model
            swarm_model.save_checkpoint(save_dir, episode, f"swarm_model_{swarm_id}")
    
    print(f"Saved agents checkpoint at episode {episode}")