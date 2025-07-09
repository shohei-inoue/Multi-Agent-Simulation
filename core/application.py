"""
Main application class that orchestrates the entire system.
Provides a clean interface for running simulations and experiments.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from dataclasses import dataclass

from core.config import config, ConfigManager
from core.logging import Logger, get_component_logger
from core.factories import (
    algorithm_factory, agent_factory, model_factory, environment_factory
)
from params.simulation import Param


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str
    parameters: Dict[str, Any]
    num_episodes: int
    max_steps_per_episode: int
    save_models: bool = True
    save_logs: bool = True
    save_visualizations: bool = True


class Application:
    """Main application class"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or config
        self.logger = Logger("application")
        self.component_logger = get_component_logger("application")
        
        # Components
        self.environment = None
        self.agent = None
        self.algorithm = None
        self.model = None
        
        # Experiment tracking
        self.current_experiment: Optional[ExperimentConfig] = None
        self.experiment_results: List[Dict[str, Any]] = []
    
    def setup(self, params: Param):
        """Setup the application with parameters"""
        self.component_logger.log_component_event("setup_started", {"params": str(params)})
        
        # Create log directories
        self.config.create_log_directories()
        
        # Create environment
        self.environment = environment_factory.create(
            "exploration", 
            param=params
        )
        
        # Create algorithm
        self.algorithm = algorithm_factory.create(
            params.agent.algorithm,
            env=self.environment
        )
        
        # Create model if learning is enabled
        if params.agent.isLearning and params.agent.learningParameter:
            self.model = model_factory.create(
                params.agent.learningParameter.model,
                input_dim=32  # This should be calculated from observation space
            )
        
        # Create agent
        agent_kwargs = {
            "env": self.environment,
            "algorithm": self.algorithm,
            "model": self.model,
            "optimizer": params.agent.learningParameter.optimizer if params.agent.learningParameter else "adam",
            "gamma": params.agent.learningParameter.gamma if params.agent.learningParameter else 0.99,
            "n_steps": params.agent.learningParameter.nStep if params.agent.learningParameter else 5,
            "max_steps_per_episode": params.agent.maxStepsPerEpisode,
            "action_space": self.environment.action_space
        }
        
        self.agent = agent_factory.create_agent(
            "a2c" if params.agent.isLearning else "simple",
            **agent_kwargs
        )
        
        self.component_logger.log_component_event("setup_completed")
    
    def run_experiment(self, experiment_config: ExperimentConfig):
        """Run a complete experiment"""
        self.current_experiment = experiment_config
        self.component_logger.log_component_event(
            "experiment_started", 
            {"experiment": experiment_config.name}
        )
        
        start_time = time.time()
        results = []
        
        try:
            for episode in range(experiment_config.num_episodes):
                episode_result = self._run_episode(episode)
                results.append(episode_result)
                
                # Log progress
                if episode % 10 == 0:
                    self.component_logger.log_component_event(
                        "episode_completed",
                        {"episode": episode, "total_episodes": experiment_config.num_episodes}
                    )
        
        except Exception as e:
            self.component_logger.log_component_event("experiment_failed", {"error": str(e)})
            raise
        
        finally:
            end_time = time.time()
            experiment_summary = {
                "experiment_name": experiment_config.name,
                "duration": end_time - start_time,
                "total_episodes": len(results),
                "results": results
            }
            
            self.experiment_results.append(experiment_summary)
            self._save_experiment_results(experiment_summary)
            
            self.component_logger.log_component_event(
                "experiment_completed",
                {"duration": experiment_summary["duration"]}
            )
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single episode"""
        if self.environment is None or self.agent is None:
            raise RuntimeError("Environment and agent must be initialized before running episodes")
        
        # log_dirの決定
        log_dir = getattr(self.config, 'simulation', None)
        if log_dir and hasattr(log_dir, 'log_dir'):
            log_dir = log_dir.log_dir
        else:
            log_dir = "logs"

        # A2CAgentのtrain_one_episodeを呼び出し、保存処理を一元化
        if hasattr(self.agent, "train_one_episode"):
            total_reward = self.agent.train_one_episode(episode=episode, log_dir=log_dir)
            step_count = getattr(self.agent, "step_count", 0)
            exploration_rate = getattr(self.environment, "exploration_ratio", 0.0)
            final_exploration_rate = getattr(self.environment.scorer, "exploration_rate", [0.0])
            if isinstance(final_exploration_rate, list) and final_exploration_rate:
                final_exploration_rate = final_exploration_rate[-1]
            elif hasattr(final_exploration_rate, "__getitem__") and len(final_exploration_rate) > 0:
                final_exploration_rate = final_exploration_rate[-1]
            else:
                final_exploration_rate = 0.0
            episode_result = {
                "episode": episode,
                "total_reward": total_reward,
                "steps": step_count,
                "duration": 0.0,  # 必要なら計測
                "exploration_rate": exploration_rate,
                "final_exploration_rate": final_exploration_rate
            }
            self.component_logger.log_episode(episode_result)
            return episode_result

        # --- fallback: 従来の処理（学習しない場合など） ---
        episode_start_time = time.time()
        state = self.environment.reset()
        total_reward = 0
        step_count = 0
        max_steps = self.current_experiment.max_steps_per_episode if self.current_experiment else 1000
        while step_count < max_steps:
            action_tensor, action_dict = self.agent.get_action(
                state, episode, log_dir=log_dir
            )
            if hasattr(self.agent, 'capture_frame'):
                self.agent.capture_frame()
            next_state, reward, done, truncated, info = self.environment.step(action_dict)
            state = next_state
            total_reward += reward
            step_count += 1
            if done:
                break
        episode_duration = time.time() - episode_start_time
        episode_result = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": step_count,
            "duration": episode_duration,
            "exploration_rate": self.environment.exploration_ratio,
            "final_exploration_rate": self.environment.scorer.exploration_rate[-1] if (hasattr(self.environment.scorer, "exploration_rate") and self.environment.scorer.exploration_rate and len(self.environment.scorer.exploration_rate) > 0) else 0.0
        }
        self.component_logger.log_episode(episode_result)
        return episode_result
    
    def _save_experiment_results(self, experiment_summary: Dict[str, Any]):
        """Save experiment results"""
        if not self.current_experiment or not self.current_experiment.save_logs:
            return
        
        # Save to JSON
        results_file = self.config.get_log_path(
            "csvs", 
            f"experiment_{self.current_experiment.name if self.current_experiment else 'unknown'}.json"
        )
        
        import json
        with open(results_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2, default=str)
        
        # Save metrics
        experiment_name = self.current_experiment.name if self.current_experiment else 'unknown'
        self.logger.save_metrics(f"experiment_{experiment_name}_metrics.json")
        self.logger.save_episodes(f"experiment_{experiment_name}_episodes.csv")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        if not self.experiment_results:
            return {"message": "No experiments run yet"}
        
        total_experiments = len(self.experiment_results)
        total_episodes = sum(len(exp["results"]) for exp in self.experiment_results)
        total_duration = sum(exp["duration"] for exp in self.experiment_results)
        
        return {
            "total_experiments": total_experiments,
            "total_episodes": total_episodes,
            "total_duration": total_duration,
            "experiments": [exp["experiment_name"] for exp in self.experiment_results]
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.component_logger.log_component_event("cleanup_started")
        
        # Save final logs
        self.logger.save_metrics()
        self.logger.save_episodes()
        
        # Close any open resources
        if self.environment and hasattr(self.environment, 'close'):
            self.environment.close()
        
        self.component_logger.log_component_event("cleanup_completed")


def create_application(params: Param) -> Application:
    """Factory function to create application"""
    app = Application()
    app.setup(params)
    return app 