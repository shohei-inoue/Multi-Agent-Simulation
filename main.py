"""
Main entry point for the red-group-behavior project.
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
from params.simulation import Param


def main():
    """Main application entry point"""
    logger = get_logger("main")
    logger.info("Starting red-group-behavior application...")
    
    try:
        # Initialize parameters
        param = Param()
        logger.info("Parameters initialized")
        
        # Create and setup application
        app = create_application(param)
        logger.info("Application setup completed")
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            name="default_experiment",
            description="Default swarm exploration experiment",
            parameters={
                "algorithm": param.agent.algorithm,
                "learning_enabled": param.agent.isLearning,
                "robot_count": param.explore.robotNum,
                "map_size": f"{param.environment.map.width}x{param.environment.map.height}"
            },
            num_episodes=param.agent.learningParameter.episodeNum if param.agent.learningParameter else 1,
            max_steps_per_episode=param.agent.maxStepsPerEpisode,
            save_models=True,
            save_logs=True,
            save_visualizations=config.system.enable_gif_save
        )
        
        # Run experiment
        logger.info(f"Starting experiment: {experiment_config.name}")
        app.run_experiment(experiment_config)
        
        # Get and log summary
        summary = app.get_experiment_summary()
        logger.info(f"Experiment completed. Summary: {summary}")
        
        # Cleanup
        app.cleanup()
        logger.info("Application cleanup completed")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise
    
    logger.info("Application finished successfully")


if __name__ == "__main__":
    main()