"""
Algorithm factory for creating different types of algorithms.
Provides centralized algorithm creation with configuration management.
"""

from typing import Dict, Any, Optional
from algorithms.vfh_fuzzy import AlgorithmVfhFuzzy
from core.logging import get_component_logger

logger = get_component_logger("algorithm_factory")


def select_algorithm(algorithm_type: str, **kwargs) -> Any:
    if algorithm_type == "vfh_fuzzy":
        return AlgorithmVfhFuzzy(**kwargs)
    # 順次追加        
    else:
        raise ValueError(f"Unknown agent type: {algorithm_type}")