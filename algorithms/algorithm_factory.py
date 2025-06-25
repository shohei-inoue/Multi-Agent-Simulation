from algorithms.vfh_fuzzy import AlgorithmVfhFuzzy

def select_algorithm(algorithm_type: str, **kwargs):
    if algorithm_type == "vfh_fuzzy":
        return AlgorithmVfhFuzzy(**kwargs)
    # 順次追加        
    else:
        raise ValueError(f"Unknown agent type: {algorithm_type}")