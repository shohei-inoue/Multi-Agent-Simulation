from agents.agent_a2c import A2CAgent
import tensorflow as tf

def create_agent(agent_type: str, **kwargs):
    if agent_type == "a2c":
        # --- オプティマイザ変換 ---
        if isinstance(kwargs["optimizer"], str):
            kwargs["optimizer"] = create_optimizer(kwargs["optimizer"], kwargs["learning_late"])
        
        return A2CAgent(
            env=kwargs["env"],
            algorithm=kwargs["algorithm"],
            model=kwargs["model"],
            optimizer=kwargs["optimizer"],
            gamma=kwargs["gamma"],
            n_steps=kwargs["n_steps"],
            max_steps_per_episode=kwargs["max_steps_per_episode"],
            action_space=kwargs["action_space"]
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_optimizer(optimizer_name: str, learning_rate: float):
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")