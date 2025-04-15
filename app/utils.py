import os
import yaml


def load_params(config_path: str = "config.yaml") -> dict:
    """Load preprocessing parameters from a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        return config
    except KeyError:
        raise KeyError("Missing 'preprocessing' section in params.yaml")