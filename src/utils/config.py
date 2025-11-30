"""
Configuration utilities
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON or YAML file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            yaml.dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)

