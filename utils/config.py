# utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            "automl": {
                "max_training_time": 1800,
                "enable_ensemble": True,
                "enable_feature_engineering": True,
                "optimization_metric": "accuracy"
            },
            "hyperparameter_optimization": {
                "method": "Bayesian Optimization",
                "n_trials": 100,
                "cv_folds": 5,
                "enable_early_stopping": True
            },
            "model_explanation": {
                "method": "SHAP",
                "sample_size": 1000,
                "enable_confidence_intervals": True
            },
            "deployment": {
                "default_framework": "Flask API",
                "generate_dockerfile": True,
                "generate_kubernetes": False
            },
            "data_preprocessing": {
                "handle_missing_values": True,
                "encode_categorical": True,
                "scale_features": True,
                "feature_selection": False,
                "max_features": 50
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "configs/default.yaml"):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_automl_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("automl", {})

def get_optimization_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("hyperparameter_optimization", {})

def get_deployment_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("deployment", {})

def update_config(section: str, key: str, value: Any):
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_default_automl_params() -> Dict[str, Any]:
    config = load_config()
    automl = config.get("automl", {})
    
    return {
        "max_training_time": automl.get("max_training_time", 1800),
        "enable_ensemble": automl.get("enable_ensemble", True),
        "enable_feature_engineering": automl.get("enable_feature_engineering", True),
        "optimization_metric": automl.get("optimization_metric", "accuracy")
    }