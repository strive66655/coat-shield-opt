import os
import shutil
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dirs(cfg: dict, config_path: str) -> dict:
    exp_name = cfg["experiment"]["name"]
    output_root = cfg["experiment"]["output_root"]

    exp_dir = os.path.join(output_root, exp_name)
    model_dir = os.path.join(exp_dir, "models")
    plot_dir = os.path.join(exp_dir, "plots")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    cfg["_runtime"] = {
        "exp_dir": exp_dir,
        "model_dir": model_dir,
        "plot_dir": plot_dir,
    }
    
    if cfg["output"].get("save_config_copy", True):
        shutil.copy(config_path, os.path.join(exp_dir, "config_used.yaml"))

    return cfg