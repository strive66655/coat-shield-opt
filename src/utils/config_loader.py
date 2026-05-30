import os
import shutil
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = resolve_active_dataset(config)
    return config


def resolve_active_dataset(config: dict) -> dict:
    datasets = config.get("datasets")
    if not datasets:
        return config

    active_dataset = config.get("active_dataset")
    if not active_dataset:
        raise KeyError("config.yaml contains datasets but misses active_dataset")
    if active_dataset not in datasets:
        raise KeyError(
            f"active_dataset={active_dataset} is not defined. "
            f"Available datasets: {list(datasets.keys())}"
        )

    selected = datasets[active_dataset]

    experiment = dict(config.get("experiment", {}))
    experiment.update(selected.get("experiment", {}))

    config["experiment"] = experiment
    config["data"] = selected["data"]
    config["target"] = selected["target"]
    config["_active_dataset"] = active_dataset

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
