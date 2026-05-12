import argparse
import json
import os
import sys
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from src.models.sklearn_model import save_sklearn_model, train_random_forest, train_xgboost
from src.train.mlp_trainer import train_residual_mlp
from src.utils.config_loader import load_config, prepare_dirs
from src.utils.dataset import build_dataset, load_data
from src.utils.metrics import calc_metrics, save_loss_curve, save_pred_plots

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/surrogate_config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = prepare_dirs(cfg, args.config)

    df = load_data(cfg)
    X, y, input_cols, output_cols, used_data = build_dataset(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["experiment"]["test_size"],
        random_state=cfg["experiment"]["random_state"],
    )

    enabled = cfg["models"]["enabled"]
    all_metrics = []

    if enabled.get("RandomForest", False):
        print("\n====== 训练 RandomForest ======")

        rf_model = train_random_forest(cfg, X_train, y_train)
        y_pred = rf_model.predict(X_test)

        all_metrics.extend(calc_metrics(y_test, y_pred, output_cols, "RandomForest"))

        if cfg["output"].get("save_models", True):
            save_sklearn_model(cfg, rf_model, "RandomForest", input_cols, output_cols)

        if cfg["output"].get("save_plots", True):
            save_pred_plots(cfg, y_test, y_pred, "RandomForest", output_cols)

    if enabled.get("XGBoost", False):
        print("\n====== 训练 XGBoost ======")

        xgb_model = train_xgboost(cfg, X_train, y_train)

        if xgb_model is not None:
            y_pred = xgb_model.predict(X_test)

            all_metrics.extend(calc_metrics(y_test, y_pred, output_cols, "XGBoost"))

            if cfg["output"].get("save_models", True):
                save_sklearn_model(cfg, xgb_model, "XGBoost", input_cols, output_cols)

            if cfg["output"].get("save_plots", True):
                save_pred_plots(cfg, y_test, y_pred, "XGBoost", output_cols)

    if enabled.get("ResidualMLP", False):
        print("\n====== 训练 ResidualMLP ======")

        _, y_pred, save_obj = train_residual_mlp(
            cfg,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        all_metrics.extend(calc_metrics(y_test, y_pred, output_cols, "ResidualMLP"))

        save_obj["input_cols"] = input_cols
        save_obj["output_cols"] = output_cols
        save_obj["target_type"] = cfg["target"]["type"]

        if cfg["output"].get("save_models", True):
            torch.save(
                save_obj,
                os.path.join(
                    cfg["_runtime"]["model_dir"],
                    "ResidualMLP_improvement_surrogate.pth",
                ),
            )

        if cfg["output"].get("save_plots", True):
            save_pred_plots(cfg, y_test, y_pred, "ResidualMLP", output_cols)
            save_loss_curve(cfg, save_obj, "ResidualMLP")

    exp_dir = cfg["_runtime"]["exp_dir"]
    metrics_df = pd.DataFrame(all_metrics)

    if cfg["output"].get("save_metrics", True):
        metrics_df.to_csv(
            os.path.join(exp_dir, "surrogate_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if cfg["output"].get("save_clean_dataset", True):
        used_data.to_csv(
            os.path.join(exp_dir, "used_dataset.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    meta = {
        "input_cols": input_cols,
        "output_cols": output_cols,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "target_type": cfg["target"]["type"],
        "note": "Surrogate model predicts shielding improvement at 1MeV, 5MeV and 10MeV.",
    }

    with open(os.path.join(exp_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    print("\n====== 实验完成 ======")
    print(metrics_df)
    print("\n结果目录：", exp_dir)


if __name__ == "__main__":
    main()