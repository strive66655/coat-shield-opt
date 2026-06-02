import argparse
import copy
import json
import os
import sys
import warnings

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.residual_model import ResidualMLP
from src.utils.config_loader import load_config, prepare_dirs
from src.utils.dataset import build_dataset, load_data
from src.utils.metrics import save_loss_curve, save_pred_plots


warnings.filterwarnings("ignore")


def train_residual_mlp_task(cfg, X_train, y_train, X_test, y_test, task_name):
    params = cfg["models"]["ResidualMLP"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    model = ResidualMLP(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=params["hidden_dim"],
        num_blocks=params["num_blocks"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    loss_fn = nn.MSELoss()
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_state = None
    patience_count = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_dataset)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test_t), y_test_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"{task_name} epoch={epoch:04d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

        if patience_count >= params["patience"]:
            print(f"{task_name} early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_s = model(X_test_t).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_s)

    save_obj = {
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "output_dim": y_train.shape[1],
        "params": params,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return model, y_pred, save_obj


def save_residual_mlp_model(cfg, save_obj, model_name, input_cols, output_cols):
    save_obj["input_cols"] = input_cols
    save_obj["output_cols"] = output_cols
    save_obj["target_type"] = cfg["target"]["type"]

    if cfg["output"].get("save_models", True):
        save_path = os.path.join(
            cfg["_runtime"]["model_dir"],
            f"{model_name}_improvement_surrogate.pth",
        )
        torch.save(save_obj, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config yaml.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = prepare_dirs(cfg, args.config)

    df = load_data(cfg)
    model_tasks = cfg["target"].get("model_tasks")
    if not model_tasks:
        model_tasks = [
            {
                "model_name": cfg["target"].get("model_name", "ResidualMLP"),
                "output_cols": cfg["target"]["output_cols"],
            }
        ]

    all_metrics = []
    used_data_frames = []
    input_cols = cfg["features"]["input_cols"]
    all_output_cols = []

    for task in model_tasks:
        model_name = task["model_name"]
        output_cols = task["output_cols"]
        task_cfg = dict(cfg)
        task_cfg["target"] = dict(cfg["target"])
        task_cfg["target"]["output_cols"] = output_cols

        X, y, input_cols, output_cols, used_data = build_dataset(df, task_cfg)
        used_data_frames.append(used_data)
        all_output_cols.extend(output_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg["experiment"]["test_size"],
            random_state=cfg["experiment"]["random_state"],
        )

        print(f"\n====== Train ResidualMLP: {model_name} ======")
        _, y_pred, save_obj = train_residual_mlp_task(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_name=model_name,
        )

        for idx, output_col in enumerate(output_cols):
            all_metrics.append(
                {
                    "model": model_name,
                    "target": output_col,
                    "MAE": mean_absolute_error(y_test[:, idx], y_pred[:, idx]),
                    "R2": r2_score(y_test[:, idx], y_pred[:, idx]),
                    "best_val_loss_scaled": (
                        min(save_obj["val_losses"]) if save_obj["val_losses"] else None
                    ),
                    "epochs_run": len(save_obj["train_losses"]),
                }
            )

        save_residual_mlp_model(
            cfg=cfg,
            save_obj=save_obj,
            model_name=model_name,
            input_cols=input_cols,
            output_cols=output_cols,
        )

        if cfg["output"].get("save_plots", True):
            save_pred_plots(cfg, y_test, y_pred, model_name, output_cols)
            save_loss_curve(cfg, save_obj, model_name)

    exp_dir = cfg["_runtime"]["exp_dir"]
    metrics_df = pd.DataFrame(all_metrics)

    if cfg["output"].get("save_metrics", True):
        metrics_df.to_csv(
            os.path.join(exp_dir, "surrogate_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if cfg["output"].get("save_clean_dataset", True):
        used_dataset = pd.concat(used_data_frames, axis=1)
        used_dataset = used_dataset.loc[:, ~used_dataset.columns.duplicated()]
        used_dataset.to_csv(
            os.path.join(exp_dir, "used_dataset.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    meta = {
        "input_cols": input_cols,
        "output_cols": list(dict.fromkeys(all_output_cols)),
        "model_tasks": model_tasks,
        "num_samples": int(len(df)),
        "num_features": int(len(input_cols)),
        "target_type": cfg["target"]["type"],
    }

    with open(os.path.join(exp_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    print("\n====== Training complete ======")
    print(metrics_df)
    print("\nOutput directory:", exp_dir)


if __name__ == "__main__":
    main()
