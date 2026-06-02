import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_metrics(y_true, y_pred, output_cols, model_name):
    rows = []

    for i, target_name in enumerate(output_cols):
        rows.append({
            "model": model_name,
            "target": target_name,
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "RMSE": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "R2": r2_score(y_true[:, i], y_pred[:, i]),
        })

    rows.append({
        "model": model_name,
        "target": "average",
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    })

    return rows


def save_pred_plots(cfg, y_true, y_pred, model_name, output_cols):
    plot_dir = cfg["_runtime"]["plot_dir"]

    for i, target_name in enumerate(output_cols):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.8)

        min_v = min(y_true[:, i].min(), y_pred[:, i].min())
        max_v = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} - {target_name}")
        plt.tight_layout()

        save_path = os.path.join(plot_dir, f"{model_name}_{target_name}_pred_vs_true.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


def save_loss_curve(cfg, save_obj, model_name):
    if "train_losses" not in save_obj or "val_losses" not in save_obj:
        return

    plot_dir = cfg["_runtime"]["plot_dir"]

    plt.figure(figsize=(7, 5))
    plt.plot(save_obj["train_losses"], label="train")
    plt.plot(save_obj["val_losses"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"{model_name}_loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
