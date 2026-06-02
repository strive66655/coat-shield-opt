import numpy as np
import pandas as pd


def load_data(cfg: dict) -> pd.DataFrame:
    file_path = cfg["data"]["file_path"]
    file_type = cfg["data"].get("file_type", "csv")

    if file_type != "csv":
        raise ValueError(f"当前脚本只支持 csv，收到 file_type={file_type}")

    df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    print("\n====== 原始数据列名 ======")
    print(df.columns.tolist())

    return df


def build_dataset(df: pd.DataFrame, cfg: dict):
    input_cols = cfg["features"]["input_cols"]
    output_cols = cfg["target"]["output_cols"]

    missing_cols = [c for c in input_cols + output_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"以下列名在数据中找不到：{missing_cols}\n"
            f"当前数据列名为：{df.columns.tolist()}"
        )

    used_cols = input_cols + output_cols
    data = df[used_cols].copy()

    for col in used_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[input_cols] = data[input_cols].fillna(0)
    data = data.dropna(subset=output_cols)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if all(c in data.columns for c in ["B4C_thickness_um", "W_thickness_um", "ZnO_thickness_um"]):
        data["added_thickness_um"] = (
            data["B4C_thickness_um"]
            + data["W_thickness_um"]
            + data["ZnO_thickness_um"]
        )

    data["min_improvement"] = data[output_cols].min(axis=1)
    data["mean_improvement_calc"] = data[output_cols].mean(axis=1)
    data["pass_50_all_energy"] = data["min_improvement"] >= 0.5

    X = data[input_cols].values
    y = data[output_cols].values

    print("\n====== 数据集信息 ======")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("输入列：", input_cols)
    print("输出列：", output_cols)

    print("\n====== 目标值范围 ======")
    for i, col in enumerate(output_cols):
        print(
            f"{col}: min={y[:, i].min():.6f}, "
            f"max={y[:, i].max():.6f}, "
            f"mean={y[:, i].mean():.6f}"
        )

    print("\n====== 50% 提升统计 ======")
    print("所有目标都 >= 0.5 的样本数：", int(data["pass_50_all_energy"].sum()))
    print("总样本数：", len(data))

    return X, y, input_cols, output_cols, data
