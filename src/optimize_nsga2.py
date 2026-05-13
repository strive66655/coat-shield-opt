import torch
import os
import argparse
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from models.residual_model import ResidualMLP


def load_residual_mlp_model(model_path: str):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    params = checkpoint["params"]

    model = ResidualMLP(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=params["hidden_dim"],
        num_blocks=params["num_blocks"],
        dropout=params["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_improvement(model, checkpoint, X: np.ndarray):

    x_scaler = checkpoint["x_scaler"]
    y_scaler = checkpoint["y_scaler"]

    X_scaled = x_scaler.transform(X)

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_scaled = model(X_tensor).numpy()

    y_pred = y_scaler.inverse_transform(y_scaled)

    return y_pred


class CoatingShieldProblem(ElementwiseProblem):

    def __init__(
        self,
        model,
        checkpoint,
        al_thickness_um: float = 300.0,
        b4c_bounds=(0.0, 450.0),
        w_bounds=(0.0, 60.0),
        zno_bounds=(0.0, 200.0),
        max_coating_thickness_um: float = 500.0,
        min_required_improvement: float = 0.5,
    ):
        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([b4c_bounds[0], w_bounds[0], zno_bounds[0]], dtype=float),
            xu=np.array([b4c_bounds[1], w_bounds[1], zno_bounds[1]], dtype=float),
        )

        self.model = model
        self.checkpoint = checkpoint
        self.al_thickness_um = al_thickness_um
        self.max_coating_thickness_um = max_coating_thickness_um
        self.min_required_improvement = min_required_improvement

    def _evaluate(self, x, out, *args, **kwargs):
        b4c = float(x[0])
        w = float(x[1])
        zno = float(x[2])

        coating_thickness = b4c + w + zno
        X_model = np.array([[self.al_thickness_um, b4c, w, zno]], dtype=float)
        y_pred = predict_improvement(self.model, self.checkpoint, X_model)[0]

        improvement_1mev = float(y_pred[0])
        improvement_5mev = float(y_pred[1])
        improvement_10mev = float(y_pred[2])

        min_improvement = min(improvement_1mev, improvement_5mev, improvement_10mev)

        # pymoo 默认最小化
        # 目标 1：最大化 min_improvement，所以写成负号
        f1 = -min_improvement

        # 目标 2：最小化新增涂层厚度
        f2 = coating_thickness

        # 约束 1：新增涂层厚度 <= 500 μm
        g1 = coating_thickness - self.max_coating_thickness_um

        # 约束 2：三个能量点中最小提升率 >= 0.5
        g2 = self.min_required_improvement - min_improvement

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

def run_nsga2(
    model_path: str,
    output_dir: str,
    al_thickness_um: float = 300.0,
    max_coating_thickness_um: float = 500.0,
    min_required_improvement: float = 0.5,
    b4c_bounds=(0.0, 450.0),
    w_bounds=(0.0, 60.0),
    zno_bounds=(50.0, 200.0),
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    model, checkpoint = load_residual_mlp_model(model_path)

    problem = CoatingShieldProblem(
        model=model,
        checkpoint=checkpoint,
        al_thickness_um=al_thickness_um,
        b4c_bounds=b4c_bounds,
        w_bounds=w_bounds,
        zno_bounds=zno_bounds,
        max_coating_thickness_um=max_coating_thickness_um,
        min_required_improvement=min_required_improvement,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    result = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    X_opt = result.X
    F_opt = result.F

    if X_opt is None or len(X_opt) == 0:
        print("\nNSGA-II 没有找到候选解。")
        return None, None

    rows = []

    for i in range(len(X_opt)):
        b4c = float(X_opt[i, 0])
        w = float(X_opt[i, 1])
        zno = float(X_opt[i, 2])

        coating_thickness = b4c + w + zno
        X_model = np.array([[al_thickness_um, b4c, w, zno]], dtype=float)
        y_pred = predict_improvement(model, checkpoint, X_model)[0]

        improvement_1mev = float(y_pred[0])
        improvement_5mev = float(y_pred[1])
        improvement_10mev = float(y_pred[2])

        min_improvement = min(improvement_1mev, improvement_5mev, improvement_10mev )

        mean_improvement = float(np.mean(y_pred))

        pass_coating_thickness = coating_thickness <= max_coating_thickness_um
        pass_50_all_energy = min_improvement >= min_required_improvement
        is_feasible = pass_coating_thickness and pass_50_all_energy

        rows.append(
            {
                "Al_thickness_um": al_thickness_um,
                "B4C_thickness_um": b4c,
                "W_thickness_um": w,
                "ZnO_thickness_um": zno,
                "coating_thickness_um": coating_thickness,
                "pred_improvement_1Mev": improvement_1mev,
                "pred_improvement_5Mev": improvement_5mev,
                "pred_improvement_10Mev": improvement_10mev,
                "pred_min_improvement": min_improvement,
                "pred_mean_improvement": mean_improvement,
                "pass_coating_thickness": pass_coating_thickness,
                "pass_50_all_energy": pass_50_all_energy,
                "is_feasible": is_feasible,
                "objective_1_neg_min_improvement": float(F_opt[i, 0]),
                "objective_2_coating_thickness": float(F_opt[i, 1]),
            }
        )

    df = pd.DataFrame(rows)

    # 排序逻辑：
    # 1. 可行方案优先
    # 2. 最小提升率越大越好
    # 3. 涂层厚度越小越好
    df_sorted = df.sort_values(
        by=["is_feasible", "pred_min_improvement", "coating_thickness_um"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    pareto_path = os.path.join(output_dir, "nsga2_pareto_candidates.csv")
    df_sorted.to_csv(pareto_path, index=False, encoding="utf-8-sig")

    feasible_df = df_sorted[df_sorted["is_feasible"]].copy()
    feasible_path = os.path.join(output_dir, "nsga2_feasible_candidates.csv")
    feasible_df.to_csv(feasible_path, index=False, encoding="utf-8-sig")

    print("\n====== NSGA-II 优化完成 ======")
    print("Pareto 候选数量:", len(df_sorted))
    print("可行候选数量:", len(feasible_df))
    print("Pareto 候选保存到:", pareto_path)
    print("可行候选保存到:", feasible_path)

    if len(feasible_df) > 0:
        print("\n====== 推荐前 10 个可行方案 ======")
        print(
            feasible_df[
                [
                    "Al_thickness_um",
                    "B4C_thickness_um",
                    "W_thickness_um",
                    "ZnO_thickness_um",
                    "coating_thickness_um",
                    "pred_improvement_1Mev",
                    "pred_improvement_5Mev",
                    "pred_improvement_10Mev",
                    "pred_min_improvement",
                    "pred_mean_improvement",
                ]
            ].head(10)
        )
    else:
        print(
            "\n没有找到同时满足："
            "\n1. B4C + W + ZnO <= 500 μm"
            "\n2. 三个能量点最小提升率 >= 0.5"
            "\n的方案。"
        )

    return df_sorted, feasible_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/exp_baseline_e/models/ResidualMLP_improvement_surrogate.pth",
        help="训练好的 ResidualMLP 模型路径",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/exp_nsga2_resmlp",
        help="NSGA-II 优化结果输出目录",
    )

    parser.add_argument(
        "--al_thickness_um",
        type=float,
        default=300.0,
        help="固定 Al 厚度，单位 μm。该值只作为代理模型输入，不计入涂层厚度约束。",
    )

    parser.add_argument(
        "--max_coating_thickness_um",
        type=float,
        default=500.0,
        help="新增涂层厚度上限，单位 μm。只计算 B4C + W + ZnO。",
    )

    parser.add_argument(
        "--min_required_improvement",
        type=float,
        default=0.5,
        help="最小屏蔽提升率要求。0.5 表示 50%。",
    )

    parser.add_argument(
        "--pop_size",
        type=int,
        default=100,
        help="NSGA-II 种群大小",
    )

    parser.add_argument(
        "--n_gen",
        type=int,
        default=200,
        help="NSGA-II 迭代代数",
    )

    args = parser.parse_args()

    run_nsga2(
        model_path=args.model_path,
        output_dir=args.output_dir,
        al_thickness_um=args.al_thickness_um,
        max_coating_thickness_um=args.max_coating_thickness_um,
        min_required_improvement=args.min_required_improvement,

        b4c_bounds=(0.0, 450.0),
        w_bounds=(0.0, 60.0),
        zno_bounds=(50.0, 200.0),

        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=42,
    )


if __name__ == "__main__":
    main()