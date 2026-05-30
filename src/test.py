import argparse
import os
import sys

import torch
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.residual_model import ResidualMLP


DEFAULT_DOMAIN_KEY = "__all__"
AL_DENSITY_G_CM3 = 2.7
B4C_DENSITY_G_CM3 = 2.52
W_DENSITY_G_CM3 = 19.35
ZNO_DENSITY_G_CM3 = 5.61
UM_TO_CM = 0.0001


def format_zno_key(zno: float):
    return int(zno) if float(zno).is_integer() else float(zno)


def calc_areal_density(
    al_thickness_um: float,
    b4c_thickness_um: float,
    w_thickness_um: float,
    zno_thickness_um: float,
):
    al_areal_density = AL_DENSITY_G_CM3 * UM_TO_CM * al_thickness_um
    b4c_areal_density = B4C_DENSITY_G_CM3 * UM_TO_CM * b4c_thickness_um
    w_areal_density = W_DENSITY_G_CM3 * UM_TO_CM * w_thickness_um
    zno_areal_density = ZNO_DENSITY_G_CM3 * UM_TO_CM * zno_thickness_um
    base_areal_density = al_areal_density + zno_areal_density
    added_coating_areal_density = b4c_areal_density + w_areal_density
    total_areal_density = base_areal_density + added_coating_areal_density
    coating_mass_increase_ratio = added_coating_areal_density / base_areal_density

    return {
        "Al_areal_density_g_cm2": al_areal_density,
        "B4C_areal_density_g_cm2": b4c_areal_density,
        "W_areal_density_g_cm2": w_areal_density,
        "ZnO_areal_density_g_cm2": zno_areal_density,
        "base_areal_density_g_cm2": base_areal_density,
        "added_coating_areal_density_g_cm2": added_coating_areal_density,
        "total_areal_density_g_cm2": total_areal_density,
        "coating_mass_increase_ratio": coating_mass_increase_ratio,
    }


def min_max_normalize(series):
    min_value = float(series.min())
    max_value = float(series.max())
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def add_tradeoff_scores(df: pd.DataFrame):
    if len(df) == 0:
        return df

    df = df.copy()
    df["efficiency_score"] = min_max_normalize(df["selected_energy_improvement"])
    df["mass_score"] = 1.0 - min_max_normalize(df["total_areal_density_g_cm2"])
    df["thickness_score"] = 1.0 - min_max_normalize(df["coating_thickness_um"])
    df["tradeoff_score"] = (
        df["efficiency_score"] + df["mass_score"] + df["thickness_score"]
    ) / 3.0
    df["efficiency_weighted_tradeoff_score"] = (
        0.5 * df["efficiency_score"]
        + 0.25 * df["mass_score"]
        + 0.25 * df["thickness_score"]
    )
    return df


def select_representative_candidates(feasible_df: pd.DataFrame):
    if len(feasible_df) == 0:
        return pd.DataFrame()

    rows = []
    group_cols = ["ZnO_thickness_um", "energy_target"]
    selectors = [
        (
            "max_efficiency",
            ["selected_energy_improvement", "total_areal_density_g_cm2", "coating_thickness_um"],
            [False, True, True],
        ),
        (
            "min_total_mass",
            ["total_areal_density_g_cm2", "selected_energy_improvement", "coating_thickness_um"],
            [True, False, True],
        ),
        (
            "min_thickness",
            ["coating_thickness_um", "selected_energy_improvement", "total_areal_density_g_cm2"],
            [True, False, True],
        ),
        (
            "balanced_tradeoff",
            [
                "efficiency_weighted_tradeoff_score",
                "tradeoff_score",
                "selected_energy_improvement",
                "total_areal_density_g_cm2",
                "coating_thickness_um",
            ],
            [False, False, False, True, True],
        ),
    ]

    for _, group_df in feasible_df.groupby(group_cols, sort=False):
        for recommendation_type, by_cols, ascending in selectors:
            row = group_df.sort_values(by=by_cols, ascending=ascending).iloc[0].copy()
            row["recommendation_type"] = recommendation_type
            rows.append(row)

    recommendation_df = pd.DataFrame(rows)
    if len(recommendation_df) == 0:
        return recommendation_df

    duplicate_key_cols = [
        "ZnO_thickness_um",
        "energy_target",
        "B4C_thickness_um",
        "W_thickness_um",
        "coating_thickness_um",
        "selected_energy_improvement",
    ]
    recommendation_df["recommendation_type"] = recommendation_df.groupby(
        duplicate_key_cols
    )["recommendation_type"].transform(lambda values: " + ".join(sorted(set(values))))
    recommendation_df = recommendation_df.drop_duplicates(subset=duplicate_key_cols)
    return recommendation_df.sort_values(
        by=["ZnO_thickness_um", "energy_index", "recommendation_type"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def clean_csv_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def readable_prediction_columns(df: pd.DataFrame):
    columns = {}
    for col in df.columns:
        if not col.startswith("pred_improvement_"):
            continue
        energy_label = col.replace("pred_improvement_", "").replace("Mev", "MeV")
        columns[f"pred_{energy_label}"] = df[col]
    return columns


def build_readable_recommendation_df(recommendation_df: pd.DataFrame):
    if len(recommendation_df) == 0:
        return recommendation_df

    df = clean_csv_columns(recommendation_df)
    readable_cols = {
        "ZnO_um": df["ZnO_thickness_um"],
        "energy": df["energy_target"],
        "recommendation": df["recommendation_type"],
        "Al_um": df["Al_thickness_um"],
        "B4C_um": df["B4C_thickness_um"],
        "W_um": df["W_thickness_um"],
        "B4C_plus_W_um": df["B4C_thickness_um"] + df["W_thickness_um"],
        "total_thickness_um": df["coating_thickness_um"],
        "base_areal_density_g_cm2": df["base_areal_density_g_cm2"],
        "added_areal_density_g_cm2": df["added_coating_areal_density_g_cm2"],
        "total_areal_density_g_cm2": df["total_areal_density_g_cm2"],
        "mass_increase_pct": df["coating_mass_increase_ratio"] * 100.0,
        "selected_improvement": df["selected_energy_improvement"],
        **readable_prediction_columns(df),
        "tradeoff_score": df["tradeoff_score"],
        "eff_weighted_score": df["efficiency_weighted_tradeoff_score"],
    }
    readable = pd.DataFrame(readable_cols)

    round_map = {
        "ZnO_um": 0,
        "Al_um": 0,
        "B4C_um": 3,
        "W_um": 3,
        "B4C_plus_W_um": 3,
        "total_thickness_um": 3,
        "base_areal_density_g_cm2": 6,
        "added_areal_density_g_cm2": 6,
        "total_areal_density_g_cm2": 6,
        "mass_increase_pct": 2,
        "selected_improvement": 4,
        "tradeoff_score": 3,
        "eff_weighted_score": 3,
    }
    for col in readable.columns:
        if col.startswith("pred_"):
            round_map[col] = 4

    return readable.round(round_map)


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

    return y_scaler.inverse_transform(y_scaled)


def load_training_domain_bounds(
    training_domain_path: str,
    margin_um: float = 0.0,
    group_by_zno: bool = True,
):
    if not training_domain_path:
        return None
    if not os.path.exists(training_domain_path):
        print(f"Training domain file not found, fallback to default bounds: {training_domain_path}")
        return None

    df = pd.read_csv(training_domain_path)
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = ["B4C_thickness_um", "W_thickness_um", "ZnO_thickness_um"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Training domain file misses columns {missing_cols}, fallback to default bounds.")
        return None

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols)

    def make_bounds(group_df):
        return {
            "b4c_bounds": (
                max(0.0, float(group_df["B4C_thickness_um"].min()) - margin_um),
                float(group_df["B4C_thickness_um"].max()) + margin_um,
            ),
            "w_bounds": (
                max(0.0, float(group_df["W_thickness_um"].min()) - margin_um),
                float(group_df["W_thickness_um"].max()) + margin_um,
            ),
        }

    bounds_by_zno = {}

    if not group_by_zno:
        bounds_by_zno[DEFAULT_DOMAIN_KEY] = make_bounds(df)
        bounds = bounds_by_zno[DEFAULT_DOMAIN_KEY]
        print("\n====== Training-domain bounds, global ======")
        print(f"B4C={bounds['b4c_bounds']} | W={bounds['w_bounds']}")
        return bounds_by_zno

    for zno, group_df in df.groupby("ZnO_thickness_um"):
        bounds_by_zno[format_zno_key(float(zno))] = make_bounds(group_df)

    print("\n====== Training-domain bounds by ZnO ======")
    for zno_key, bounds in bounds_by_zno.items():
        print(
            f"ZnO={zno_key} um | "
            f"B4C={bounds['b4c_bounds']} | "
            f"W={bounds['w_bounds']}"
        )

    return bounds_by_zno


class FixedZnOEnergyProblem(ElementwiseProblem):
    def __init__(
        self,
        model,
        checkpoint,
        fixed_zno_thickness_um: float,
        energy_index: int,
        al_thickness_um: float = 300.0,
        b4c_bounds=(0.0, 450.0),
        w_bounds=(0.0, 60.0),
        max_coating_thickness_um: float = 500.0,
        max_coating_mass_increase_ratio: float = 0.2,
        min_required_improvement: float = 0.5,
    ):
        super().__init__(
            n_var=2,
            n_obj=3,
            n_ieq_constr=3,
            xl=np.array([b4c_bounds[0], w_bounds[0]], dtype=float),
            xu=np.array([b4c_bounds[1], w_bounds[1]], dtype=float),
        )

        self.model = model
        self.checkpoint = checkpoint
        self.fixed_zno_thickness_um = fixed_zno_thickness_um
        self.energy_index = energy_index
        self.al_thickness_um = al_thickness_um
        self.max_coating_thickness_um = max_coating_thickness_um
        self.max_coating_mass_increase_ratio = max_coating_mass_increase_ratio
        self.min_required_improvement = min_required_improvement

    def _evaluate(self, x, out, *args, **kwargs):
        b4c = float(x[0])
        w = float(x[1])
        zno = self.fixed_zno_thickness_um
        coating_thickness = b4c + w + zno

        X_model = np.array([[self.al_thickness_um, b4c, w, zno]], dtype=float)
        y_pred = predict_improvement(self.model, self.checkpoint, X_model)[0]
        selected_improvement = float(y_pred[self.energy_index])
        density = calc_areal_density(self.al_thickness_um, b4c, w, zno)

        out["F"] = [
            density["total_areal_density_g_cm2"],
            coating_thickness,
            -selected_improvement,
        ]
        out["G"] = [
            coating_thickness - self.max_coating_thickness_um,
            density["coating_mass_increase_ratio"] - self.max_coating_mass_increase_ratio,
            self.min_required_improvement - selected_improvement,
        ]


def build_result_row(
    model,
    checkpoint,
    x_opt,
    f_opt,
    fixed_zno_thickness_um: float,
    energy_label: str,
    energy_index: int,
    energy_column: str,
    output_cols,
    al_thickness_um: float,
    max_coating_thickness_um: float,
    max_coating_mass_increase_ratio: float,
    min_required_improvement: float,
):
    b4c = float(x_opt[0])
    w = float(x_opt[1])
    zno = float(fixed_zno_thickness_um)
    coating_thickness = b4c + w + zno

    X_model = np.array([[al_thickness_um, b4c, w, zno]], dtype=float)
    y_pred = predict_improvement(model, checkpoint, X_model)[0]
    improvements = [float(value) for value in y_pred]
    selected_improvement = float(improvements[energy_index])
    density = calc_areal_density(al_thickness_um, b4c, w, zno)

    pass_coating_thickness = coating_thickness <= max_coating_thickness_um
    pass_coating_mass = (
        density["coating_mass_increase_ratio"] <= max_coating_mass_increase_ratio
    )
    pass_selected_energy = selected_improvement >= min_required_improvement

    row = {
        "energy_target": energy_label,
        "energy_index": energy_index,
        "Al_thickness_um": al_thickness_um,
        "B4C_thickness_um": b4c,
        "W_thickness_um": w,
        "ZnO_thickness_um": zno,
        "coating_thickness_um": coating_thickness,
        **density,
        "selected_energy_column": energy_column,
        "selected_energy_improvement": selected_improvement,
        "pred_min_improvement": min(improvements),
        "pred_mean_improvement": float(np.mean(improvements)),
        "pass_coating_thickness": pass_coating_thickness,
        "pass_coating_mass": pass_coating_mass,
        "pass_selected_energy": pass_selected_energy,
        "is_feasible": pass_coating_thickness and pass_coating_mass and pass_selected_energy,
        "objective_1_total_areal_density": float(f_opt[0]),
        "objective_2_coating_thickness": float(f_opt[1]),
        "objective_3_neg_selected_improvement": float(f_opt[2]),
    }

    for output_col, value in zip(output_cols, improvements):
        row[f"pred_{output_col}"] = value

    return row


def run_nsga2_for_one_group(
    model,
    checkpoint,
    fixed_zno_thickness_um: float,
    energy_label: str,
    energy_index: int,
    energy_column: str,
    output_cols,
    output_dir: str,
    al_thickness_um: float = 300.0,
    max_coating_thickness_um: float = 500.0,
    max_coating_mass_increase_ratio: float = 0.2,
    min_required_improvement: float = 0.5,
    b4c_bounds=(0.0, 450.0),
    w_bounds=(0.0, 60.0),
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
):
    problem = FixedZnOEnergyProblem(
        model=model,
        checkpoint=checkpoint,
        fixed_zno_thickness_um=fixed_zno_thickness_um,
        energy_index=energy_index,
        al_thickness_um=al_thickness_um,
        b4c_bounds=b4c_bounds,
        w_bounds=w_bounds,
        max_coating_thickness_um=max_coating_thickness_um,
        max_coating_mass_increase_ratio=max_coating_mass_increase_ratio,
        min_required_improvement=min_required_improvement,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    if result.X is None or len(result.X) == 0:
        print(f"ZnO={fixed_zno_thickness_um:.0f} um | {energy_label} | no candidates")
        return pd.DataFrame()

    rows = [
        build_result_row(
            model=model,
            checkpoint=checkpoint,
            x_opt=result.X[i],
            f_opt=result.F[i],
            fixed_zno_thickness_um=fixed_zno_thickness_um,
            energy_label=energy_label,
            energy_index=energy_index,
            energy_column=energy_column,
            output_cols=output_cols,
            al_thickness_um=al_thickness_um,
            max_coating_thickness_um=max_coating_thickness_um,
            max_coating_mass_increase_ratio=max_coating_mass_increase_ratio,
            min_required_improvement=min_required_improvement,
        )
        for i in range(len(result.X))
    ]

    df_scored = add_tradeoff_scores(pd.DataFrame(rows))
    df_sorted = df_scored.sort_values(
        by=[
            "is_feasible",
            "tradeoff_score",
            "selected_energy_improvement",
            "total_areal_density_g_cm2",
            "coating_thickness_um",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    group_dir = os.path.join(
        output_dir,
        f"ZnO_{int(fixed_zno_thickness_um)}um",
        f"energy_{energy_label}",
    )
    os.makedirs(group_dir, exist_ok=True)

    pareto_path = os.path.join(group_dir, "nsga2_pareto_candidates.csv")
    feasible_path = os.path.join(group_dir, "nsga2_feasible_candidates.csv")

    df_sorted.to_csv(pareto_path, index=False, encoding="utf-8-sig")

    feasible_df = df_sorted[df_sorted["is_feasible"]].copy()
    feasible_df.to_csv(feasible_path, index=False, encoding="utf-8-sig")

    print(
        f"ZnO={fixed_zno_thickness_um:.0f} um | "
        f"{energy_label} | "
        f"Pareto={len(df_sorted)} | "
        f"Feasible={len(feasible_df)}"
    )

    return df_sorted


def run_grouped_nsga2(
    model_path: str,
    output_dir: str,
    zno_values,
    energy_points,
    al_thickness_um: float = 300.0,
    max_coating_thickness_um: float = 500.0,
    max_coating_mass_increase_ratio: float = 0.2,
    min_required_improvement: float = 0.5,
    b4c_bounds=(0.0, 450.0),
    w_bounds=(0.0, 60.0),
    training_domain_bounds=None,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    model, checkpoint = load_residual_mlp_model(model_path)
    output_cols = checkpoint.get("output_cols") or [
        f"output_{i}" for i in range(checkpoint["output_dim"])
    ]
    energy_targets = parse_energy_targets(energy_points, output_cols)
    all_results = []

    for zno in zno_values:
        zno_key = format_zno_key(float(zno))
        zno_bounds = None
        if training_domain_bounds:
            zno_bounds = training_domain_bounds.get(zno_key)
            if zno_bounds is None:
                zno_bounds = training_domain_bounds.get(DEFAULT_DOMAIN_KEY)
        group_b4c_bounds = zno_bounds["b4c_bounds"] if zno_bounds else b4c_bounds
        group_w_bounds = zno_bounds["w_bounds"] if zno_bounds else w_bounds

        for energy_label, energy_index, energy_column in energy_targets:
            df_group = run_nsga2_for_one_group(
                model=model,
                checkpoint=checkpoint,
                fixed_zno_thickness_um=float(zno),
                energy_label=energy_label,
                energy_index=energy_index,
                energy_column=energy_column,
                output_cols=output_cols,
                output_dir=output_dir,
                al_thickness_um=al_thickness_um,
                max_coating_thickness_um=max_coating_thickness_um,
                max_coating_mass_increase_ratio=max_coating_mass_increase_ratio,
                min_required_improvement=min_required_improvement,
                b4c_bounds=group_b4c_bounds,
                w_bounds=group_w_bounds,
                pop_size=pop_size,
                n_gen=n_gen,
                seed=seed,
            )

            if len(df_group) > 0:
                all_results.append(df_group)

    if len(all_results) == 0:
        print("No candidates were found for any ZnO/energy group.")
        return None, None

    merged_df = pd.concat(all_results, axis=0, ignore_index=True)
    merged_df = merged_df.sort_values(
        by=[
            "is_feasible",
            "energy_index",
            "tradeoff_score",
            "selected_energy_improvement",
            "total_areal_density_g_cm2",
            "coating_thickness_um",
        ],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)

    merged_path = os.path.join(output_dir, "all_zno_energy_pareto_candidates.csv")
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")

    feasible_df = merged_df[merged_df["is_feasible"]].copy()
    feasible_path = os.path.join(output_dir, "all_zno_energy_feasible_candidates.csv")
    feasible_df.to_csv(feasible_path, index=False, encoding="utf-8-sig")

    recommendation_df = pd.DataFrame()
    if len(feasible_df) > 0:
        recommendation_df = select_representative_candidates(feasible_df)

    recommendation_full_path = os.path.join(output_dir, "recommended_candidate_by_group_full.csv")
    recommendation_path = os.path.join(output_dir, "recommended_candidate_by_group.csv")
    recommendation_df = clean_csv_columns(recommendation_df)
    recommendation_df.to_csv(recommendation_full_path, index=False, encoding="utf-8-sig")
    readable_recommendation_df = build_readable_recommendation_df(recommendation_df)
    readable_recommendation_df.to_csv(recommendation_path, index=False, encoding="utf-8-sig")

    print("\n====== Grouped NSGA-II optimization complete ======")
    print("ZnO groups:", zno_values)
    print("Energy groups:", [item[0] for item in energy_targets])
    print("Merged Pareto candidates:", len(merged_df))
    print("Merged feasible candidates:", len(feasible_df))
    print("Merged Pareto file:", merged_path)
    print("Merged feasible file:", feasible_path)
    print("Recommended-by-group file:", recommendation_path)
    print("Recommended-by-group full file:", recommendation_full_path)

    if len(readable_recommendation_df) > 0:
        print("\n====== Recommended candidate for each ZnO/energy group ======")
        print(readable_recommendation_df)

    return merged_df, feasible_df


def parse_float_values(values_str: str):
    values = []
    for item in values_str.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_bounds(value: str):
    values = parse_float_values(value)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {value}")
    if values[0] > values[1]:
        raise argparse.ArgumentTypeError(f"Lower bound must be <= upper bound, got {value}")
    return tuple(values)


def parse_bool(value: str):
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value}")


def build_energy_target_map(output_cols):
    target_map = {}
    for index, output_col in enumerate(output_cols):
        label = output_col.replace("improvement_", "")
        key = label.lower()
        short_key = key.replace("mev", "")
        target = (label, index, f"pred_{output_col}")
        target_map[key] = target
        target_map[short_key] = target
    return target_map


def parse_energy_targets(values_str: str, output_cols):
    target_map = build_energy_target_map(output_cols)
    targets = []
    seen = set()

    for item in values_str.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in target_map:
            valid = ", ".join(sorted(target_map.keys()))
            raise ValueError(f"Unknown energy point: {item}. Valid values: {valid}")

        target = target_map[key]
        if target[0] not in seen:
            targets.append(target)
            seen.add(target[0])

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/exp_baseline_p/models/ResidualMLP_improvement_surrogate.pth",
        help="Path to the trained ResidualMLP model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/exp_nsga2_by_zno_energy_p",
        help="Output directory for grouped NSGA-II results.",
    )
    parser.add_argument(
        "--zno_values",
        type=str,
        default="50,100,150,200",
        help="Fixed ZnO groups, for example: 50,100,150,200.",
    )
    parser.add_argument(
        "--energy_points",
        type=str,
        default="10,20,50",
        help="Energy point groups to optimize independently, for example: 10,20,50.",
    )
    parser.add_argument("--al_thickness_um", type=float, default=300.0)
    parser.add_argument("--max_coating_thickness_um", type=float, default=500.0)
    parser.add_argument(
        "--max_coating_mass_increase_ratio",
        type=float,
        default=0.2,
        help="Maximum B4C+W areal-density increase relative to the Al+ZnO base.",
    )
    parser.add_argument("--min_required_improvement", type=float, default=0.5)
    parser.add_argument(
        "--b4c_bounds",
        type=parse_bounds,
        default=None,
        help="Manual B4C bounds as min,max. If set, it overrides training-domain B4C bounds.",
    )
    parser.add_argument(
        "--w_bounds",
        type=parse_bounds,
        default=None,
        help="Manual W bounds as min,max. If set, it overrides training-domain W bounds.",
    )
    parser.add_argument(
        "--training_domain_path",
        type=str,
        default="experiments/exp_baseline_p/used_dataset.csv",
        help="Training dataset used to adapt B4C/W search ranges.",
    )
    parser.add_argument(
        "--group_domain_by_zno",
        type=parse_bool,
        default=True,
        help="Whether to adapt B4C/W bounds separately for each ZnO group.",
    )
    parser.add_argument("--domain_margin_um", type=float, default=0.0)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    default_b4c_bounds = args.b4c_bounds if args.b4c_bounds is not None else (0.0, 450.0)
    default_w_bounds = args.w_bounds if args.w_bounds is not None else (0.0, 60.0)

    training_domain_bounds = None
    if args.b4c_bounds is None or args.w_bounds is None:
        training_domain_bounds = load_training_domain_bounds(
            args.training_domain_path,
            margin_um=args.domain_margin_um,
            group_by_zno=args.group_domain_by_zno,
        )

        if training_domain_bounds:
            for bounds in training_domain_bounds.values():
                if args.b4c_bounds is not None:
                    bounds["b4c_bounds"] = args.b4c_bounds
                if args.w_bounds is not None:
                    bounds["w_bounds"] = args.w_bounds
    else:
        print("\n====== Manual bounds ======")
        print(f"B4C={default_b4c_bounds} | W={default_w_bounds}")

    run_grouped_nsga2(
        model_path=args.model_path,
        output_dir=args.output_dir,
        zno_values=parse_float_values(args.zno_values),
        energy_points=args.energy_points,
        al_thickness_um=args.al_thickness_um,
        max_coating_thickness_um=args.max_coating_thickness_um,
        max_coating_mass_increase_ratio=args.max_coating_mass_increase_ratio,
        min_required_improvement=args.min_required_improvement,
        b4c_bounds=default_b4c_bounds,
        w_bounds=default_w_bounds,
        training_domain_bounds=training_domain_bounds,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
