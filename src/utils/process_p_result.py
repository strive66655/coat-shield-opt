from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/仿真结果0511.xlsx")
OUTPUT_DIR = Path("data/processed")


def main():
    df = pd.read_excel(INPUT_PATH, sheet_name="p_result")

    clean = pd.DataFrame()
    clean["source_row"] = df.index + 2
    clean["record_type"] = "design_candidate"
    clean.loc[df.index < 2, "record_type"] = "baseline"

    mapping = {
        "Al厚度(μm)": "Al_thickness_um",
        "B4C厚度(μm)": "B4C_thickness_um",
        "W厚度(μm)": "W_thickness_um",
        "ZnO厚度(μm)": "ZnO_thickness_um",
        "Al面密度(g/cm2)": "Al_areal_density_g_cm2",
        "B4C面密度(g/cm2)": "B4C_areal_density_g_cm2",
        "W面密度(g/cm2)": "W_areal_density_g_cm2",
        "ZnO面密度(g/cm2)": "ZnO_areal_density_g_cm2",
        "总面密度(g/cm2)": "total_areal_density_g_cm2",
        "E_10MeV(Tev) ": "energy_10Mev_Tev",
        "E_20MeV(Tev)": "energy_20Mev_Tev",
        "E_50MeV(Tev)": "energy_50Mev_Tev",
        "b=1-E[]/E0": "attenuation_10Mev",
        "Unnamed: 16": "attenuation_20Mev",
        "Unnamed: 17": "attenuation_50Mev",
        "efficence=b/a-1": "improvement_10Mev",
        "Unnamed: 20": "improvement_20Mev",
        "5000000次事件，质子能量": "improvement_50Mev",
    }

    for source_col, target_col in mapping.items():
        clean[target_col] = df[source_col] if source_col in df.columns else pd.NA

    for col in clean.columns:
        if col != "record_type":
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    improvement_cols = [
        "improvement_10Mev",
        "improvement_20Mev",
        "improvement_50Mev",
    ]
    clean["mean_improvement"] = clean[improvement_cols].mean(axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "p_clean.csv"
    xlsx_path = OUTPUT_DIR / "p_clean.xlsx"
    summary_path = OUTPUT_DIR / "p_summary.txt"

    clean.to_csv(csv_path, index=False, encoding="utf-8-sig")
    clean.to_excel(xlsx_path, index=False)
    summary_path.write_text(build_summary(clean, improvement_cols), encoding="utf-8")

    print(f"wrote {csv_path} shape={clean.shape}")
    print(f"wrote {xlsx_path}")
    print(f"wrote {summary_path}")


def build_summary(clean: pd.DataFrame, improvement_cols: list[str]) -> str:
    design = clean[clean["record_type"] == "design_candidate"].copy()
    lines = [
        "P-result data processing summary",
        f"Total records: {len(clean)}",
        f"Design candidates: {len(design)}",
        f"Baselines: {int((clean['record_type'] == 'baseline').sum())}",
        "",
    ]

    for col in improvement_cols + ["mean_improvement"]:
        ranked = design.sort_values(col, ascending=False).head(5)
        lines.append(f"Top 5 by {col}:")
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            lines.append(
                f"  {rank}. row={int(row.source_row)}, {col}={getattr(row, col)}, "
                f"B4C={row.B4C_thickness_um} um, "
                f"W={row.W_thickness_um} um, "
                f"ZnO={row.ZnO_thickness_um} um"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
