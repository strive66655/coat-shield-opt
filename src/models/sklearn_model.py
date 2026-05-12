import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_random_forest(cfg, X_train, y_train):
    p = cfg["models"]["RandomForest"]

    base_model = RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        random_state=cfg["experiment"]["random_state"],
        n_jobs=p["n_jobs"],
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    return model


def train_xgboost(cfg, X_train, y_train):
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        print("\nXGBoost 未安装，跳过。")
        print("安装方式：pip install xgboost")
        print("错误信息:", e)
        return None

    p = cfg["models"]["XGBoost"]

    base_model = XGBRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        objective=p["objective"],
        random_state=cfg["experiment"]["random_state"],
        n_jobs=p["n_jobs"],
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    return model


def save_sklearn_model(cfg, model, model_name, input_cols, output_cols):
    model_dir = cfg["_runtime"]["model_dir"]

    save_obj = {
        "model": model,
        "model_name": model_name,
        "input_cols": input_cols,
        "output_cols": output_cols,
        "target_type": cfg["target"]["type"],
    }

    save_path = os.path.join(model_dir, f"{model_name}_improvement_surrogate.pkl")
    joblib.dump(save_obj, save_path)
