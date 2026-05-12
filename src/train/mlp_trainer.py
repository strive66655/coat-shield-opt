import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.residual_model import ResidualMLP


def train_residual_mlp(cfg, X_train, y_train, X_test, y_test):
    p = cfg["models"]["ResidualMLP"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ResidualMLP 使用设备:", device)

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
        batch_size=p["batch_size"],
        shuffle=True,
    )

    model = ResidualMLP(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=p["hidden_dim"],
        num_blocks=p["num_blocks"],
        dropout=p["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=p["lr"],
        weight_decay=p["weight_decay"],
    )

    loss_fn = nn.MSELoss()

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_state = None
    patience_count = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, p["epochs"] + 1):
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
            val_pred = model(X_test_t)
            val_loss = loss_fn(val_pred, y_test_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if patience_count >= p["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_s = model(X_test_t).cpu().numpy()

    y_pred = y_scaler.inverse_transform(pred_s)

    save_obj = {
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "output_dim": y_train.shape[1],
        "params": p,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    return model, y_pred, save_obj
