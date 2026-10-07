import logging
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from src.config.settings import settings
from src.database.connection import SessionLocal
from src.features.build_features import FeatureBuilder
from src.models.dataset import ChurnMultimodalDataset
from src.models.pytorch_multimodal import ChurnMultiModalNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Trainer")


def train_multimodal_model(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Churn_Prediction_Multimodal")

    # 1. Extract data
    db = SessionLocal()
    try:
        builder = FeatureBuilder(db)
        df_tab, np_seq, texts, labels = builder.extract_multimodal_dataset()
    finally:
        db.close()

    # 2. Dataset & DataLoaders
    full_dataset = ChurnMultimodalDataset(df_tab, np_seq, texts, labels)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on execution device: {device}")

    # 3. Model Initialization
    model = ChurnMultiModalNet(
        num_numeric_features=df_tab.shape[1],
        seq_input_size=np_seq.shape[2],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 4. MLflow Experiment Run
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": str(device),
            "num_samples": len(full_dataset),
        })

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(
                    batch["x_tab"].to(device),
                    batch["x_seq"].to(device),
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = criterion(logits, batch["label"].to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            y_true, y_probs = [], []

            with torch.no_grad():
                for batch in val_loader:
                    logits = model(
                        batch["x_tab"].to(device),
                        batch["x_seq"].to(device),
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                    )
                    loss = criterion(logits, batch["label"].to(device))
                    val_loss += loss.item()

                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_probs.extend(probs)
                    y_true.extend(batch["label"].numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_roc_auc = roc_auc_score(y_true, y_probs)

            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            val_pr_auc = auc(recall, precision)

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_roc_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
            }, step=epoch)

        # Log Model Artifact
        mlflow.pytorch.log_model(model, "model")
        logger.info("Training pipeline completed and artifact registered to MLflow.")


if __name__ == "__main__":
    train_multimodal_model(epochs=3, batch_size=16)