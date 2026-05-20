import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam

from models.trust_score import compute_trust_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    BATCH_SIZE,
    CENTRALIZED_TRAIN_PATH,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBED_DIM,
    FEATURE_COLS,
    FIGURES_DIR,
    GMM_COMPONENTS,
    GMM_COVARIANCE_TYPE,
    GMM_REG_COVAR,
    HIDDEN_DIM,
    LOGS_DIR,
    LR,
    MODELS_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
    TABLES_DIR,
    TARGET_COL,
    TEST_PATH,
    VAL_SIZE,
    WEIGHT_DECAY,
)
from models.classifier import MedicalMLP
from models.gmm_reliability import (
    compute_class_priors,
    compute_gmm_reliability,
    fit_classwise_gmms,
)
from utils.data_utils import (
    create_dataloader,
    get_pos_weight,
    prepare_centralized_splits,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_embeddings = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits, embeddings = model(features, return_embedding=True)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        all_logits.append(logits.cpu().numpy().reshape(-1))
        all_probs.append(probs.cpu().numpy().reshape(-1))
        all_preds.append(preds.cpu().numpy().reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))
        all_embeddings.append(embeddings.cpu().numpy())

    return {
        "logits": np.concatenate(all_logits).reshape(-1),
        "probs": np.concatenate(all_probs).reshape(-1),
        "preds": np.concatenate(all_preds).astype(int).reshape(-1),
        "labels": np.concatenate(all_labels).astype(int).reshape(-1),
        "embeddings": np.concatenate(all_embeddings),
    }


def compute_classification_metrics(y_true, y_prob, y_pred):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    return metrics


def compute_predicted_class_confidence(y_prob):
    return np.maximum(y_prob, 1.0 - y_prob)


def compute_error_detection_metrics(y_true, y_pred, reliability_scores):
    error_labels = (y_true != y_pred).astype(int)
    error_scores = 1.0 - reliability_scores

    if len(np.unique(error_labels)) < 2:
        return {"error_auroc": np.nan, "error_auprc": np.nan}

    return {
        "error_auroc": float(roc_auc_score(error_labels, error_scores)),
        "error_auprc": float(average_precision_score(error_labels, error_scores)),
    }


def risk_at_k(y_true, y_pred, reliability_scores, k_percent=5):
    error_labels = (y_true != y_pred).astype(int)
    n = len(reliability_scores)
    k = max(1, int(np.ceil(n * k_percent / 100.0)))

    lowest_idx = np.argsort(reliability_scores)[:k]
    return float(error_labels[lowest_idx].mean())


def plot_risk_coverage(y_true, y_pred, softmax_scores, gmm_scores, trust_scores, save_path):
    def compute_curve(reliability):
        error = (y_true != y_pred).astype(int)
        sorted_idx = np.argsort(-reliability)
        error_sorted = error[sorted_idx]

        n = len(error)
        coverages = []
        risks = []

        for k in range(1, n + 1):
            coverages.append(k / n)
            risks.append(error_sorted[:k].mean())

        return coverages, risks

    cov_s, risk_s = compute_curve(softmax_scores)
    cov_g, risk_g = compute_curve(gmm_scores)
    cov_t, risk_t = compute_curve(trust_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(cov_s, risk_s, label="Softmax Confidence")
    plt.plot(cov_g, risk_g, label="GMM Reliability")
    plt.plot(cov_t, risk_t, label="TrustScore")

    plt.xlabel("Coverage")
    plt.ylabel("Risk (Error Rate)")
    plt.title("Risk鈥揅overage Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_concentration(y_true, y_pred, softmax_scores, gmm_scores, trust_scores, save_path):
    error = (y_true != y_pred).astype(int)

    def compute_curve(reliability):
        sorted_idx = np.argsort(reliability)  # Start from the least reliable samples / 从最不可靠的样本开始
        error_sorted = error[sorted_idx]

        n = len(error_sorted)
        coverage = np.arange(1, n + 1) / n
        risk = np.cumsum(error_sorted) / np.arange(1, n + 1)

        return coverage, risk

    cov_s, risk_s = compute_curve(softmax_scores)
    cov_g, risk_g = compute_curve(gmm_scores)
    cov_t, risk_t = compute_curve(trust_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(cov_s, risk_s, label="Softmax (CL)", linewidth=2)
    plt.plot(cov_g, risk_g, label="GMM (CL)", linewidth=2)
    plt.plot(cov_t, risk_t, label="TrustScore (CL)", linewidth=2)

    plt.xlabel("Fraction of Most Unreliable Samples")
    plt.ylabel("Error Rate (Risk)")
    plt.title("Error Concentration Curve")

    plt.legend()
    plt.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_distribution_plot(confidence_scores, gmm_scores, trust_scores, y_true, y_pred):
    error_labels = (y_true != y_pred).astype(int)

    plt.figure(figsize=(8, 5))
    plt.hist(
        gmm_scores[error_labels == 0],
        bins=40,
        alpha=0.6,
        label="GMM reliability - correct",
        density=True,
    )
    plt.hist(
        gmm_scores[error_labels == 1],
        bins=40,
        alpha=0.6,
        label="GMM reliability - incorrect",
        density=True,
    )
    plt.xlabel("Reliability score")
    plt.ylabel("Density")
    plt.title("GMM reliability distribution on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gmm_reliability_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(
        confidence_scores[error_labels == 0],
        bins=40,
        alpha=0.6,
        label="Confidence - correct",
        density=True,
    )
    plt.hist(
        confidence_scores[error_labels == 1],
        bins=40,
        alpha=0.6,
        label="Confidence - incorrect",
        density=True,
    )
    plt.xlabel("Confidence score")
    plt.ylabel("Density")
    plt.title("Softmax confidence distribution on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(
        trust_scores[error_labels == 0],
        bins=40,
        alpha=0.6,
        label="TrustScore - correct",
        density=True,
    )
    plt.hist(
        trust_scores[error_labels == 1],
        bins=40,
        alpha=0.6,
        label="TrustScore - incorrect",
        density=True,
    )
    plt.xlabel("TrustScore")
    plt.ylabel("Density")
    plt.title("TrustScore distribution on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trustscore_distribution.png", dpi=300)
    plt.close()


def main():
    set_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")
    print(f"Using features: {FEATURE_COLS}")
    print(f"Target column: {TARGET_COL}")

    split_data = prepare_centralized_splits(
        train_csv_path=str(CENTRALIZED_TRAIN_PATH),
        test_csv_path=str(TEST_PATH),
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        val_size=VAL_SIZE,
        random_state=RANDOM_SEED,
    )

    train_loader = create_dataloader(split_data.X_train, split_data.y_train, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(split_data.X_val, split_data.y_val, BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(split_data.X_test, split_data.y_test, BATCH_SIZE, shuffle=False)

    model = MedicalMLP(
        input_dim=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    pos_weight = get_pos_weight(split_data.y_train).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auroc = -np.inf
    best_state_dict = None
    patience_counter = 0
    training_log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        val_outputs = predict_model(model, val_loader, DEVICE)
        val_metrics = compute_classification_metrics(
            val_outputs["labels"],
            val_outputs["probs"],
            val_outputs["preds"],
        )

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
        }
        training_log.append(log_row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f}"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state_dict is None:
        raise RuntimeError("No best model was saved during training.")

    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), MODELS_DIR / "centralized_mlp_best.pt")

    train_eval_loader = create_dataloader(split_data.X_train, split_data.y_train, BATCH_SIZE, shuffle=False)
    train_outputs = predict_model(model, train_eval_loader, DEVICE)
    test_outputs = predict_model(model, test_loader, DEVICE)

    test_metrics = compute_classification_metrics(
        test_outputs["labels"],
        test_outputs["probs"],
        test_outputs["preds"],
    )

    gmms = fit_classwise_gmms(
        embeddings=train_outputs["embeddings"],
        labels=train_outputs["labels"],
        n_components=GMM_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        reg_covar=GMM_REG_COVAR,
        random_state=RANDOM_SEED,
    )

    class_priors = compute_class_priors(train_outputs["labels"])

    gmm_reliability = compute_gmm_reliability(
        embeddings=test_outputs["embeddings"],
        predicted_labels=test_outputs["preds"],
        gmms=gmms,
        class_priors=class_priors,
    )

    trust_scores = compute_trust_score(
        train_embeddings=train_outputs["embeddings"],
        train_labels=train_outputs["labels"],
        test_embeddings=test_outputs["embeddings"],
        predicted_labels=test_outputs["preds"],
        k=1,
        metric="euclidean",
        bounded=True,
    )

    trust_error_metrics = compute_error_detection_metrics(
        y_true=test_outputs["labels"],
        y_pred=test_outputs["preds"],
        reliability_scores=trust_scores,
    )

    confidence_scores = compute_predicted_class_confidence(test_outputs["probs"])

    confidence_error_metrics = compute_error_detection_metrics(
        y_true=test_outputs["labels"],
        y_pred=test_outputs["preds"],
        reliability_scores=confidence_scores,
    )
    gmm_error_metrics = compute_error_detection_metrics(
        y_true=test_outputs["labels"],
        y_pred=test_outputs["preds"],
        reliability_scores=gmm_reliability,
    )

    risk_table = pd.DataFrame(
        [
            {
                "method": "softmax_confidence",
                "risk_at_1_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], confidence_scores, 1),
                "risk_at_5_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], confidence_scores, 5),
                "risk_at_10_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], confidence_scores, 10),
            },
            {
                "method": "trust_score",
                "risk_at_1_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], trust_scores, 1),
                "risk_at_5_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], trust_scores, 5),
                "risk_at_10_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], trust_scores, 10),
            },
            {
                "method": "gmm_reliability",
                "risk_at_1_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], gmm_reliability, 1),
                "risk_at_5_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], gmm_reliability, 5),
                "risk_at_10_percent": risk_at_k(test_outputs["labels"], test_outputs["preds"], gmm_reliability, 10),
            },
        ]
    )

    classification_df = pd.DataFrame([test_metrics])
    error_detection_df = pd.DataFrame(
        [
            {
                "method": "softmax_confidence",
                **confidence_error_metrics,
            },
            {
                "method": "trust_score",
                **trust_error_metrics,
            },
            {
                "method": "gmm_reliability",
                **gmm_error_metrics,
            },
        ]
    )

    plot_error_concentration(
        y_true=test_outputs["labels"],
        y_pred=test_outputs["preds"],
        softmax_scores=confidence_scores,
        gmm_scores=gmm_reliability,
        trust_scores=trust_scores,
        save_path=FIGURES_DIR / "error_concentration_cl.png"
    )

    predictions_df = pd.DataFrame(
        {
            "y_true": test_outputs["labels"],
            "y_prob": test_outputs["probs"],
            "y_pred": test_outputs["preds"],
            "softmax_confidence": confidence_scores,
            "trust_score": trust_scores,
            "gmm_reliability": gmm_reliability,
            "is_error": (test_outputs["labels"] != test_outputs["preds"]).astype(int),
        }
    )

    pd.DataFrame(training_log).to_csv(LOGS_DIR / "centralized_training_log.csv", index=False)
    classification_df.to_csv(TABLES_DIR / "centralized_classification_metrics.csv", index=False)
    error_detection_df.to_csv(TABLES_DIR / "centralized_error_detection_metrics.csv", index=False)
    risk_table.to_csv(TABLES_DIR / "centralized_risk_at_k.csv", index=False)
    predictions_df.to_csv(TABLES_DIR / "centralized_test_predictions_with_reliability.csv", index=False)

    with open(LOGS_DIR / "centralized_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classification_metrics": test_metrics,
                "confidence_error_metrics": confidence_error_metrics,
                "trust_error_metrics": trust_error_metrics,
                "gmm_error_metrics": gmm_error_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_distribution_plot(
        confidence_scores=confidence_scores,
        gmm_scores=gmm_reliability,
        trust_scores=trust_scores,
        y_true=test_outputs["labels"],
        y_pred=test_outputs["preds"],
    )

    plot_risk_coverage(
        y_true=test_outputs["labels"],
        y_pred=test_outputs["preds"],
        softmax_scores=confidence_scores,
        gmm_scores=gmm_reliability,
        trust_scores=trust_scores,
        save_path=FIGURES_DIR / "risk_coverage_curve.png"
    )

    print("\n===== Final Test Classification Metrics =====")
    print(classification_df.to_string(index=False))

    print("\n===== Error Detection Metrics =====")
    print(error_detection_df.to_string(index=False))

    print("\n===== Risk@K =====")
    print(risk_table.to_string(index=False))

    print("\nSaved files:")
    print(f"- {MODELS_DIR / 'centralized_mlp_best.pt'}")
    print(f"- {TABLES_DIR / 'centralized_classification_metrics.csv'}")
    print(f"- {TABLES_DIR / 'centralized_error_detection_metrics.csv'}")
    print(f"- {TABLES_DIR / 'centralized_risk_at_k.csv'}")
    print(f"- {TABLES_DIR / 'centralized_test_predictions_with_reliability.csv'}")
    print(f"- {FIGURES_DIR / 'gmm_reliability_distribution.png'}")
    print(f"- {FIGURES_DIR / 'confidence_distribution.png'}")
    print(f"- {FIGURES_DIR / 'trustscore_distribution.png'}")
    print(f"- {FIGURES_DIR / 'error_concentration_cl.png'}")
    print(f"- {FIGURES_DIR / 'risk_coverage_curve.png'}")


if __name__ == "__main__":
    main()
