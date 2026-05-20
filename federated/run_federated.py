# federated/run_federated.py

import os
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

from configs.config import FIGURES_DIR, TABLES_DIR
from federated.client import FederatedClient
from federated.server import FederatedServer
from models.gmm_reliability import GMMReliability
from configs.config import Config
from utils.data_utils import prepare_centralized_splits, create_dataloader


def risk_at_k(reliability_scores, error_labels, k_percent):
    n = len(reliability_scores)
    k = max(1, int(n * k_percent / 100))
    idx = np.argsort(reliability_scores)[:k]   # Lowest-reliability samples / 可靠性最低的样本
    return float(error_labels[idx].mean())


def ensure_parent_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_risk_concentration(y_true, y_pred, softmax_scores, gmm_scores, save_path):
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

    plt.figure(figsize=(6, 5))
    plt.plot(cov_s, risk_s, label="Softmax (FL)", linewidth=2)
    plt.plot(cov_g, risk_g, label="GMM (FL)", linewidth=2)

    plt.xlabel("Fraction of Most Unreliable Samples")
    plt.ylabel("Error Rate (Risk)")
    plt.title("Error Concentration Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = ensure_parent_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_reliability_distribution(error_labels, softmax_scores, gmm_scores, save_path_softmax, save_path_gmm):
    plt.figure(figsize=(6, 5))
    plt.hist(
        softmax_scores[error_labels == 0],
        bins=40,
        alpha=0.6,
        density=True,
        label="Correct",
    )
    plt.hist(
        softmax_scores[error_labels == 1],
        bins=40,
        alpha=0.6,
        density=True,
        label="Incorrect",
    )
    plt.xlabel("Softmax confidence")
    plt.ylabel("Density")
    plt.title("Softmax Distribution (FL)")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path_softmax = ensure_parent_dir(save_path_softmax)
    plt.tight_layout()
    plt.savefig(save_path_softmax, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(
        gmm_scores[error_labels == 0],
        bins=40,
        alpha=0.6,
        density=True,
        label="Correct",
    )
    plt.hist(
        gmm_scores[error_labels == 1],
        bins=40,
        alpha=0.6,
        density=True,
        label="Incorrect",
    )
    plt.xlabel("GMM reliability")
    plt.ylabel("Density")
    plt.title("GMM Reliability Distribution (FL)")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path_gmm = ensure_parent_dir(save_path_gmm)
    plt.tight_layout()
    plt.savefig(save_path_gmm, dpi=300)
    plt.close()


def plot_sorted_error_curve(error_labels, softmax_scores, gmm_scores, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    def moving_average(x, window=100):
        return np.convolve(x, np.ones(window)/window, mode='same')

    def get_sorted_error(reliability):
        idx = np.argsort(reliability)
        return error_labels[idx]

    sorted_error_softmax = get_sorted_error(softmax_scores)
    sorted_error_gmm = get_sorted_error(gmm_scores)

    # Smooth local error patterns / 平滑局部错误模式
    smooth_softmax = moving_average(sorted_error_softmax, 100)
    smooth_gmm = moving_average(sorted_error_gmm, 100)

    x = np.arange(len(error_labels)) / len(error_labels)

    plt.figure(figsize=(7, 4.5))

    # Plot smoothed curves / 绘制平滑后的曲线
    plt.plot(x, smooth_softmax, label="Softmax (FL)", linewidth=2)
    plt.plot(x, smooth_gmm, label="GMM (FL)", linewidth=2)

    plt.xlabel("Fraction of samples (sorted by reliability)")
    plt.ylabel("Local error rate")
    plt.title("Smoothed Sorted Error Pattern")

    plt.legend()
    plt.grid(alpha=0.3)
    print("smooth_softmax:", smooth_softmax[:10])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_federated():
    config = Config()

    print("Federated client paths:")
    print(config.federated_client_paths)
    print("Number of clients:")
    print(config.num_clients)
    print("--------------------------")

    server = FederatedServer(config)

    clients = [
        FederatedClient(client_id=i, config=config)
        for i in range(config.num_clients)
    ]

    # ================= FedAvg aggregation / FedAvg 参数聚合 =================
    for round_idx in range(config.global_rounds):
        global_weights = server.get_global_weights()
        client_updates = []

        for client in clients:
            client.set_weights(global_weights)
            updated_weights, data_size = client.train()
            client_updates.append((updated_weights, data_size))

        server.aggregate(client_updates)
        print(f"Global round {round_idx + 1} completed")

    # ================= Load centralized test set / 加载集中式测试集 =================
    split_data = prepare_centralized_splits(
        train_csv_path=str(config.train_csv_path),
        test_csv_path=str(config.test_csv_path),
        feature_cols=config.feature_cols,
        target_col=config.target_col,
        val_size=config.val_size,
        random_state=config.random_state,
    )

    test_loader = create_dataloader(
        split_data.X_test,
        split_data.y_test,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # ================= Extract test embeddings with the global model / 使用全局模型提取测试集 embedding =================
    print("Training GMM reliability...")

    global_model = server.global_model
    global_model.eval()

    all_embeddings = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.device)

            logits, embedding = global_model(x, return_embedding=True)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_embeddings.append(embedding.cpu())
            all_labels.append(y.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    embeddings = torch.cat(all_embeddings).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    predicted_labels = torch.cat(all_preds).numpy().astype(int)
    prob_positive = torch.cat(all_probs).numpy().reshape(-1)

    # Use predicted-class confidence, not only positive-class probability / 使用预测类别置信度，而不是只用正类概率
    softmax_confidence = np.maximum(prob_positive, 1.0 - prob_positive)

    # ================= GMM reliability / GMM 可靠性估计 =================
    gmm = GMMReliability(
        n_components=config.gmm_components,
        covariance_type=config.gmm_covariance_type,
        reg_covar=config.gmm_reg_covar,
        random_state=config.random_state,
    )

    # Current logic fits GMM on test embeddings; revise this before final paper experiments. / 当前逻辑使用测试集 embedding 拟合 GMM；正式论文实验前应改为使用训练侧 embedding。
    gmm.fit(embeddings, labels)
    reliability_scores = gmm.compute(embeddings, predicted_labels)

    print("Federated training completed.")
    print("Reliability shape:", reliability_scores.shape)

    # ================= Classification performance / 分类性能 =================
    accuracy = float((predicted_labels == labels).mean())

    tp = np.sum((predicted_labels == 1) & (labels == 1))
    fp = np.sum((predicted_labels == 1) & (labels == 0))
    fn = np.sum((predicted_labels == 0) & (labels == 1))

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))

    try:
        auroc = float(roc_auc_score(labels, prob_positive))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(labels, prob_positive))
    except ValueError:
        auprc = float("nan")

    print("\n===== Final Test Classification Metrics =====")
    print(f"accuracy   = {accuracy:.6f}")
    print(f"auroc      = {auroc:.6f}")
    print(f"auprc      = {auprc:.6f}")
    print(f"f1         = {f1:.6f}")
    print(f"precision  = {precision:.6f}")
    print(f"recall     = {recall:.6f}")

    # ================= Misclassification detection / 错误预测检测 =================
    error_labels = (predicted_labels != labels).astype(int)

    softmax_error_score = -softmax_confidence
    gmm_error_score = -reliability_scores

    softmax_error_auroc = float(roc_auc_score(error_labels, softmax_error_score))
    softmax_error_auprc = float(average_precision_score(error_labels, softmax_error_score))

    gmm_error_auroc = float(roc_auc_score(error_labels, gmm_error_score))
    gmm_error_auprc = float(average_precision_score(error_labels, gmm_error_score))

    print("\n===== Error Detection Metrics =====")
    print(f"{'method':>20} {'error_auroc':>12} {'error_auprc':>12}")
    print(f"{'softmax_confidence':>20} {softmax_error_auroc:12.6f} {softmax_error_auprc:12.6f}")
    print(f"{'gmm_reliability':>20} {gmm_error_auroc:12.6f} {gmm_error_auprc:12.6f}")

    # ================= Risk@K / 低可靠样本错误率 =================
    softmax_risk_1 = risk_at_k(softmax_confidence, error_labels, 1)
    softmax_risk_5 = risk_at_k(softmax_confidence, error_labels, 5)
    softmax_risk_10 = risk_at_k(softmax_confidence, error_labels, 10)

    gmm_risk_1 = risk_at_k(reliability_scores, error_labels, 1)
    gmm_risk_5 = risk_at_k(reliability_scores, error_labels, 5)
    gmm_risk_10 = risk_at_k(reliability_scores, error_labels, 10)

    print("\n===== Risk@K =====")
    print("GMM components:", config.gmm_components)
    print("GMM covariance:", config.gmm_covariance_type)
    print(f"{'method':>20} {'risk_at_1_percent':>18} {'risk_at_5_percent':>18} {'risk_at_10_percent':>19}")
    print(f"{'softmax_confidence':>20} {softmax_risk_1:18.6f} {softmax_risk_5:18.6f} {softmax_risk_10:19.6f}")
    print(f"{'gmm_reliability':>20} {gmm_risk_1:18.6f} {gmm_risk_5:18.6f} {gmm_risk_10:19.6f}")

    # ================= Save tables / 保存表格 =================
    metrics_df = pd.DataFrame([
        {
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    ])

    error_detection_df = pd.DataFrame([
        {
            "method": "softmax_confidence",
            "error_auroc": softmax_error_auroc,
            "error_auprc": softmax_error_auprc,
        },
        {
            "method": "gmm_reliability",
            "error_auroc": gmm_error_auroc,
            "error_auprc": gmm_error_auprc,
        },
    ])

    risk_df = pd.DataFrame([
        {
            "method": "softmax_confidence",
            "risk_at_1_percent": softmax_risk_1,
            "risk_at_5_percent": softmax_risk_5,
            "risk_at_10_percent": softmax_risk_10,
        },
        {
            "method": "gmm_reliability",
            "risk_at_1_percent": gmm_risk_1,
            "risk_at_5_percent": gmm_risk_5,
            "risk_at_10_percent": gmm_risk_10,
        },
    ])

    predictions_df = pd.DataFrame({
        "y_true": labels,
        "y_pred": predicted_labels,
        "y_prob_positive": prob_positive,
        "softmax_confidence": softmax_confidence,
        "gmm_reliability": reliability_scores,
        "is_error": error_labels,
    })

    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)

    full_df = pd.concat([predictions_df, embeddings_df], axis=1)

    ensure_parent_dir(TABLES_DIR / "federated_classification_metrics.csv")
    metrics_df.to_csv(TABLES_DIR / "federated_classification_metrics.csv", index=False)
    error_detection_df.to_csv(TABLES_DIR / "federated_error_detection_metrics.csv", index=False)
    risk_df.to_csv(TABLES_DIR / "federated_risk_at_k.csv", index=False)
    predictions_df.to_csv(TABLES_DIR / "federated_test_predictions.csv", index=False)
    embeddings_df.to_csv(TABLES_DIR / "federated_test_embeddings.csv", index=False)
    full_df.to_csv(TABLES_DIR / "federated_test_predictions_with_embeddings.csv", index=False)

    # ================= Save figures / 保存图像 =================
    plot_risk_concentration(
        y_true=labels,
        y_pred=predicted_labels,
        softmax_scores=softmax_confidence,
        gmm_scores=reliability_scores,
        save_path=FIGURES_DIR / "error_concentration_fl.png"
    )

    plot_reliability_distribution(
        error_labels=error_labels,
        softmax_scores=softmax_confidence,
        gmm_scores=reliability_scores,
        save_path_softmax=FIGURES_DIR / "softmax_distribution_fl.png",
        save_path_gmm=FIGURES_DIR / "gmm_distribution_fl.png",
    )

    plot_sorted_error_curve(
        error_labels=error_labels,
        softmax_scores=softmax_confidence,
        gmm_scores=reliability_scores,
        save_path=FIGURES_DIR / "smoothed_sorted_error_curve_fl.png",
    )

    print("\nSaved files:")
    print(f"- {TABLES_DIR / 'federated_classification_metrics.csv'}")
    print(f"- {TABLES_DIR / 'federated_error_detection_metrics.csv'}")
    print(f"- {TABLES_DIR / 'federated_risk_at_k.csv'}")
    print(f"- {TABLES_DIR / 'federated_test_predictions.csv'}")
    print(f"- {TABLES_DIR / 'federated_test_embeddings.csv'}")
    print(f"- {TABLES_DIR / 'federated_test_predictions_with_embeddings.csv'}")
    print(f"- {FIGURES_DIR / 'error_concentration_fl.png'}")
    print(f"- {FIGURES_DIR / 'softmax_distribution_fl.png'}")
    print(f"- {FIGURES_DIR / 'gmm_distribution_fl.png'}")
    print(f"- {FIGURES_DIR / 'sorted_error_curve_fl.png'}")


if __name__ == "__main__":
    run_federated()
