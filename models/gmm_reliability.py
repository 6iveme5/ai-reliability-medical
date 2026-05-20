# models/gmm_reliability.py

from typing import Dict, Optional
import numpy as np
from sklearn.mixture import GaussianMixture


def _safe_logsumexp(values: np.ndarray) -> float:
    max_v = np.max(values)
    return max_v + np.log(np.sum(np.exp(values - max_v)))


class GMMReliability:
    """
    Class-wise Gaussian Mixture Model reliability estimator.
    按类别拟合的高斯混合模型可靠性估计器。

    Reliability = posterior probability of predicted class:
    可靠性分数 = 预测类别的后验概率：
        p(class = y_hat | z)
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        reg_covar: float = 1e-5,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.gmms: Dict[int, GaussianMixture] = {}
        self.class_priors: Optional[Dict[int, float]] = None
        self.classes = None

    # Fit class-wise density models / 按类别拟合密度模型
    def fit(self, embeddings: np.ndarray, labels: np.ndarray):

        self.gmms = {}
        self.classes = np.unique(labels)

        # Estimate class priors / 估计类别先验概率
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        self.class_priors = {
            int(k): float(v / total) for k, v in zip(unique, counts)
        }

        # Fit one GMM per class / 每个类别单独拟合一个 GMM
        for cls in self.classes:
            class_embeddings = embeddings[labels == cls]

            effective_components = min(self.n_components, len(class_embeddings))
            effective_components = max(effective_components, 1)

            gmm = GaussianMixture(
                n_components=effective_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            )

            gmm.fit(class_embeddings)
            self.gmms[int(cls)] = gmm

    # Compute reliability scores / 计算可靠性分数
    def compute(
        self,
        embeddings: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> np.ndarray:

        if not self.gmms:
            raise RuntimeError("GMMReliability must be fitted before calling compute().")

        classes = sorted(self.gmms.keys())
        n_samples = embeddings.shape[0]

        log_joint = np.zeros((n_samples, len(classes)), dtype=np.float64)

        for idx, cls in enumerate(classes):
            log_likelihood = self.gmms[cls].score_samples(embeddings)
            prior = max(self.class_priors.get(cls, 1e-12), 1e-12)
            log_joint[:, idx] = log_likelihood + np.log(prior)

        reliability = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            denom = _safe_logsumexp(log_joint[i])
            pred_cls = int(predicted_labels[i])
            pred_idx = classes.index(pred_cls)
            reliability[i] = np.exp(log_joint[i, pred_idx] - denom)

        return reliability.astype(np.float32)


# Compatibility wrapper functions for old pipeline / 兼容旧流程的封装函数
def compute_class_priors(labels):
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {int(u): c / total for u, c in zip(unique, counts)}


def fit_classwise_gmms(
    embeddings,
    labels,
    n_components=3,
    covariance_type="full",
    reg_covar=1e-5,
    random_state=42,
):
    model = GMMReliability(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
    )
    model.fit(embeddings, labels)
    return model


def compute_gmm_reliability(
    embeddings,
    predicted_labels,
    gmms,
    class_priors=None,
):
    """
    gmms is actually a fitted GMMReliability model.
    这里的 gmms 实际上是已经拟合好的 GMMReliability 实例。
    """
    return gmms.compute(embeddings, predicted_labels)
