import numpy as np
from sklearn.neighbors import NearestNeighbors


def _to_1d(x):
    return np.asarray(x).reshape(-1)


class TrustScoreComputer:
    """
    Trust Score in embedding space.
    embedding 空间中的 Trust Score 可靠性估计。

    For a test sample z with predicted label y_hat:
    对于预测标签为 y_hat 的测试样本 z：
        trust_score = d(z, nearest sample from nearest non-predicted class)
                      ------------------------------------------------------
                      d(z, nearest sample from predicted class) + eps

    Larger score => more reliable.
    分数越大，表示预测越可靠。

    Optionally map raw score to [0, 1] by:
    可选地将原始分数映射到 [0, 1]：
        s = score / (1 + score)
    """

    def __init__(self, k: int = 1, metric: str = "euclidean", eps: float = 1e-12, bounded: bool = True):
        self.k = k
        self.metric = metric
        self.eps = eps
        self.bounded = bounded

        self.classes_ = None
        self.class_embeddings_ = {}
        self.class_nn_models_ = {}

    def fit(self, embeddings, labels):
        X = np.asarray(embeddings, dtype=np.float32)
        y = _to_1d(labels).astype(int)

        self.classes_ = np.unique(y)
        self.class_embeddings_.clear()
        self.class_nn_models_.clear()

        for cls in self.classes_:
            Xc = X[y == cls]
            if len(Xc) == 0:
                continue

            nn = NearestNeighbors(
                n_neighbors=min(self.k, len(Xc)),
                metric=self.metric,
            )
            nn.fit(Xc)

            self.class_embeddings_[cls] = Xc
            self.class_nn_models_[cls] = nn

        return self

    def _distance_to_class(self, z, cls):
        nn = self.class_nn_models_[cls]
        dist, _ = nn.kneighbors(z.reshape(1, -1), n_neighbors=1, return_distance=True)
        return float(dist[0, 0])

    def score(self, embeddings, predicted_labels):
        X = np.asarray(embeddings, dtype=np.float32)
        y_pred = _to_1d(predicted_labels).astype(int)

        scores = np.zeros(len(X), dtype=np.float32)

        for i, (z, pred_cls) in enumerate(zip(X, y_pred)):
            if pred_cls not in self.class_nn_models_:
                scores[i] = 0.0
                continue

            d_pred = self._distance_to_class(z, pred_cls)

            d_other = np.inf
            for cls in self.classes_:
                if cls == pred_cls:
                    continue
                if cls not in self.class_nn_models_:
                    continue

                d_cls = self._distance_to_class(z, cls)
                if d_cls < d_other:
                    d_other = d_cls

            raw_score = d_other / (d_pred + self.eps)

            if self.bounded:
                raw_score = raw_score / (1.0 + raw_score)

            scores[i] = raw_score

        if self.bounded:
            scores = np.clip(scores, 0.0, 1.0)

        return scores


def compute_trust_score(
    train_embeddings,
    train_labels,
    test_embeddings,
    predicted_labels,
    k: int = 1,
    metric: str = "euclidean",
    bounded: bool = True,
):
    computer = TrustScoreComputer(k=k, metric=metric, bounded=bounded)
    computer.fit(train_embeddings, train_labels)
    return computer.score(test_embeddings, predicted_labels)
