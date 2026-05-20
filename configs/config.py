# configs/config.py

from pathlib import Path
import torch


class Config:
    def __init__(self):

        # =============================
        # Path settings / 路径设置
        # =============================

        self.root_dir = Path(__file__).resolve().parent.parent

        self.data_dir = self.root_dir / "data"

        # Centralized-learning data / 集中式学习数据
        self.train_csv_path = (
            self.data_dir / "centralized" / "trainData_cleaned.csv"
        )
        self.test_csv_path = (
            self.data_dir / "test" / "testData_cleaned.csv"
        )

        # Federated-learning client data / 联邦学习客户端数据
        self.federated_dir = self.data_dir / "federated"
        self.federated_client_paths = sorted(
            self.federated_dir.glob("*.csv")
        )

        # Output directories / 结果输出目录
        self.results_dir = self.root_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.logs_dir = self.results_dir / "logs"
        self.models_dir = self.results_dir / "models"

        for _dir in [
            self.results_dir,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
            self.models_dir,
        ]:
            _dir.mkdir(parents=True, exist_ok=True)

        # =============================
        # Data settings / 数据设置
        # =============================

        self.target_col = "outcomeType"

        self.feature_cols = [
            "patientGender",
            "patientAge",
            "glasgowScale",
            "hematocrit",
            "hemoglobin",
            "leucocitos",
            "lymphocytes",
            "urea",
            "creatinine",
            "platelets",
            "diuresis",
            "SBP",
            "DBP",
            "glasgowScale_missing",
        ]

        self.input_dim = len(self.feature_cols)

        self.random_state = 42
        self.val_size = 0.15

        # =============================
        # Training settings / 训练设置
        # =============================

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 1e-4

        # Centralized training / 集中式训练
        self.num_epochs = 80
        self.early_stopping_patience = 12

        # Federated training / 联邦训练
        self.num_clients = len(self.federated_client_paths)
        self.global_rounds = 20
        self.local_epochs = 2

        # =============================
        # Model settings / 模型设置
        # =============================

        self.hidden_dim = 64
        self.embed_dim = 32
        self.dropout = 0.2

        # =============================
        # GMM settings / GMM 可靠性估计设置
        # =============================

        self.gmm_components = 3
        self.gmm_covariance_type = "full"
        self.gmm_reg_covar = 1e-5


# =============================
# Global config instance / 全局配置实例
# =============================

_config = Config()

# =============================
# Export variables for legacy code compatibility / 导出变量以兼容旧版代码
# =============================

BATCH_SIZE = _config.batch_size
CENTRALIZED_TRAIN_PATH = _config.train_csv_path
TEST_PATH = _config.test_csv_path

DEVICE = _config.device
DROPOUT = _config.dropout
EMBED_DIM = _config.embed_dim
HIDDEN_DIM = _config.hidden_dim

LR = _config.lr
WEIGHT_DECAY = _config.weight_decay
NUM_EPOCHS = _config.num_epochs
EARLY_STOPPING_PATIENCE = _config.early_stopping_patience

FEATURE_COLS = _config.feature_cols
TARGET_COL = _config.target_col
VAL_SIZE = _config.val_size
RANDOM_SEED = _config.random_state

FIGURES_DIR = _config.figures_dir
TABLES_DIR = _config.tables_dir
LOGS_DIR = _config.logs_dir
MODELS_DIR = _config.models_dir

GMM_COMPONENTS = _config.gmm_components
GMM_COVARIANCE_TYPE = _config.gmm_covariance_type
GMM_REG_COVAR = _config.gmm_reg_covar
