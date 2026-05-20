# federated/client.py

import copy
import torch
import torch.nn as nn
from models.classifier import MedicalMLP
from utils.data_utils import load_federated_client_data, create_dataloader


class FederatedClient:
    def __init__(self, client_id, config):
        self.client_id = client_id
        self.config = config
        self.device = config.device

        self.model = MedicalMLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        ).to(self.device)

        # Use the configured client CSV path / 使用配置中的客户端 CSV 路径
        csv_path = config.federated_client_paths[client_id]

        X, y = load_federated_client_data(
            csv_path,
            config.feature_cols,
            config.target_col,
        )

        self.train_loader = create_dataloader(
            X,
            y,
            batch_size=config.batch_size,
            shuffle=True,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def set_weights(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr
        )

        for _ in range(self.config.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.float().to(self.device)

                optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()

        return copy.deepcopy(self.model.state_dict()), len(self.train_loader.dataset)
