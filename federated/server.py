# federated/server.py

import copy
from models.classifier import MedicalMLP


class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.global_model = MedicalMLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        ).to(self.device)

    def aggregate(self, client_updates):
        total_samples = sum([num for _, num in client_updates])

        new_weights = copy.deepcopy(client_updates[0][0])

        for key in new_weights.keys():
            new_weights[key] = sum(
                client_state[key] * (num / total_samples)
                for client_state, num in client_updates
            )

        self.global_model.load_state_dict(new_weights)

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())