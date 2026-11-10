"""Owned graph inference copy of Physica A evaluate_pv_graph_ppo._make_inference_model."""
import torch
import torch.nn as nn
import torch.nn.functional as functional

def build_graph(hidden_dim, message_layer_count, max_nodes):
    class MessagePassingLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.message = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            node_embeddings,
            neighbor_index,
            neighbor_mask,
            edge_features,
            node_mask,
        ):
            batch_size, node_count, neighbor_count = neighbor_index.shape
            batch_index = torch.arange(
                batch_size, device=node_embeddings.device
            ).view(batch_size, 1, 1)
            batch_index = batch_index.expand(
                batch_size, node_count, neighbor_count
            )
            neighbors = node_embeddings[batch_index, neighbor_index]
            messages = self.message(torch.cat((neighbors, edge_features), dim=-1))
            weights = neighbor_mask.unsqueeze(-1)
            messages = messages * weights
            aggregated = messages.sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
            update = self.update(torch.cat((node_embeddings, aggregated), dim=-1))
            output = self.norm(node_embeddings + update)
            return functional.relu(output) * node_mask.unsqueeze(-1)

    class InferencePvGraphModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_nodes = max_nodes
            self.node_encoder = nn.Sequential(
                nn.Linear(48, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.message_layers = nn.ModuleList(
                MessagePassingLayer() for _ in range(message_layer_count)
            )
            self.actor_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
            # It is unused for action selection, but retaining it makes strict
            # checkpoint loading catch any model mismatch.
            self.value_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, observation):
            node_features = observation["node_features"].float()
            node_mask = observation["node_mask"].float()
            neighbor_index = observation["neighbor_index"].long()
            neighbor_mask = observation["neighbor_mask"].float()
            edge_features = observation["edge_features"].float()
            action_mask = observation["action_mask"].bool()
            global_features = observation["global_features"].float()

            global_embedding = self.global_encoder(global_features)
            nodes = self.node_encoder(node_features)
            nodes = functional.relu(
                nodes + global_embedding.unsqueeze(1)
            ) * node_mask.unsqueeze(-1)
            for layer in self.message_layers:
                nodes = layer(
                    nodes,
                    neighbor_index,
                    neighbor_mask,
                    edge_features,
                    node_mask,
                )
            repeated_global = global_embedding.unsqueeze(1).expand(
                -1, self.max_nodes, -1
            )
            logits = self.actor_head(torch.cat((nodes, repeated_global), dim=-1))
            return logits.masked_fill(~action_mask, -1.0e9)

    return InferencePvGraphModel()
