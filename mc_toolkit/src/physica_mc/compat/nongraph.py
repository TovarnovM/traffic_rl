"""Owned inference cores copied from Physica A nongraph_ppo; no RLlib wrappers."""
from __future__ import annotations
from collections.abc import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

class _SelfResidualLayer(nn.Module):
    """Graph-layer-shaped node MLP with no inter-node communication."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        self, node_embeddings: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        self_message = self.message(node_embeddings)
        update = self.update(torch.cat((node_embeddings, self_message), dim=-1))
        output = self.norm(node_embeddings + update)
        return F.relu(output) * node_mask.unsqueeze(-1)


class NodeMlpCore(nn.Module):
    """Shared per-node MLP actor with a pooled centralized critic."""

    def __init__(
        self,
        *,
        node_dim: int,
        global_dim: int,
        max_nodes: int,
        hidden_dim: int,
        layer_count: int,
    ) -> None:
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_layers = nn.ModuleList(
            _SelfResidualLayer(hidden_dim) for _ in range(layer_count)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, observation: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = observation["node_features"].float()
        node_mask = observation["node_mask"].float()
        action_mask = observation["action_mask"].bool()
        global_features = observation["global_features"].float()

        global_embedding = self.global_encoder(global_features)
        nodes = self.node_encoder(node_features)
        nodes = F.relu(nodes + global_embedding.unsqueeze(1))
        nodes = nodes * node_mask.unsqueeze(-1)
        for layer in self.self_layers:
            nodes = layer(nodes, node_mask)

        repeated_global = global_embedding.unsqueeze(1).expand(
            -1, self.max_nodes, -1
        )
        logits = self.actor_head(torch.cat((nodes, repeated_global), dim=-1))
        logits = logits.masked_fill(~action_mask, -1.0e9)

        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (nodes * node_mask.unsqueeze(-1)).sum(dim=1) / denominator
        value = self.value_head(
            torch.cat((pooled, global_embedding), dim=-1)
        ).squeeze(-1)
        return logits, value, node_mask


class _DilatedGridResidualLayer(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int) -> None:
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.spatial = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 3),
            dilation=(1, dilation),
            padding=(1, dilation),
        )
        self.spatial_norm = nn.GroupNorm(groups, hidden_dim)
        self.channel = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.channel_norm = nn.GroupNorm(groups, hidden_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        update = F.relu(self.spatial_norm(self.spatial(grid)))
        update = self.channel_norm(self.channel(update))
        return F.relu(grid + update)


class GridCnnCore(nn.Module):
    """PV-centred fixed-grid CNN that never consumes dynamic graph tensors."""

    def __init__(
        self,
        *,
        node_dim: int,
        global_dim: int,
        max_nodes: int,
        num_lanes: int,
        front_distance: int,
        back_distance: int,
        hidden_dim: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.max_nodes = int(max_nodes)
        self.num_lanes = int(num_lanes)
        self.front_distance = int(front_distance)
        self.back_distance = int(back_distance)
        self.grid_width = self.front_distance + self.back_distance + 1
        expected_nodes = self.num_lanes * self.grid_width
        if self.max_nodes != expected_nodes:
            raise ValueError(
                f"grid CNN expected {expected_nodes} slots from geometry, "
                f"observation exposes {self.max_nodes}"
            )

        groups = 8 if hidden_dim % 8 == 0 else 1
        # The extra raster channel marks active AV cells.  HDV and PV context
        # remains present in the existing engineered node features.
        self.stem = nn.Conv2d(node_dim + 1, hidden_dim, kernel_size=1)
        self.stem_norm = nn.GroupNorm(groups, hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_layers = nn.ModuleList(
            _DilatedGridResidualLayer(hidden_dim, dilation)
            for dilation in dilations
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _slot_coordinates(
        self, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_position = node_features[..., 0]
        signed_position = torch.where(
            relative_position >= 0.0,
            torch.round(relative_position * self.front_distance),
            torch.round(relative_position * self.back_distance),
        ).long()
        column = (signed_position + self.back_distance).clamp(
            0, self.grid_width - 1
        )
        if self.num_lanes == 1:
            lane = torch.zeros_like(column)
        else:
            lane = torch.round(
                node_features[..., 1] * (self.num_lanes - 1)
            ).long().clamp(0, self.num_lanes - 1)
        return lane, column

    def _rasterize(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        lane: torch.Tensor,
        column: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = node_features.shape[0]
        channels_last = node_features.new_zeros(
            batch_size,
            self.num_lanes,
            self.grid_width,
            self.node_dim + 1,
        )
        batch_index, slot_index = torch.nonzero(
            node_mask.bool(), as_tuple=True
        )
        if batch_index.numel() > 0:
            selected_lane = lane[batch_index, slot_index]
            selected_column = column[batch_index, slot_index]
            channels_last[
                batch_index,
                selected_lane,
                selected_column,
                : self.node_dim,
            ] = node_features[batch_index, slot_index]
            channels_last[
                batch_index,
                selected_lane,
                selected_column,
                self.node_dim,
            ] = 1.0
        return channels_last.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _gather_slots(
        grid_embeddings: torch.Tensor,
        lane: torch.Tensor,
        column: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _hidden_dim, _lanes, _width = grid_embeddings.shape
        batch_index = torch.arange(
            batch_size, device=grid_embeddings.device
        ).unsqueeze(1).expand_as(lane)
        channels_last = grid_embeddings.permute(0, 2, 3, 1)
        slots = channels_last[batch_index, lane, column]
        return slots * node_mask.unsqueeze(-1)

    def forward(
        self, observation: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = observation["node_features"].float()
        node_mask = observation["node_mask"].float()
        action_mask = observation["action_mask"].bool()
        global_features = observation["global_features"].float()
        lane, column = self._slot_coordinates(node_features)
        raster = self._rasterize(node_features, node_mask, lane, column)

        global_embedding = self.global_encoder(global_features)
        grid = self.stem_norm(self.stem(raster))
        grid = F.relu(grid + global_embedding.unsqueeze(-1).unsqueeze(-1))
        for layer in self.residual_layers:
            grid = layer(grid)
        slots = self._gather_slots(grid, lane, column, node_mask)

        logits = self.actor_head(slots).masked_fill(~action_mask, -1.0e9)
        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (slots * node_mask.unsqueeze(-1)).sum(dim=1) / denominator
        value = self.value_head(
            torch.cat((pooled, global_embedding), dim=-1)
        ).squeeze(-1)
        return logits, value, node_mask

