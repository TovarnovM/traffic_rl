"""Minimal RLlib/Torch graph model and factorized PPO loss for PV control.

This module intentionally uses RLlib's classic Policy/ModelV2 stack.  It keeps
the implementation small and gives the loss direct access to the old and new
per-node logits required for nodewise PPO clipping.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "graph PPO requires PyTorch; install local training dependencies with "
        "'python -m pip install -e \".[rl,ray,numba]\"'"
    ) from exc

try:
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
    from ray.rllib.evaluation.postprocessing import Postprocessing
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.policy.sample_batch import SampleBatch
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "graph PPO requires RLlib; install local training dependencies with "
        "'python -m pip install -e \".[rl,ray,numba]\"'"
    ) from exc


class _MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
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
        node_embeddings: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, neighbor_count = neighbor_index.shape
        batch_index = torch.arange(
            batch_size, device=node_embeddings.device
        ).view(batch_size, 1, 1)
        batch_index = batch_index.expand(batch_size, node_count, neighbor_count)
        neighbor_embeddings = node_embeddings[batch_index, neighbor_index]
        messages = self.message(
            torch.cat((neighbor_embeddings, edge_features), dim=-1)
        )
        neighbor_weight = neighbor_mask.unsqueeze(-1)
        messages = messages * neighbor_weight
        denominator = neighbor_weight.sum(dim=2).clamp_min(1.0)
        aggregated = messages.sum(dim=2) / denominator
        update = self.update(torch.cat((node_embeddings, aggregated), dim=-1))
        output = self.norm(node_embeddings + update)
        return F.relu(output) * node_mask.unsqueeze(-1)


class PvGraphTorchModel(TorchModelV2, nn.Module):
    """Two message-passing layers, a shared node actor, and graph critic."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        original_space = getattr(obs_space, "original_space", obs_space)
        component_spaces = getattr(original_space, "spaces", original_space)
        node_shape = tuple(component_spaces["node_features"].shape)
        edge_shape = tuple(component_spaces["edge_features"].shape)
        global_shape = tuple(component_spaces["global_features"].shape)
        self.max_nodes = int(node_shape[0])
        self.action_count = 3
        expected_outputs = self.max_nodes * self.action_count
        if num_outputs != expected_outputs:
            raise ValueError(
                f"graph model requires {expected_outputs} action logits, got {num_outputs}"
            )
        custom = dict(model_config.get("custom_model_config", {}))
        hidden_dim = int(custom.get("hidden_dim", 64))
        message_layers = int(custom.get("message_layers", 2))
        if hidden_dim < 1 or message_layers < 1:
            raise ValueError("hidden_dim and message_layers must be >= 1")

        self.node_encoder = nn.Sequential(
            nn.Linear(int(node_shape[1]), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(int(global_shape[0]), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            _MessagePassingLayer(hidden_dim, int(edge_shape[2]))
            for _ in range(message_layers)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_count),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._last_value: torch.Tensor | None = None
        self._last_node_mask: torch.Tensor | None = None

    @property
    def last_node_mask(self) -> torch.Tensor:
        if self._last_node_mask is None:
            raise RuntimeError("model forward must run before reading node mask")
        return self._last_node_mask

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        node_features = obs["node_features"].float()
        node_mask = obs["node_mask"].float()
        neighbor_index = obs["neighbor_index"].long()
        neighbor_mask = obs["neighbor_mask"].float()
        edge_features = obs["edge_features"].float()
        action_mask = obs["action_mask"].bool()
        global_features = obs["global_features"].float()

        global_embedding = self.global_encoder(global_features)
        node_embeddings = self.node_encoder(node_features)
        node_embeddings = F.relu(
            node_embeddings + global_embedding.unsqueeze(1)
        ) * node_mask.unsqueeze(-1)
        for layer in self.message_layers:
            node_embeddings = layer(
                node_embeddings,
                neighbor_index,
                neighbor_mask,
                edge_features,
                node_mask,
            )

        repeated_global = global_embedding.unsqueeze(1).expand(
            -1, self.max_nodes, -1
        )
        logits = self.actor_head(
            torch.cat((node_embeddings, repeated_global), dim=-1)
        )
        logits = logits.masked_fill(~action_mask, -1.0e9)

        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (node_embeddings * node_mask.unsqueeze(-1)).sum(dim=1) / denominator
        self._last_value = self.value_head(
            torch.cat((pooled, global_embedding), dim=-1)
        ).squeeze(-1)
        self._last_node_mask = node_mask
        return logits.reshape(logits.shape[0], -1), state

    def value_function(self):
        if self._last_value is None:
            raise RuntimeError("model forward must run before value_function")
        return self._last_value


def _masked_node_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / denominator


def _explained_variance(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target_variance = torch.var(target, unbiased=False)
    residual_variance = torch.var(target - prediction, unbiased=False)
    return torch.where(
        target_variance > 1.0e-8,
        1.0 - residual_variance / target_variance,
        torch.zeros_like(target_variance),
    )


def factorized_ppo_loss(policy, model, dist_class, train_batch):
    """PPO clipping per active graph node, averaged once per env step."""

    del dist_class
    logits, _state = model(train_batch)
    node_mask = model.last_node_mask
    batch_size = logits.shape[0]
    max_nodes = model.max_nodes
    new_logits = logits.reshape(batch_size, max_nodes, 3)
    old_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS].reshape(
        batch_size, max_nodes, 3
    )
    actions = train_batch[SampleBatch.ACTIONS].long().reshape(
        batch_size, max_nodes
    )

    new_log_probs_all = F.log_softmax(new_logits, dim=-1)
    old_log_probs_all = F.log_softmax(old_logits, dim=-1)
    selected = actions.unsqueeze(-1)
    new_log_probs = new_log_probs_all.gather(-1, selected).squeeze(-1)
    old_log_probs = old_log_probs_all.gather(-1, selected).squeeze(-1)
    ratios = torch.exp((new_log_probs - old_log_probs).clamp(-20.0, 20.0))

    advantages = train_batch[Postprocessing.ADVANTAGES].float().reshape(
        batch_size, 1
    )
    clip_param = float(policy.config["clip_param"])
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
    per_node_surrogate = torch.minimum(unclipped, clipped)
    mean_policy_loss = -_masked_node_mean(
        per_node_surrogate, node_mask
    ).mean()

    probabilities = torch.softmax(new_logits, dim=-1)
    entropy_per_node = -(
        probabilities * new_log_probs_all
    ).sum(dim=-1)
    mean_entropy = _masked_node_mean(entropy_per_node, node_mask).mean()
    old_probabilities = torch.softmax(old_logits, dim=-1)
    kl_per_node = (
        old_probabilities * (old_log_probs_all - new_log_probs_all)
    ).sum(dim=-1)
    mean_kl = _masked_node_mean(kl_per_node, node_mask).mean()

    value_prediction = model.value_function()
    if bool(policy.config.get("use_critic", True)):
        value_targets = train_batch[Postprocessing.VALUE_TARGETS].float()
        previous_values = train_batch[SampleBatch.VF_PREDS].float()
        vf_clip = float(policy.config["vf_clip_param"])
        value_clipped = previous_values + torch.clamp(
            value_prediction - previous_values, -vf_clip, vf_clip
        )
        vf_loss_unclipped = torch.square(value_prediction - value_targets)
        vf_loss_clipped = torch.square(value_clipped - value_targets)
        mean_vf_loss = torch.maximum(
            vf_loss_unclipped, vf_loss_clipped
        ).mean()
        vf_explained_var = _explained_variance(value_targets, value_prediction)
    else:
        mean_vf_loss = torch.zeros((), device=logits.device)
        vf_explained_var = torch.zeros((), device=logits.device)

    entropy_coeff = float(
        getattr(policy, "entropy_coeff", policy.config.get("entropy_coeff", 0.0))
    )
    kl_coeff = getattr(policy, "kl_coeff", policy.config.get("kl_coeff", 0.0))
    vf_loss_coeff = float(policy.config.get("vf_loss_coeff", 1.0))
    total_loss = (
        mean_policy_loss
        + vf_loss_coeff * mean_vf_loss
        - entropy_coeff * mean_entropy
        + kl_coeff * mean_kl
    )

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = vf_explained_var
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl
    model.tower_stats["mean_active_nodes"] = node_mask.sum(dim=-1).mean()
    return total_loss


class FactorizedPPOTorchPolicy(PPOTorchPolicy):
    """PPO Torch policy with the nodewise factorized loss.

    Current RLlib releases expose ``PPOTorchPolicy`` as a normal class whose
    ``loss`` method is intended to be overridden.  Older generated-policy
    examples used ``with_updates()``, but that factory is no longer present in
    recent Ray versions.  Direct subclassing also works with the older classic
    Policy stack used by this training runner.
    """

    def loss(self, model, dist_class, train_batch):
        return factorized_ppo_loss(self, model, dist_class, train_batch)


class FactorizedGraphPPO(PPO):
    """Classic-stack PPO using the nodewise factorized Torch policy."""

    @classmethod
    def get_default_policy_class(cls, config):
        framework = (
            config.get("framework")
            if hasattr(config, "get")
            else getattr(config, "framework_str", None)
        )
        if framework not in {"torch", None}:
            raise ValueError("FactorizedGraphPPO supports only framework='torch'")
        return FactorizedPPOTorchPolicy


__all__ = [
    "FactorizedGraphPPO",
    "FactorizedPPOTorchPolicy",
    "PvGraphTorchModel",
    "factorized_ppo_loss",
]
