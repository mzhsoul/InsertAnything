from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _infer_obs_dim_from_obs_order(
    obs_order: list[str] | tuple[str, ...],
    *,
    append_prev_actions: bool = True,
) -> int:
    """Implementation helper."""
    if not isinstance(obs_order, (list, tuple)):
        raise TypeError("obs_order must be a list or tuple of observation component names.")

    dims = {
        "fingertip_pos_rel_fixed": 3,
        "fingertip_quat": 4,
        "fingertip_quat_rel_fixed": 4,
        "ee_linvel": 3,
        "ee_angvel": 3,
        "contact_force": 3,
        "prev_actions": 6,
    }

    normalized_order = [str(name) for name in obs_order]

    if any(name == "prev_actions" for name in normalized_order):
        raise ValueError(
            'Do not put "prev_actions" directly inside obs_order; keep it controlled by append_prev_actions.'
        )

    unknown = [name for name in normalized_order if name not in dims]
    if unknown:
        raise ValueError(f"Unsupported observation names in obs_order: {unknown}")

    if len(set(normalized_order)) != len(normalized_order):
        raise ValueError(f"Duplicate observation names in obs_order: {normalized_order}")

    total_dim = sum(dims[name] for name in normalized_order)
    if append_prev_actions:
        total_dim += dims["prev_actions"]

    return int(total_dim)


@dataclass(slots=True)
class TorchPolicyConfig:
    """Implementation helper."""

    obs_dim: int | None = None
    obs_order: list[str] | tuple[str, ...] | None = None
    append_prev_actions: bool = True

    action_dim: int = 6
    hidden_size: int = 1024
    num_layers: int = 2
    mlp_units: tuple[int, int, int] = (512, 128, 64)
    obs_norm_eps: float = 1e-5
    obs_clip: float | None = None
    device: str = "cpu"
    dtype: torch.dtype | str = torch.float32

    def __post_init__(self) -> None:
        """Implementation helper."""

        if isinstance(self.dtype, str):
            if self.dtype not in _DTYPE_MAP:
                raise ValueError(f"Unsupported dtype string: {self.dtype}. Supported: {list(_DTYPE_MAP.keys())}")
            self.dtype = _DTYPE_MAP[self.dtype]

        if not isinstance(self.mlp_units, tuple) or len(self.mlp_units) != 3:
            raise ValueError(f"mlp_units must be a tuple of length 3, got {self.mlp_units}")

        normalized_obs_order: tuple[str, ...] | None = None
        inferred_obs_dim: int | None = None

        if self.obs_order is not None:
            normalized_obs_order = tuple(str(x) for x in self.obs_order)
            inferred_obs_dim = _infer_obs_dim_from_obs_order(
                normalized_obs_order,
                append_prev_actions=self.append_prev_actions,
            )

        if self.obs_dim is None and inferred_obs_dim is None:
            raise ValueError("TorchPolicyConfig requires either obs_dim or obs_order.")

        if self.obs_dim is None:
            self.obs_dim = inferred_obs_dim
        elif inferred_obs_dim is not None and int(self.obs_dim) != int(inferred_obs_dim):
            raise ValueError(
                "obs_dim and obs_order are inconsistent: "
                f"obs_dim={self.obs_dim}, inferred_from_obs_order={inferred_obs_dim}"
            )

        self.obs_dim = int(self.obs_dim)

        if normalized_obs_order is not None:
            self.obs_order = normalized_obs_order

    @classmethod
    def from_basic_spec(
        cls,
        *,
        obs_dim: int | None = None,
        obs_order: list[str] | tuple[str, ...] | None = None,
        append_prev_actions: bool = True,
        action_dim: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 2,
        mlp_units: tuple[int, int, int] = (512, 128, 64),
        obs_norm_eps: float = 1e-5,
        obs_clip: float | None = None,
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
    ) -> TorchPolicyConfig:
        """Implementation helper."""
        return cls(
            obs_dim=obs_dim,
            obs_order=obs_order,
            append_prev_actions=append_prev_actions,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            mlp_units=mlp_units,
            obs_norm_eps=obs_norm_eps,
            obs_clip=obs_clip,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_obs_order(
        cls,
        obs_order: list[str] | tuple[str, ...],
        *,
        append_prev_actions: bool = True,
        action_dim: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 2,
        mlp_units: tuple[int, int, int] = (512, 128, 64),
        obs_norm_eps: float = 1e-5,
        obs_clip: float | None = None,
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
    ) -> TorchPolicyConfig:
        """Implementation helper."""
        return cls(
            obs_order=obs_order,
            append_prev_actions=append_prev_actions,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            mlp_units=mlp_units,
            obs_norm_eps=obs_norm_eps,
            obs_clip=obs_clip,
            device=device,
            dtype=dtype,
        )

    def to_serializable_dict(self) -> dict[str, Any]:
        """Implementation helper."""
        return {
            "obs_dim": int(self.obs_dim),
            "obs_order": list(self.obs_order) if self.obs_order is not None else None,
            "append_prev_actions": bool(self.append_prev_actions),
            "action_dim": int(self.action_dim),
            "hidden_size": int(self.hidden_size),
            "num_layers": int(self.num_layers),
            "mlp_units": list(self.mlp_units),
            "obs_norm_eps": float(self.obs_norm_eps),
            "obs_clip": None if self.obs_clip is None else float(self.obs_clip),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
        }


@dataclass
class PolicyState:

    h: torch.Tensor
    c: torch.Tensor


class TorchRecurrentActorPolicy(nn.Module):
    """Implementation helper."""

    def __init__(self, cfg: TorchPolicyConfig):
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "running_mean",
            torch.zeros(cfg.obs_dim, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "running_var",
            torch.ones(cfg.obs_dim, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "running_count",
            torch.tensor(1.0, dtype=torch.float64),
            persistent=True,
        )

        self.rnn = nn.LSTM(
            input_size=cfg.obs_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=False,
        )
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)

        self.actor_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.mlp_units[0]),
            nn.ELU(),
            nn.Linear(cfg.mlp_units[0], cfg.mlp_units[1]),
            nn.ELU(),
            nn.Linear(cfg.mlp_units[1], cfg.mlp_units[2]),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(cfg.mlp_units[2], cfg.action_dim)
        self.sigma_head = nn.Linear(cfg.mlp_units[2], cfg.action_dim)
        self.sigma_param = nn.Parameter(torch.zeros(cfg.action_dim))
        self._sigma_is_state_independent = False
        self.value_head = nn.Linear(cfg.mlp_units[2], 1)

        self.to(device=cfg.device, dtype=cfg.dtype)

    @classmethod
    def from_rlg_model_dict(
        cls,
        model_dict: dict,
        cfg: TorchPolicyConfig,
    ) -> TorchRecurrentActorPolicy:
        """Implementation helper."""
        policy = cls(cfg)
        policy.load_from_rlg_model_dict(model_dict)
        return policy

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cfg: TorchPolicyConfig,
        model_key: str = "model",
    ) -> TorchRecurrentActorPolicy:
        ckpt = cls._safe_torch_load(checkpoint_path)
        if model_key not in ckpt:
            raise KeyError(f"Checkpoint does not contain top-level key '{model_key}'.")
        return cls.from_rlg_model_dict(ckpt[model_key], cfg)

    @staticmethod
    def _safe_torch_load(path: str | Path):
        path = Path(path)
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def load_from_rlg_model_dict(self, model_dict: dict) -> None:
        """Implementation helper."""
        required_keys = [
            "running_mean_std.running_mean",
            "running_mean_std.running_var",
            "running_mean_std.count",
            "a2c_network.rnn.rnn.weight_ih_l0",
            "a2c_network.rnn.rnn.weight_hh_l0",
            "a2c_network.rnn.rnn.bias_ih_l0",
            "a2c_network.rnn.rnn.bias_hh_l0",
            "a2c_network.rnn.rnn.weight_ih_l1",
            "a2c_network.rnn.rnn.weight_hh_l1",
            "a2c_network.rnn.rnn.bias_ih_l1",
            "a2c_network.rnn.rnn.bias_hh_l1",
            "a2c_network.layer_norm.weight",
            "a2c_network.layer_norm.bias",
            "a2c_network.actor_mlp.0.weight",
            "a2c_network.actor_mlp.0.bias",
            "a2c_network.actor_mlp.2.weight",
            "a2c_network.actor_mlp.2.bias",
            "a2c_network.actor_mlp.4.weight",
            "a2c_network.actor_mlp.4.bias",
            "a2c_network.mu.weight",
            "a2c_network.mu.bias",
            "a2c_network.value.weight",
            "a2c_network.value.bias",
        ]
        missing = [k for k in required_keys if k not in model_dict]
        if missing:
            raise KeyError(f"Missing required keys in model_dict: {missing}")

        has_state_independent_sigma = "a2c_network.sigma" in model_dict
        has_state_dependent_sigma = all(
            key in model_dict for key in ("a2c_network.sigma.weight", "a2c_network.sigma.bias")
        )
        if not has_state_independent_sigma and not has_state_dependent_sigma:
            raise KeyError(
                "Checkpoint must contain either state-independent 'a2c_network.sigma' or "
                "state-dependent 'a2c_network.sigma.weight' and 'a2c_network.sigma.bias'."
            )

        self.running_mean.copy_(model_dict["running_mean_std.running_mean"].detach().cpu())
        self.running_var.copy_(model_dict["running_mean_std.running_var"].detach().cpu())
        self.running_count.copy_(model_dict["running_mean_std.count"].detach().cpu())

        self.rnn.weight_ih_l0.data.copy_(model_dict["a2c_network.rnn.rnn.weight_ih_l0"])
        self.rnn.weight_hh_l0.data.copy_(model_dict["a2c_network.rnn.rnn.weight_hh_l0"])
        self.rnn.bias_ih_l0.data.copy_(model_dict["a2c_network.rnn.rnn.bias_ih_l0"])
        self.rnn.bias_hh_l0.data.copy_(model_dict["a2c_network.rnn.rnn.bias_hh_l0"])

        self.rnn.weight_ih_l1.data.copy_(model_dict["a2c_network.rnn.rnn.weight_ih_l1"])
        self.rnn.weight_hh_l1.data.copy_(model_dict["a2c_network.rnn.rnn.weight_hh_l1"])
        self.rnn.bias_ih_l1.data.copy_(model_dict["a2c_network.rnn.rnn.bias_ih_l1"])
        self.rnn.bias_hh_l1.data.copy_(model_dict["a2c_network.rnn.rnn.bias_hh_l1"])

        self.layer_norm.weight.data.copy_(model_dict["a2c_network.layer_norm.weight"])
        self.layer_norm.bias.data.copy_(model_dict["a2c_network.layer_norm.bias"])

        self.actor_mlp[0].weight.data.copy_(model_dict["a2c_network.actor_mlp.0.weight"])
        self.actor_mlp[0].bias.data.copy_(model_dict["a2c_network.actor_mlp.0.bias"])

        self.actor_mlp[2].weight.data.copy_(model_dict["a2c_network.actor_mlp.2.weight"])
        self.actor_mlp[2].bias.data.copy_(model_dict["a2c_network.actor_mlp.2.bias"])

        self.actor_mlp[4].weight.data.copy_(model_dict["a2c_network.actor_mlp.4.weight"])
        self.actor_mlp[4].bias.data.copy_(model_dict["a2c_network.actor_mlp.4.bias"])

        self.mu_head.weight.data.copy_(model_dict["a2c_network.mu.weight"])
        self.mu_head.bias.data.copy_(model_dict["a2c_network.mu.bias"])

        self._sigma_is_state_independent = has_state_independent_sigma
        if has_state_independent_sigma:
            sigma = model_dict["a2c_network.sigma"].detach().reshape(-1)
            if sigma.numel() != self.cfg.action_dim:
                raise ValueError(
                    "State-independent sigma dimension does not match action_dim: "
                    f"sigma={sigma.numel()}, action_dim={self.cfg.action_dim}"
                )
            self.sigma_param.data.copy_(sigma)
        else:
            self.sigma_head.weight.data.copy_(model_dict["a2c_network.sigma.weight"])
            self.sigma_head.bias.data.copy_(model_dict["a2c_network.sigma.bias"])

        self.value_head.weight.data.copy_(model_dict["a2c_network.value.weight"])
        self.value_head.bias.data.copy_(model_dict["a2c_network.value.bias"])

    def get_initial_state(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> PolicyState:
        dev = device if device is not None else next(self.parameters()).device
        h = torch.zeros(
            self.cfg.num_layers,
            batch_size,
            self.cfg.hidden_size,
            device=dev,
            dtype=self.cfg.dtype,
        )
        c = torch.zeros_like(h)
        return PolicyState(h=h, c=c)

    def reset_state(
        self,
        state: PolicyState,
        reset_mask: torch.Tensor,
    ) -> PolicyState:
        """Implementation helper."""
        if reset_mask is None:
            return state

        if reset_mask.dtype != torch.bool:
            reset_mask = reset_mask.to(dtype=torch.bool)

        mask = reset_mask.view(1, -1, 1)
        state.h = state.h.masked_fill(mask, 0.0)
        state.c = state.c.masked_fill(mask, 0.0)
        return state

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: shape (B, obs_dim)
        obs_norm = (obs-mean)/sqrt(var+eps)
        """
        if obs.ndim != 2 or obs.shape[-1] != self.cfg.obs_dim:
            raise ValueError(f"Expected obs shape (B, {self.cfg.obs_dim}), got {tuple(obs.shape)}")

        mean = self.running_mean.to(device=obs.device, dtype=obs.dtype)
        var = self.running_var.to(device=obs.device, dtype=obs.dtype)

        obs_norm = (obs - mean) / torch.sqrt(var + self.cfg.obs_norm_eps)

        if self.cfg.obs_clip is not None:
            obs_norm = torch.clamp(obs_norm, -self.cfg.obs_clip, self.cfg.obs_clip)

        return obs_norm

    def forward(
        self,
        obs: torch.Tensor,
        state: PolicyState,
        reset_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> dict:
        """
        obs: (B, obs_dim)
        state.h/state.c: (num_layers, B, hidden_size)

        returns:
            {
                "action": (B, action_dim),
                "new_state": PolicyState,
                ... optional aux tensors ...
            }
        """
        if obs.ndim != 2:
            raise ValueError(f"Expected obs to have shape (B, obs_dim), got {tuple(obs.shape)}")

        obs = obs.to(device=next(self.parameters()).device, dtype=self.cfg.dtype)
        state = self.reset_state(state, reset_mask)

        obs_norm = self.normalize_obs(obs)

        rnn_in = obs_norm.unsqueeze(0)
        rnn_out, (new_h, new_c) = self.rnn(rnn_in, (state.h, state.c))
        rnn_out = rnn_out.squeeze(0)

        rnn_out_norm = self.layer_norm(rnn_out)
        actor_latent = self.actor_mlp(rnn_out_norm)

        mu = self.mu_head(actor_latent)
        if self._sigma_is_state_independent:
            sigma = self.sigma_param.unsqueeze(0).expand(obs.shape[0], -1)
        else:
            sigma = self.sigma_head(actor_latent)
        value = self.value_head(actor_latent)

        new_state = PolicyState(h=new_h, c=new_c)

        out = {
            "action": mu,
            "new_state": new_state,
        }

        if return_aux:
            out.update({
                "obs_norm": obs_norm,
                "rnn_output": rnn_out,
                "rnn_output_norm": rnn_out_norm,
                "actor_latent": actor_latent,
                "mu": mu,
                "sigma": sigma,
                "value": value,
            })

        return out

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state: PolicyState,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PolicyState]:

        out = self.forward(obs, state, reset_mask=reset_mask, return_aux=False)
        return out["action"], out["new_state"]
