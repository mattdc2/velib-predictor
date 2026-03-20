"""LSTM architecture for Velib bike-availability forecasting.

Architecture overview
---------------------
::

    station_id  ──► StationEmbedding ──► emb (B, E)
                                              │
    x (B, T, F) ──────────────────────────────┤ cat per time-step
                                              ▼
                                    InputProjection  (B, T, F+E) → (B, T, H)
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   LSTM Encoder  │  2 layers × H units
                                    │   dropout = 0.2 │  batch_first = True
                                    └────────┬────────┘
                                     h_n[-1] │  last-layer hidden (B, H)
                                             │
                                    cat(h_n[-1], emb) → context (B, H+E)
                                         │              │
                                         ▼              ▼
                               MechanicalHead      EBikeHead
                               Linear(H+E, Hh)    Linear(H+E, Hh)
                               ReLU               ReLU
                               Linear(Hh, 1)      Linear(Hh, 1)
                                         │              │
                                    num_mechanical  num_ebike
                                         └──────┬───────┘
                                           num_bikes_available
                                           (= mech + ebike)

Legend: B=batch, T=seq_len, F=n_features, E=embed_dim, H=hidden_size, Hh=head_dim

Design decisions
----------------
* **Station embedding** — each station_id maps to a learnable dense vector.
  The embedding is broadcast across the sequence so the LSTM sees station
  identity at every time step, and is again concatenated to the final hidden
  state to give the output heads direct access to station context.

* **Two output heads** — ``num_mechanical`` and ``num_ebike`` are predicted
  independently, matching the ``station_status`` schema.  ``total`` is derived
  as their sum rather than predicted separately, to avoid inconsistent outputs.

* **InputProjection** — a single linear layer that projects the concatenated
  (feature + embedding) input into the LSTM's hidden dimension.  This decouples
  the feature size from the LSTM width and lets us add a ``LayerNorm`` before
  the recurrent layers without changing the LSTM interface.

* **No sigmoid/softplus** on outputs — the downstream training loss (MAE/MSE)
  is applied on raw logits; clamping to non-negative values is done at inference
  time by the predictor (task 5.3).

Public API
----------
* :class:`LSTMConfig`   — Pydantic model holding all hyperparameters
* :class:`LSTMOutput`   — dataclass returned by :meth:`VelibLSTM.forward`
* :class:`VelibLSTM`    — the main ``nn.Module``
* :func:`build_model`   — convenience constructor from an :class:`LSTMConfig`
* :func:`diagram`       — print an ASCII architecture diagram
* :func:`count_parameters` — count trainable parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LSTMConfig(BaseModel):
    """Hyperparameter specification for :class:`VelibLSTM`.

    Attributes:
        n_stations:    Number of unique stations (embedding table size).
        n_features:    Number of numeric features per time step (input width).
        seq_len:       Input sequence length in time steps (default: ``24``).
        embed_dim:     Station embedding dimension (default: ``16``).
        hidden_size:   LSTM hidden units per layer (default: ``128``).
        num_layers:    Number of stacked LSTM layers (default: ``2``).
        dropout:       Dropout between LSTM layers; ignored when
                       ``num_layers == 1`` (default: ``0.2``).
        head_dim:      Hidden units in each output-head MLP (default: ``64``).
        layer_norm:    Whether to apply LayerNorm after the input projection
                       (default: ``True``).
    """

    n_stations: int
    n_features: int
    seq_len: int = 24
    embed_dim: int = 16
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    head_dim: int = 64
    layer_norm: bool = True

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class LSTMOutput:
    """Predictions returned by :meth:`VelibLSTM.forward`.

    Attributes:
        mechanical: Predicted ``num_mechanical`` bikes, shape ``(B, 1)``.
        ebike:      Predicted ``num_ebike`` bikes, shape ``(B, 1)``.
    """

    mechanical: torch.Tensor
    ebike: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        """Sum of mechanical and e-bike predictions, shape ``(B, 1)``."""
        return self.mechanical + self.ebike

    def clamp(self, min_val: float = 0.0) -> "LSTMOutput":
        """Return a new :class:`LSTMOutput` with predictions clamped to *min_val*."""
        return LSTMOutput(
            mechanical=self.mechanical.clamp(min=min_val),
            ebike=self.ebike.clamp(min=min_val),
        )


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _StationEmbedding(nn.Module):
    """Learnable embedding for station identity.

    Maps integer station indices to dense vectors of dimension *embed_dim*.
    An optional dropout discourages over-reliance on station identity alone.
    """

    def __init__(self, n_stations: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embed_dim, padding_idx=None)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, station_ids: torch.Tensor) -> torch.Tensor:
        """Return station embeddings, shape ``(B, embed_dim)``."""
        return self.drop(self.embedding(station_ids))


class _OutputHead(nn.Module):
    """Two-layer MLP output head for a single prediction target.

    Architecture: Linear → ReLU → Dropout → Linear → scalar
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions, shape ``(B, 1)``."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class VelibLSTM(nn.Module):
    """Multi-output LSTM forecasting model for Velib station availability.

    Inputs
    ------
    ``x``:          Float tensor ``(B, T, F)`` — B batch size, T sequence
                    length, F number of features per time step.
    ``station_ids``: Long tensor ``(B,)`` — integer station indices in
                    ``[0, n_stations)``.

    Outputs
    -------
    :class:`LSTMOutput` with ``mechanical`` and ``ebike`` tensors of shape
    ``(B, 1)``.  Access ``.total`` for the combined prediction.

    Args:
        config: :class:`LSTMConfig` instance holding all hyperparameters.
    """

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config

        # ── Station embedding ──────────────────────────────────────────────
        self.station_embedding = _StationEmbedding(
            n_stations=config.n_stations,
            embed_dim=config.embed_dim,
        )

        # ── Input projection ──────────────────────────────────────────────
        # Projects (features + embedding) → hidden_size so the LSTM input
        # width matches hidden_size (simplifies stacked LSTM initialisation).
        input_dim = config.n_features + config.embed_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        self.input_norm: nn.Module = (
            nn.LayerNorm(config.hidden_size) if config.layer_norm else nn.Identity()
        )

        # ── LSTM encoder ──────────────────────────────────────────────────
        # dropout is applied between layers; has no effect when num_layers=1
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # ── Output heads ──────────────────────────────────────────────────
        # Context fed to each head: last hidden state + station embedding
        context_dim = config.hidden_size + config.embed_dim
        self.mechanical_head = _OutputHead(context_dim, config.head_dim, dropout=config.dropout)
        self.ebike_head = _OutputHead(context_dim, config.head_dim, dropout=config.dropout)

        logger.info(
            f"VelibLSTM initialised — stations={config.n_stations}, "
            f"features={config.n_features}, seq_len={config.seq_len}, "
            f"hidden={config.hidden_size}×{config.num_layers}, "
            f"embed={config.embed_dim}, params={count_parameters(self):,}"
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        station_ids: torch.Tensor,
    ) -> LSTMOutput:
        """Run a forward pass.

        Args:
            x:           Feature sequence, shape ``(B, T, F)``.
            station_ids: Station indices, shape ``(B,)``.

        Returns:
            :class:`LSTMOutput` with ``mechanical`` and ``ebike`` predictions.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Station embedding → broadcast over sequence
        emb = self.station_embedding(station_ids)  # (B, E)
        emb_seq = emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, E)

        # 2. Concatenate embedding with per-step features
        x_aug = torch.cat([x, emb_seq], dim=-1)  # (B, T, F+E)

        # 3. Project + normalise input
        x_proj = self.input_norm(self.input_proj(x_aug))  # (B, T, H)

        # 4. LSTM encoding
        _, (h_n, _) = self.lstm(x_proj)  # h_n: (layers, B, H)
        last_hidden = h_n[-1]  # (B, H)

        # 5. Build context: last hidden state + station embedding
        context = torch.cat([last_hidden, emb], dim=-1)  # (B, H+E)

        # 6. Output heads
        mechanical = self.mechanical_head(context)  # (B, 1)
        ebike = self.ebike_head(context)  # (B, 1)

        return LSTMOutput(mechanical=mechanical, ebike=ebike)

    # ------------------------------------------------------------------
    def reset_hidden(self) -> None:
        """No-op — LSTM state is not persistent across batches.

        Included for API symmetry with stateful RNN variants.
        """

    # ------------------------------------------------------------------
    def encode(
        self,
        x: torch.Tensor,
        station_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the context vector without passing through output heads.

        Useful for transfer learning, probing, or downstream tasks that
        need the learned representation.

        Args:
            x:           Feature sequence, shape ``(B, T, F)``.
            station_ids: Station indices, shape ``(B,)``.

        Returns:
            Context tensor of shape ``(B, hidden_size + embed_dim)``.
        """
        batch_size, seq_len, _ = x.shape
        emb = self.station_embedding(station_ids)
        emb_seq = emb.unsqueeze(1).expand(-1, seq_len, -1)
        x_aug = torch.cat([x, emb_seq], dim=-1)
        x_proj = self.input_norm(self.input_proj(x_aug))
        _, (h_n, _) = self.lstm(x_proj)
        return torch.cat([h_n[-1], emb], dim=-1)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_model(config: LSTMConfig, device: Optional[str] = None) -> VelibLSTM:
    """Construct a :class:`VelibLSTM` and move it to *device*.

    Args:
        config: Model hyperparameters.
        device: PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
                Auto-detected when ``None``.

    Returns:
        Initialised :class:`VelibLSTM` on the requested device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = VelibLSTM(config).to(device)
    logger.info(f"VelibLSTM on device='{device}'")
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def diagram(config: Optional[LSTMConfig] = None) -> str:
    """Return an ASCII architecture diagram as a string.

    Args:
        config: When provided, fills in concrete dimension values.
                Uses placeholder symbols when ``None``.

    Returns:
        Multi-line string — print it or embed it in a notebook cell.
    """
    if config is not None:
        F = config.n_features
        E = config.embed_dim
        H = config.hidden_size
        L = config.num_layers
        Hh = config.head_dim
        T = config.seq_len
        ctx = H + E
    else:
        F, E, H, L, Hh, T, ctx = "F", "E", "H", "L", "Hh", "T", "H+E"  # type: ignore[assignment]
    lines = [
        "┌──────────────────────────────────────────────────────────────────┐",
        "│                         VelibLSTM                                │",
        "├──────────────────────────────────────────────────────────────────┤",
        "│                                                                  │",
        "│  station_ids (B,)                                               │",
        "│       │                                                         │",
        "│       ▼                                                         │",
        "│  ┌──────────────────────────────────┐                           │",
        "│  │  StationEmbedding                │                           │",
        f"│  │  Embedding(n_stations, {E})      │──► emb (B, {E})           │",
        "│  └──────────────────────────────────┘         │                 │",
        "│                                               │ broadcast (T)   │",
        f"│  x (B, T={T}, F={F})  ────────────────────────┤                 │",
        "│                                     cat along F axis            │",
        "│                                               ▼                 │",
        "│  ┌──────────────────────────────────────────────────────┐       │",
        f"│  │  InputProjection  Linear({F}+{E}, {H})  +  LayerNorm │       │",
        "│  └──────────────────────────┬───────────────────────────┘       │",
        f"│                             │  (B, T, {H})                      │",
        "│                             ▼                                   │",
        "│  ┌──────────────────────────────────────────────────────┐       │",
        "│  │  LSTM Encoder                                        │       │",
        f"│  │  input={H}  hidden={H}  layers={L}  dropout=0.2      │       │",
        "│  └──────────────────────────┬───────────────────────────┘       │",
        f"│                      h_n[-1]│  last hidden  (B, {H})            │",
        "│                             │                                   │",
        f"│              cat(h_n[-1], emb) → context (B, {ctx})             │",
        "│                      │                  │                       │",
        "│                      ▼                  ▼                       │",
        "│         ┌────────────────────┐  ┌────────────────────┐          │",
        "│         │  MechanicalHead    │  │    EBikeHead       │          │",
        f"│         │  Linear({ctx},{Hh})│  │  Linear({ctx},{Hh})│          │",
        "│         │  ReLU              │  │  ReLU              │          │",
        f"│         │  Linear({Hh}, 1)   │  │  Linear({Hh}, 1)   │          │",
        "│         └─────────┬──────────┘  └──────────┬─────────┘          │",
        "│               num_mechanical           num_ebike                │",
        "│                      │                  │                       │",
        "│                      └────────┬─────────┘                       │",
        "│                         (= mechanical + ebike)                  │",
        "│                         num_bikes_available                     │",
        "│                                                                  │",
        "└──────────────────────────────────────────────────────────────────┘",
    ]

    return "\n".join(lines)
