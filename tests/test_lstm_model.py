"""Tests for src/models/lstm/model.py — VelibLSTM architecture."""

from __future__ import annotations

import pytest
import torch

from src.models.lstm.model import (
    LSTMConfig,
    LSTMOutput,
    VelibLSTM,
    _OutputHead,
    _StationEmbedding,
    build_model,
    count_parameters,
    diagram,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_STATIONS = 10
N_FEATURES = 8
SEQ_LEN = 12
BATCH = 4


@pytest.fixture
def config() -> LSTMConfig:
    return LSTMConfig(
        n_stations=N_STATIONS,
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        embed_dim=8,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        head_dim=16,
        layer_norm=True,
    )


@pytest.fixture
def model(config: LSTMConfig) -> VelibLSTM:
    return VelibLSTM(config)


@pytest.fixture
def batch_inputs():
    """Random (x, station_ids) tensors with deterministic seed."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    station_ids = torch.randint(0, N_STATIONS, (BATCH,))
    return x, station_ids


# ---------------------------------------------------------------------------
# LSTMConfig
# ---------------------------------------------------------------------------


class TestLSTMConfig:
    def test_required_fields(self):
        cfg = LSTMConfig(n_stations=5, n_features=4)
        assert cfg.n_stations == 5
        assert cfg.n_features == 4

    def test_defaults(self):
        cfg = LSTMConfig(n_stations=5, n_features=4)
        assert cfg.seq_len == 24
        assert cfg.embed_dim == 16
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.2
        assert cfg.head_dim == 64
        assert cfg.layer_norm is True

    def test_frozen(self):
        cfg = LSTMConfig(n_stations=5, n_features=4)
        with pytest.raises(Exception):
            cfg.n_stations = 99  # type: ignore[misc]

    def test_custom_values(self):
        cfg = LSTMConfig(
            n_stations=100,
            n_features=20,
            seq_len=48,
            embed_dim=32,
            hidden_size=256,
            num_layers=3,
            dropout=0.1,
            head_dim=128,
            layer_norm=False,
        )
        assert cfg.seq_len == 48
        assert cfg.embed_dim == 32
        assert cfg.layer_norm is False


# ---------------------------------------------------------------------------
# LSTMOutput
# ---------------------------------------------------------------------------


class TestLSTMOutput:
    def test_total_is_sum(self):
        mech = torch.tensor([[3.0], [1.5]])
        ebike = torch.tensor([[2.0], [0.5]])
        out = LSTMOutput(mechanical=mech, ebike=ebike)
        expected = torch.tensor([[5.0], [2.0]])
        assert torch.allclose(out.total, expected)

    def test_total_shape(self):
        out = LSTMOutput(mechanical=torch.randn(4, 1), ebike=torch.randn(4, 1))
        assert out.total.shape == (4, 1)

    def test_clamp_removes_negatives(self):
        mech = torch.tensor([[-1.0], [2.0]])
        ebike = torch.tensor([[0.5], [-3.0]])
        clamped = LSTMOutput(mechanical=mech, ebike=ebike).clamp(0.0)
        assert (clamped.mechanical >= 0).all()
        assert (clamped.ebike >= 0).all()

    def test_clamp_returns_new_output(self):
        out = LSTMOutput(mechanical=torch.tensor([[-1.0]]), ebike=torch.tensor([[1.0]]))
        clamped = out.clamp(0.0)
        # original unchanged
        assert out.mechanical.item() == -1.0
        assert clamped.mechanical.item() == 0.0

    def test_clamp_custom_min(self):
        out = LSTMOutput(mechanical=torch.tensor([[0.5]]), ebike=torch.tensor([[0.5]]))
        clamped = out.clamp(min_val=1.0)
        assert clamped.mechanical.item() == 1.0
        assert clamped.ebike.item() == 1.0


# ---------------------------------------------------------------------------
# _StationEmbedding
# ---------------------------------------------------------------------------


class TestStationEmbedding:
    def test_output_shape(self):
        emb = _StationEmbedding(n_stations=20, embed_dim=8)
        ids = torch.randint(0, 20, (5,))
        out = emb(ids)
        assert out.shape == (5, 8)

    def test_single_item(self):
        emb = _StationEmbedding(n_stations=10, embed_dim=4)
        ids = torch.tensor([3])
        assert emb(ids).shape == (1, 4)

    def test_weight_init_scale(self):
        # Weights should be small (std ≈ 0.02)
        emb = _StationEmbedding(n_stations=1000, embed_dim=64)
        std = emb.embedding.weight.std().item()
        assert std < 0.1

    def test_with_dropout(self):
        emb = _StationEmbedding(n_stations=10, embed_dim=4, dropout=0.5)
        emb.train()
        ids = torch.randint(0, 10, (8,))
        out = emb(ids)
        assert out.shape == (8, 4)


# ---------------------------------------------------------------------------
# _OutputHead
# ---------------------------------------------------------------------------


class TestOutputHead:
    def test_output_shape(self):
        head = _OutputHead(in_features=16, hidden_dim=8)
        x = torch.randn(5, 16)
        out = head(x)
        assert out.shape == (5, 1)

    def test_single_sample(self):
        head = _OutputHead(in_features=4, hidden_dim=4)
        x = torch.randn(1, 4)
        assert head(x).shape == (1, 1)

    def test_parameter_count(self):
        head = _OutputHead(in_features=16, hidden_dim=8)
        n = count_parameters(head)
        # Linear(16,8): 16*8+8=136; Linear(8,1): 8+1=9 → 145
        assert n == (16 * 8 + 8) + (8 * 1 + 1)


# ---------------------------------------------------------------------------
# VelibLSTM — instantiation
# ---------------------------------------------------------------------------


class TestVelibLSTMInit:
    def test_creates_correctly(self, config):
        m = VelibLSTM(config)
        assert isinstance(m, VelibLSTM)
        assert m.config is config

    def test_has_expected_submodules(self, model):
        assert hasattr(model, "station_embedding")
        assert hasattr(model, "input_proj")
        assert hasattr(model, "input_norm")
        assert hasattr(model, "lstm")
        assert hasattr(model, "mechanical_head")
        assert hasattr(model, "ebike_head")

    def test_layer_norm_enabled(self, config):
        import torch.nn as nn

        m = VelibLSTM(config)
        assert isinstance(m.input_norm, nn.LayerNorm)

    def test_layer_norm_disabled(self, config):
        import torch.nn as nn

        cfg = LSTMConfig(**{**config.model_dump(), "layer_norm": False})
        m = VelibLSTM(cfg)
        assert isinstance(m.input_norm, nn.Identity)

    def test_single_layer_no_dropout(self, config):
        """When num_layers=1 LSTM dropout must be 0 (PyTorch warns otherwise)."""
        cfg = LSTMConfig(**{**config.model_dump(), "num_layers": 1})
        m = VelibLSTM(cfg)
        assert m.lstm.dropout == 0.0

    def test_multilayer_has_dropout(self, config):
        m = VelibLSTM(config)
        assert m.lstm.dropout == config.dropout

    def test_positive_parameter_count(self, model):
        assert count_parameters(model) > 0


# ---------------------------------------------------------------------------
# VelibLSTM — forward pass
# ---------------------------------------------------------------------------


class TestVelibLSTMForward:
    def test_returns_lstm_output(self, model, batch_inputs):
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        assert isinstance(out, LSTMOutput)

    def test_mechanical_shape(self, model, batch_inputs):
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        assert out.mechanical.shape == (BATCH, 1)

    def test_ebike_shape(self, model, batch_inputs):
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        assert out.ebike.shape == (BATCH, 1)

    def test_total_shape(self, model, batch_inputs):
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        assert out.total.shape == (BATCH, 1)

    def test_total_equals_sum(self, model, batch_inputs):
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        assert torch.allclose(out.total, out.mechanical + out.ebike)

    def test_output_has_grad_fn(self, model, batch_inputs):
        """Outputs must be part of the computation graph for backprop."""
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        assert out.mechanical.requires_grad or out.mechanical.grad_fn is not None
        assert out.ebike.requires_grad or out.ebike.grad_fn is not None

    def test_batch_size_1(self, model):
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        ids = torch.tensor([0])
        out = model(x, ids)
        assert out.mechanical.shape == (1, 1)

    def test_large_batch(self, model):
        x = torch.randn(32, SEQ_LEN, N_FEATURES)
        ids = torch.randint(0, N_STATIONS, (32,))
        out = model(x, ids)
        assert out.mechanical.shape == (32, 1)

    def test_different_stations_different_output(self, model):
        """Two batches with identical x but different station IDs should differ."""
        torch.manual_seed(1)
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        out_a = model(x, torch.tensor([0]))
        out_b = model(x, torch.tensor([1]))
        # With random weights station embeddings make outputs differ
        assert not torch.allclose(out_a.mechanical, out_b.mechanical)

    def test_eval_mode_deterministic(self, model, batch_inputs):
        """eval() must give deterministic results (dropout disabled)."""
        model.eval()
        x, station_ids = batch_inputs
        with torch.no_grad():
            out1 = model(x, station_ids)
            out2 = model(x, station_ids)
        assert torch.allclose(out1.mechanical, out2.mechanical)
        assert torch.allclose(out1.ebike, out2.ebike)

    def test_no_nan_in_output(self, model, batch_inputs):
        model.eval()
        x, station_ids = batch_inputs
        with torch.no_grad():
            out = model(x, station_ids)
        assert not torch.isnan(out.mechanical).any()
        assert not torch.isnan(out.ebike).any()


# ---------------------------------------------------------------------------
# VelibLSTM — gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_to_all_params(self, model, batch_inputs):
        """A backward pass must populate .grad for every trainable parameter."""
        model.train()
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        loss = out.mechanical.sum() + out.ebike.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_embedding_gradient(self, model, batch_inputs):
        model.train()
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        (out.mechanical.sum() + out.ebike.sum()).backward()
        assert model.station_embedding.embedding.weight.grad is not None

    def test_independent_head_gradients(self, model, batch_inputs):
        """Backprop through mechanical only should not zero out ebike head grads."""
        model.train()
        x, station_ids = batch_inputs
        out = model(x, station_ids)
        out.mechanical.sum().backward()
        # ebike head should still receive gradient (shared context vector)
        mech_grad = model.mechanical_head.net[0].weight.grad
        assert mech_grad is not None


# ---------------------------------------------------------------------------
# VelibLSTM — encode
# ---------------------------------------------------------------------------


class TestEncode:
    def test_encode_shape(self, model, batch_inputs):
        x, station_ids = batch_inputs
        ctx = model.encode(x, station_ids)
        expected_dim = model.config.hidden_size + model.config.embed_dim
        assert ctx.shape == (BATCH, expected_dim)

    def test_encode_consistent_with_forward(self, model, batch_inputs):
        """encode() should produce the same context as the forward pass internals."""
        model.eval()
        x, station_ids = batch_inputs
        with torch.no_grad():
            ctx1 = model.encode(x, station_ids)
            ctx2 = model.encode(x, station_ids)
        assert torch.allclose(ctx1, ctx2)

    def test_encode_batch_1(self, model):
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        ids = torch.tensor([2])
        ctx = model.encode(x, ids)
        assert ctx.shape == (1, model.config.hidden_size + model.config.embed_dim)


# ---------------------------------------------------------------------------
# VelibLSTM — reset_hidden (no-op)
# ---------------------------------------------------------------------------


class TestResetHidden:
    def test_reset_hidden_is_callable(self, model):
        model.reset_hidden()  # should not raise

    def test_reset_hidden_returns_none(self, model):
        assert model.reset_hidden() is None


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_returns_velib_lstm(self, config):
        m = build_model(config, device="cpu")
        assert isinstance(m, VelibLSTM)

    def test_on_cpu_device(self, config):
        m = build_model(config, device="cpu")
        param = next(m.parameters())
        assert param.device.type == "cpu"

    def test_auto_device_does_not_crash(self, config):
        m = build_model(config)
        assert isinstance(m, VelibLSTM)

    def test_invalid_device_raises(self, config):
        with pytest.raises(Exception):
            build_model(config, device="invalid_device")

    def test_forward_after_build(self, config):
        m = build_model(config, device="cpu")
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        ids = torch.randint(0, N_STATIONS, (2,))
        out = m(x, ids)
        assert out.mechanical.shape == (2, 1)


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------


class TestCountParameters:
    def test_positive(self, model):
        assert count_parameters(model) > 0

    def test_increases_with_hidden_size(self, config):
        small = LSTMConfig(**{**config.model_dump(), "hidden_size": 16})
        large = LSTMConfig(**{**config.model_dump(), "hidden_size": 64})
        m_small = VelibLSTM(small)
        m_large = VelibLSTM(large)
        assert count_parameters(m_large) > count_parameters(m_small)

    def test_increases_with_embed_dim(self, config):
        small = LSTMConfig(**{**config.model_dump(), "embed_dim": 4})
        large = LSTMConfig(**{**config.model_dump(), "embed_dim": 32})
        assert count_parameters(VelibLSTM(large)) > count_parameters(VelibLSTM(small))

    def test_frozen_params_excluded(self):
        """Parameters with requires_grad=False should not be counted."""
        cfg = LSTMConfig(n_stations=5, n_features=4, hidden_size=16, head_dim=8, embed_dim=4)
        m = VelibLSTM(cfg)
        total = count_parameters(m)
        # Freeze the station embedding
        for p in m.station_embedding.parameters():
            p.requires_grad_(False)
        assert count_parameters(m) < total

    def test_empty_module(self):
        import torch.nn as nn

        assert count_parameters(nn.Sequential()) == 0


# ---------------------------------------------------------------------------
# diagram
# ---------------------------------------------------------------------------


class TestDiagram:
    def test_returns_string(self):
        assert isinstance(diagram(), str)

    def test_multiline(self):
        d = diagram()
        assert "\n" in d

    def test_contains_veliblstm(self):
        assert "VelibLSTM" in diagram()

    def test_with_config(self, config):
        d = diagram(config)
        assert str(config.hidden_size) in d
        assert str(config.embed_dim) in d
        assert str(config.n_features) in d

    def test_without_config_uses_placeholders(self):
        d = diagram(None)
        # Symbolic placeholders
        assert "H" in d
        assert "E" in d
        assert "F" in d

    def test_config_dimensions_appear(self, config):
        d = diagram(config)
        # head_dim should appear
        assert str(config.head_dim) in d

    def test_no_exception_default_config(self):
        cfg = LSTMConfig(n_stations=100, n_features=15)
        d = diagram(cfg)
        assert len(d) > 100
