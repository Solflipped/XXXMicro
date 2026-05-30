import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetBinaryClassifier
from model.FT_transformer import NumericalFeatureTokenizer


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class KOFFN(nn.Module):
    def __init__(
        self,
        d_token: int,
        hidden_factor: float = 4.0 / 3.0,
        dropout: float = 0.1,
        activation: str = "ReGLU",
    ):
        super().__init__()
        activation = activation.lower()
        hidden_dim = max(int(d_token * hidden_factor), 4)

        if activation in {"reglu", "geglu"}:
            first_out_dim = hidden_dim * 2
        else:
            first_out_dim = hidden_dim

        self.linear1 = nn.Linear(d_token, first_out_dim)
        self.linear2 = nn.Linear(hidden_dim, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation_name = activation
        self.reglu = ReGLU()

        if activation not in {"reglu", "geglu", "gelu"}:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.activation_name == "reglu":
            x = self.reglu(x)
        elif self.activation_name == "geglu":
            a, b = x.chunk(2, dim=-1)
            x = a * F.gelu(b)
        else:
            x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class KOFeatureGate(nn.Module):
    """
    Feature-wise gate for KO input.
    One learnable scalar per feature; sigmoid keeps gate in (0, 1).
    """

    def __init__(self, n_features: int, init_value: float = 1.5):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_features,), float(init_value)))

    def forward(self, x: torch.Tensor):
        gate = torch.sigmoid(self.logits)  # [N]
        return x * gate.unsqueeze(0), gate


class KOAttentionBlock(nn.Module):
    """
    Balanced transformer block for KO:
    keep moderate token interaction (self-attn) + strong FFN, avoid over-complex stacks.
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        ffn_hidden_factor: float = 4.0 / 3.0,
        ffn_activation: str = "ReGLU",
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_residual_dropout = nn.Dropout(residual_dropout)

        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = KOFFN(
            d_token=d_token,
            hidden_factor=ffn_hidden_factor,
            dropout=ffn_dropout,
            activation=ffn_activation,
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.attn_residual_dropout(attn_out)
        x = x + self.ffn_residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class KOLatentReadout(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_latent_tokens: int,
        n_heads: int,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        n_self_layers: int = 1,
        ffn_hidden_factor: float = 4.0 / 3.0,
        ffn_activation: str = "ReGLU",
    ):
        super().__init__()
        self.n_latent_tokens = n_latent_tokens
        self.latent_tokens = nn.Parameter(torch.randn(n_latent_tokens, d_token) * 0.02)

        self.query_norm = nn.LayerNorm(d_token)
        self.kv_norm = nn.LayerNorm(d_token)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.cross_residual_dropout = nn.Dropout(residual_dropout)

        self.cross_ffn_norm = nn.LayerNorm(d_token)
        self.cross_ffn = KOFFN(
            d_token=d_token,
            hidden_factor=ffn_hidden_factor,
            dropout=ffn_dropout,
            activation=ffn_activation,
        )
        self.cross_ffn_residual_dropout = nn.Dropout(residual_dropout)

        self.self_blocks = nn.ModuleList(
            [
                KOAttentionBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                    ffn_hidden_factor=ffn_hidden_factor,
                    ffn_activation=ffn_activation,
                )
                for _ in range(n_self_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_token)

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = sequence_tokens.shape[0]
        latent = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        q = self.query_norm(latent)
        kv = self.kv_norm(sequence_tokens)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latent = latent + self.cross_residual_dropout(cross_out)
        latent = latent + self.cross_ffn_residual_dropout(self.cross_ffn(self.cross_ffn_norm(latent)))

        for block in self.self_blocks:
            latent = block(latent)

        return self.output_norm(latent)


class KOFT(nn.Module):
    """
    KO-focused latent-token model.

    Pipeline:
    1) NumericalFeatureTokenizer: [B, N] -> [B, N, d]
    2) KO feature mixer with moderate self-attention
    3) Latent readout with n_latent_tokens cross-attention
    4) Latent pooling + binary classification
    5) Auxiliary reconstruction head for stability on small-sample/high-dim KO

    Returns:
        logits, enhanced_tokens, latent_tokens, recon_x
    """

    def __init__(
        self,
        n_num_features: int,
        d_token: int = 32,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = None,
        ffn_dropout: float = None,
        residual_dropout: float = 0.0,
        n_latent_tokens: int = 8,
        n_latent_layers: int = 1,
        ffn_hidden_factor: float = 4.0 / 3.0,
        ffn_activation: str = "ReGLU",
    ):
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")

        if attention_dropout is None:
            attention_dropout = dropout
        if ffn_dropout is None:
            ffn_dropout = dropout

        self.n_num_features = n_num_features
        self.d_token = d_token
        self.n_latent_tokens = n_latent_tokens

        self.feature_gate = KOFeatureGate(n_num_features)

        self.tokenizer = NumericalFeatureTokenizer(
            n_features=n_num_features,
            d_token=d_token,
            bias=True,
            initialization="uniform",
        )

        self.encoder = nn.ModuleList(
            [
                KOAttentionBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                    ffn_hidden_factor=ffn_hidden_factor,
                    ffn_activation=ffn_activation,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_token)

        self.latent_readout = KOLatentReadout(
            d_token=d_token,
            n_latent_tokens=n_latent_tokens,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            n_self_layers=n_latent_layers,
            ffn_hidden_factor=ffn_hidden_factor,
            ffn_activation=ffn_activation,
        )

        self.latent_pool = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Reconstruct each raw feature scalar from its enhanced token.
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def encode_tokens(self, raw_x: torch.Tensor) -> torch.Tensor:
        gated_x, _ = self.feature_gate(raw_x)
        tokens = self.tokenizer(gated_x)
        for block in self.encoder:
            tokens = block(tokens)
        return self.output_norm(tokens)

    def forward(self, raw_x: torch.Tensor):
        gated_x, gate_values = self.feature_gate(raw_x)
        tokens = self.tokenizer(gated_x)
        for block in self.encoder:
            tokens = block(tokens)
        enhanced_tokens = self.output_norm(tokens)

        latent_tokens = self.latent_readout(enhanced_tokens)
        pool_weights = torch.softmax(self.latent_pool(latent_tokens), dim=1)
        latent_repr = (pool_weights * latent_tokens).sum(dim=1)
        logits = self.classifier(latent_repr)

        recon_x = self.reconstruction_head(enhanced_tokens).squeeze(-1)
        return logits, enhanced_tokens, latent_tokens, recon_x, gate_values

    @classmethod
    def make_default(
        cls,
        n_num_features: int,
        d_token: int = 32,
        **kwargs,
    ) -> "KOFT":
        n_heads = kwargs.pop("n_heads", 4 if d_token % 4 == 0 else 2 if d_token % 2 == 0 else 1)
        return cls(
            n_num_features=n_num_features,
            d_token=d_token,
            n_layers=kwargs.pop("n_layers", 1),
            n_heads=n_heads,
            dropout=kwargs.pop("dropout", 0.1),
            attention_dropout=kwargs.pop("attention_dropout", None),
            ffn_dropout=kwargs.pop("ffn_dropout", None),
            residual_dropout=kwargs.pop("residual_dropout", 0.0),
            n_latent_tokens=kwargs.pop("n_latent_tokens", 8),
            n_latent_layers=kwargs.pop("n_latent_layers", 1),
            ffn_hidden_factor=kwargs.pop("ffn_hidden_factor", 4.0 / 3.0),
            ffn_activation=kwargs.pop("ffn_activation", "ReGLU"),
            **kwargs,
        )


class KOFTNet(NeuralNetBinaryClassifier):
    def __init__(
        self,
        *args,
        lambda_recon: float = 0.0,
        recon_warmup_epochs: int = 20,
        lambda_gate_l1: float = 1e-4,
        gate_warmup_epochs: int = 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_recon = float(lambda_recon)
        self.recon_warmup_epochs = int(recon_warmup_epochs)
        self.lambda_gate_l1 = float(lambda_gate_l1)
        self.gate_warmup_epochs = int(gate_warmup_epochs)

    def _current_recon_weight(self) -> float:
        if self.lambda_recon <= 0:
            return 0.0
        if self.recon_warmup_epochs <= 0:
            return self.lambda_recon
        epoch = len(self.history) if hasattr(self, "history") else 0
        scale = min(1.0, float(epoch + 1) / float(self.recon_warmup_epochs))
        return self.lambda_recon * scale

    def _current_gate_weight(self) -> float:
        if self.lambda_gate_l1 <= 0:
            return 0.0
        if self.gate_warmup_epochs <= 0:
            return self.lambda_gate_l1
        epoch = len(self.history) if hasattr(self, "history") else 0
        scale = min(1.0, float(epoch + 1) / float(self.gate_warmup_epochs))
        return self.lambda_gate_l1 * scale

    @staticmethod
    def _extract_input_tensor(X, ref_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(X, dict):
            if "f1_input" in X:
                X = X["f1_input"]
            else:
                X = next(iter(X.values()))

        if isinstance(X, (list, tuple)) and len(X) > 0:
            X = X[0]

        if isinstance(X, torch.Tensor):
            xt = X
        else:
            xt = torch.as_tensor(X)

        return xt.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def get_loss(self, y_pred, y_true, X=None, *args, **kwargs):
        if isinstance(y_pred, tuple):
            logits = y_pred[0]
            recon_x = y_pred[3] if len(y_pred) > 3 else None
            gate_values = y_pred[4] if len(y_pred) > 4 else None
        else:
            logits = y_pred
            recon_x = None
            gate_values = None

        y_true = y_true.float().view(-1)
        loss_bce = super().get_loss(logits, y_true, X=X, *args, **kwargs)
        total_loss = loss_bce

        if recon_x is not None and X is not None and self.lambda_recon > 0:
            x_target = self._extract_input_tensor(X, recon_x)

            if x_target.ndim > 2:
                x_target = x_target.view(x_target.shape[0], -1)

            if x_target.shape == recon_x.shape:
                loss_recon = F.mse_loss(recon_x, x_target)
                total_loss = total_loss + self._current_recon_weight() * loss_recon
            elif x_target.numel() == recon_x.numel():
                loss_recon = F.mse_loss(recon_x, x_target.view_as(recon_x))
                total_loss = total_loss + self._current_recon_weight() * loss_recon

        if gate_values is not None and self.lambda_gate_l1 > 0:
            gate_l1 = gate_values.abs().mean()
            total_loss = total_loss + self._current_gate_weight() * gate_l1

        return total_loss

    def predict_proba(self, X):
        non_probas = []
        for yp in self.forward_iter(X, training=False):
            logits = yp[0] if isinstance(yp, tuple) else yp.get("y_pred", list(yp.values())[0])
            p1 = torch.sigmoid(logits).view(-1, 1)
            non_probas.append(p1)

        p1_all = torch.cat(non_probas, dim=0).cpu().numpy()
        p0_all = 1 - p1_all
        return np.hstack([p0_all, p1_all])

    def evaluation_step(self, batch, training=False):
        X, y = batch
        with torch.set_grad_enabled(training):
            yp = self.infer(X)
            loss = self.get_loss(yp, y, X=X)
            logits = yp[0] if isinstance(yp, tuple) else yp
            return {"loss": loss, "y_pred": logits}
