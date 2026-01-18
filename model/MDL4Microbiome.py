"""
PyTorch two-stage reimplementation of the MDL4Microbiome idea.

Stage-1 (per-modality individual training):
- For each modality, train a small MLP classifier with an explicit encoder
  producing a 50-d embedding. After E1 epochs, use `encode(...)` to extract
  per-sample embeddings (train/val) from the trained individual model.

Stage-2 (shared training on concatenated embeddings):
- Concatenate embeddings from all modalities (each 50-d) to form a vector of
  size 50 * num_modalities. Train a shared classifier head for E2 epochs.

Notes:
- This module only defines models. Data loading, CV splitting, and the
  training loop live in `train.py`.
- Both stages output a single logit [B, 1] and are compatible with
  BCEWithLogitsLoss.
- Individual models expose `encode(x)` to get the 50-d representation after
  training.
"""

from typing import Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


def _mlp_block(in_dim: int, out_dim: int, *, batchnorm: bool, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.ReLU(inplace=True))
    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MDL4MIndividual(nn.Module):
    """Per-modality individual classifier with an explicit 50-d encoder.

    Mirrors the original: 200 -> 100 -> 50 (+ReLU), then a classifier head.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        embed_dim: int = 50,
        hidden: Tuple[int, int] = (200, 100),
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        # encoder to 50-d (with activation)
        self.enc1 = _mlp_block(in_dim, h1, batchnorm=batchnorm, dropout=dropout)
        self.enc2 = _mlp_block(h1, h2, batchnorm=batchnorm, dropout=dropout)
        # final embedding layer keeps activation to mirror Keras Dense(50, relu)
        self.enc3 = _mlp_block(h2, embed_dim, batchnorm=batchnorm, dropout=dropout)
        # classifier head: 50 -> 1 logit
        self.head = nn.Linear(embed_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return 50-d embedding for inputs x: [B, in_dim] -> [B, 50]."""
        z = self.enc1(x)
        z = self.enc2(z)
        z = self.enc3(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        logit = self.head(z)
        return logit

    # Factories to match repo conventions
    @classmethod
    def make_default(
        cls,
        n_num_features: int,
        *,
        embed_dim: int = 50,
        hidden: Tuple[int, int] = (200, 100),
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> "MDL4MIndividual":
        return cls(
            in_dim=n_num_features,
            embed_dim=embed_dim,
            hidden=hidden,
            dropout=dropout,
            batchnorm=batchnorm,
        )

    def make_default_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)


class MDL4MShared(nn.Module):
    """Shared classifier trained on concatenated per-modality embeddings.

    Original Keras shared model used: input_dim -> 50 -> 25 -> 2 softmax.
    Here we use: input_dim -> 50 -> 25 -> 1 (logit) to pair with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        proj_dim: int = 50,
        hidden: int = 25,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.proj = _mlp_block(input_dim, proj_dim, batchnorm=batchnorm, dropout=dropout)
        self.mid = _mlp_block(proj_dim, hidden, batchnorm=batchnorm, dropout=dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.mid(z)
        logit = self.out(z)
        return logit

    # Factories to match repo conventions
    @classmethod
    def make_default(
        cls,
        concat_dim: int,
        *,
        proj_dim: int = 50,
        hidden: int = 25,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> "MDL4MShared":
        return cls(
            input_dim=concat_dim,
            proj_dim=proj_dim,
            hidden=hidden,
            dropout=dropout,
            batchnorm=batchnorm,
        )

    def make_default_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

"""
PyTorch reimplementation of the MDL4Microbiome idea.

Core idea (aligned with the original script):
- Per-modality MLP encoder that produces a compact 50-d representation
  ("individual features").
- A shared head that takes the concatenated representations from all
  available modalities and outputs a single logit for binary classification.

Notes:
- This module focuses purely on the model definition. Data loading,
  standardization, cross-validation, and training loops live in train.py.
- Output is a raw logit (shape [B, 1]) compatible with BCEWithLogitsLoss.
- Supports unimodal (single tensor) and multimodal (dict[str, tensor]) inputs.

Typical usage (multimodal):
    model = MDL4Microbiome.make_default(
        n_species_features=species_dim,
        n_ko_features=ko_dim,
        embed_dim=50,
        encoder_hidden=(200, 100),
        head_hidden=25,
        dropout=0.0,
        batchnorm=False,
    )

Forward expects:
- Unimodal: x: torch.FloatTensor [B, D]
- Multimodal: raw_x: dict with keys like {'species': x1, 'ko': x2}

This keeps the spirit of the original MDL4Microbiome pipeline, while being
end-to-end trainable within a standard 5-fold CV loop in train.py.
"""

from typing import Dict, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """A simple MLP encoder producing a fixed-size embedding.

    Example architecture to mirror the original Keras network for individuals:
    - hidden: 200 -> 100 -> embed_dim (default 50), with ReLU activations.
    Optionally includes BatchNorm and Dropout.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, int] = (200, 100),
        embed_dim: int = 50,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        last = in_dim
        for i, h in enumerate(hidden):
            layers[f"lin{i}"] = nn.Linear(last, h)
            if batchnorm:
                layers[f"bn{i}"] = nn.BatchNorm1d(h)
            layers[f"act{i}"] = nn.ReLU(inplace=True)
            if dropout and dropout > 0:
                layers[f"drop{i}"] = nn.Dropout(dropout)
            last = h

        # projection to embedding
        layers["lin_out"] = nn.Linear(last, embed_dim)
        # keep final embedding linear; optional activation could be added

        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> [B, embed_dim]
        return self.net(x)


class SharedHead(nn.Module):
    """Shared classifier head on top of concatenated embeddings.

    Mirrors original shared model: 50 -> 25 -> 2 softmax, but here we output
    a single logit for BCEWithLogitsLoss. So we use: concat_dim -> head_hidden -> 1.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 25,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers["lin0"] = nn.Linear(in_dim, hidden)
        if batchnorm:
            layers["bn0"] = nn.BatchNorm1d(hidden)
        layers["act0"] = nn.ReLU(inplace=True)
        if dropout and dropout > 0:
            layers["drop0"] = nn.Dropout(dropout)
        layers["out"] = nn.Linear(hidden, 1)  # single logit

        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> [B, 1]
        return self.net(x)


class MDL4Microbiome(nn.Module):
    """Multi-branch MLP for microbiome modalities with a shared classifier head.

    - If constructed with multiple input dims (e.g., species + ko), each
      modality has its own encoder producing a 50-d embedding.
    - Embeddings are concatenated and passed to a shared head generating one logit.
    - If constructed with a single input dim, behaves as a standalone encoder+head.
    """

    def __init__(
        self,
        input_dims: Dict[str, int] | int,
        *,
        embed_dim: int = 50,
        encoder_hidden: Tuple[int, int] = (200, 100),
        head_hidden: int = 25,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.batchnorm = batchnorm
        self.dropout = dropout

        if isinstance(input_dims, int):
            # Unimodal
            self.single_encoder = True
            self.encoder = MLPEncoder(
                in_dim=input_dims,
                hidden=encoder_hidden,
                embed_dim=embed_dim,
                dropout=dropout,
                batchnorm=batchnorm,
            )
            concat_dim = embed_dim
        else:
            # Multimodal
            self.single_encoder = False
            self.encoders = nn.ModuleDict()
            for name, dim in input_dims.items():
                self.encoders[name] = MLPEncoder(
                    in_dim=dim,
                    hidden=encoder_hidden,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    batchnorm=batchnorm,
                )
            concat_dim = embed_dim * len(input_dims)

        self.head = SharedHead(
            in_dim=concat_dim,
            hidden=head_hidden,
            dropout=dropout,
            batchnorm=batchnorm,
        )

    def forward(self, x):
        """Forward pass.

        - Unimodal: x is a FloatTensor [B, D]
        - Multimodal: x is a dict[str, FloatTensor], each [B, D_mod]
        Returns: logits [B, 1]
        """
        if self.single_encoder:
            z = self.encoder(x)
            return self.head(z)

        # multimodal
        zs = []
        for name, encoder in self.encoders.items():
            if name not in x:
                raise KeyError(f"Missing modality '{name}' in input dict. Got keys: {list(x.keys())}")
            zs.append(encoder(x[name]))
        z = torch.cat(zs, dim=1)
        return self.head(z)

    # -------- Convenience factories to mirror other models' pattern -------- #
    @classmethod
    def make_default(
        cls,
        n_species_features: Optional[int] = None,
        n_ko_features: Optional[int] = None,
        n_num_features: Optional[int] = None,
        *,
        embed_dim: int = 50,
        encoder_hidden: Tuple[int, int] = (200, 100),
        head_hidden: int = 25,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> "MDL4Microbiome":
        """Create a model for uni- or multi-modal inputs.

        - If both n_species_features and n_ko_features are provided, build a
          multimodal model expecting a dict input with keys 'species' and 'ko'.
        - Else if n_num_features is provided, build a unimodal model expecting
          a single tensor input.
        """
        if n_species_features is not None and n_ko_features is not None:
            input_dims = {"species": n_species_features, "ko": n_ko_features}
        elif n_num_features is not None:
            input_dims = n_num_features
        else:
            raise ValueError(
                "Provide either (n_species_features and n_ko_features) for multimodal "
                "or n_num_features for unimodal."
            )

        return cls(
            input_dims=input_dims,
            embed_dim=embed_dim,
            encoder_hidden=encoder_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
            batchnorm=batchnorm,
        )

    def make_default_optimizer(self):
        """Provide a default optimizer, so train.py can override lr later.

        Matches the pattern used elsewhere in this repo.
        """
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
