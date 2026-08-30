"""Neural Additive Model (Agarwal et al., 2021), reimplemented in PyTorch.

The reference release is TensorFlow 1, so this is a port; Phase 3.3 gates the sweep on
reproducing the published Adult AUROC before any real run happens.

Architecture: one small MLP ("feature net") per input column, summed with a global
bias. The first layer optionally uses ExU units -- h(x) = f((x - b) * exp(w)) with a
capped ReLU -- which let a single unit fit the sharp jumps that tabular features often
have. Regularisation follows the paper: dropout inside each feature net, dropout over
whole feature contributions, and an output penalty on the mean squared per-feature
output.

The ``encoded`` view one-hot expands categoricals, so each level gets its own feature
net. The model stays additive and every level keeps its own readable contribution.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import SOFT_ARMS
from .base import val_objective


class ExU(nn.Module):
    """Exp-centred linear unit followed by ReLU-n."""

    def __init__(self, n_units: int, relu_n: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_units))
        self.bias = nn.Parameter(torch.empty(n_units))
        self.relu_n = relu_n
        # The paper initialises the log-weights in a tight normal around 4.0; drawing
        # them near zero makes exp(w) ~ 1 and loses the sharp-jump capacity entirely.
        nn.init.normal_(self.weight, mean=4.0, std=0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x - self.bias) * torch.exp(self.weight)
        return torch.clamp(h, min=0.0, max=self.relu_n)


class FeatureNet(nn.Module):
    def __init__(
        self, num_basis: int, hidden_sizes: list[int], dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.first = ExU(num_basis) if activation == "exu" else None
        layers: list[nn.Module] = []
        in_dim = num_basis
        if self.first is None:
            layers += [nn.Linear(1, num_basis), nn.ReLU()]
        for width in hidden_sizes:
            layers += [nn.Dropout(dropout), nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        layers += [nn.Dropout(dropout), nn.Linear(in_dim, 1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1)
        h = self.first(x.expand(-1, self.first.weight.numel())) if self.first is not None else x
        return self.body(h).squeeze(-1)


class NAMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_basis: int,
        hidden_sizes: list[int],
        dropout: float,
        feature_dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.nets = nn.ModuleList(
            FeatureNet(num_basis, hidden_sizes, dropout, activation) for _ in range(n_features)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.feature_dropout = nn.Dropout(feature_dropout)

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_features) per-feature shape-function outputs -- the explanation."""
        return torch.stack([net(x[:, [i]]) for i, net in enumerate(self.nets)], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.contributions(x)
        logits = self.feature_dropout(parts).sum(dim=1) + self.bias
        return logits, parts


class NAMModel:
    def __init__(self, seed: int, **params: object) -> None:
        self.seed = seed
        p = dict(params)
        self.hidden_sizes = list(p.get("hidden_sizes", [64, 32]))
        self.num_basis = int(p.get("num_basis_functions", 64))
        self.activation = str(p.get("activation", "exu"))
        self.dropout = float(p.get("dropout", 0.1))
        self.feature_dropout = float(p.get("feature_dropout", 0.05))
        self.output_penalty = float(p.get("output_penalty", 1e-4))
        self.l2 = float(p.get("l2", 0.0))
        self.lr = float(p.get("lr", 1e-3))
        self.batch_size = int(p.get("batch_size", 1024))
        self.max_epochs = int(p.get("max_epochs", 200))
        self.patience = int(p.get("patience", 15))
        self.device = torch.device(p.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net: NAMNet | None = None
        self.columns: list[str] = []
        self.history: list[dict[str, float]] = []

    def _tensor(self, x: pd.DataFrame) -> torch.Tensor:
        return torch.as_tensor(x.to_numpy(dtype=np.float32), device=self.device)

    def fit(self, x_train, t_train, x_val, t_val, *, arm: str) -> "NAMModel":
        torch.manual_seed(self.seed)
        self.columns = list(x_train.columns)

        xt = self._tensor(x_train)
        # BCEWithLogitsLoss takes soft targets in [0, 1] directly, so the distilled and
        # hard arms differ only in what is passed here -- never in the loss itself.
        tt = torch.as_tensor(np.asarray(t_train, dtype=np.float32), device=self.device)
        xv = self._tensor(x_val)

        self.net = NAMNet(
            xt.shape[1], self.num_basis, self.hidden_sizes, self.dropout,
            self.feature_dropout, self.activation,
        ).to(self.device)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.l2)
        criterion = nn.BCEWithLogitsLoss()

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        n = xt.shape[0]
        best_score, best_state, stale = float("inf"), None, 0

        for epoch in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(n, generator=generator).to(self.device)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                logits, parts = self.net(xt[batch])
                loss = criterion(logits, tt[batch])
                if self.output_penalty > 0:
                    loss = loss + self.output_penalty * parts.pow(2).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.item()) * batch.numel()

            val_pred = self._infer(xv)
            score = val_objective(arm, np.asarray(t_val), val_pred)
            self.history.append({"epoch": epoch, "train_loss": epoch_loss / n, "val_objective": score})

            if score < best_score - 1e-6:
                best_score, stale = score, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_objective = best_score
        return self

    @torch.no_grad()
    def _infer(self, x: torch.Tensor) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        out = []
        for start in range(0, x.shape[0], 8192):
            logits, _ = self.net(x[start : start + 8192])
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self._infer(self._tensor(x))

    @torch.no_grad()
    def feature_importances(self) -> dict[str, float]:
        """Mean absolute contribution per feature, the NAM paper's global importance."""
        raise NotImplementedError(
            "Call feature_importances_on(x) -- a NAM's importances are data-dependent, "
            "unlike an EBM's, so they must be computed on a reference sample."
        )

    @torch.no_grad()
    def feature_importances_on(self, x: pd.DataFrame) -> dict[str, float]:
        assert self.net is not None
        self.net.eval()
        parts = self.net.contributions(self._tensor(x)).abs().mean(dim=0).cpu().numpy()
        return {name: float(value) for name, value in zip(self.columns, parts)}
