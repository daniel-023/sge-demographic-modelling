"""Model architectures for demographic prediction."""

from .lstm import DemographicLSTM
from .mlp import DemographicMLP


def build_model(model_type: str, model_kwargs: dict):
    if model_type == "mlp":
        return DemographicMLP(**model_kwargs)
    if model_type == "lstm":
        return DemographicLSTM(**model_kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")


__all__ = ["DemographicMLP", "DemographicLSTM", "build_model"]
