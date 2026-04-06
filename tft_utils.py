from pathlib import Path

import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def checkpoint_score(checkpoint_path: Path) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    callbacks = checkpoint.get("callbacks", {})
    for callback_state in callbacks.values():
        if isinstance(callback_state, dict) and "best_model_score" in callback_state:
            score = callback_state["best_model_score"]
            return float(score.item() if hasattr(score, "item") else score)
    return float("inf")


def find_best_checkpoint(models_dir: Path) -> Path | None:
    candidates = sorted(models_dir.glob("tft_best*.ckpt"))
    if not candidates:
        return None
    return min(candidates, key=checkpoint_score)


def load_tft_from_checkpoint(training, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    loss = hparams.get("loss")
    quantiles = getattr(loss, "quantiles", [0.1, 0.5, 0.9])

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hparams["hidden_size"],
        attention_head_size=hparams["attention_head_size"],
        lstm_layers=hparams["lstm_layers"],
        hidden_continuous_size=hparams["hidden_continuous_size"],
        dropout=hparams["dropout"],
        loss=QuantileLoss(quantiles=quantiles),
        learning_rate=hparams["learning_rate"],
        optimizer=hparams["optimizer"],
        reduce_on_plateau_patience=hparams["reduce_on_plateau_patience"],
        log_interval=hparams["log_interval"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, quantiles
