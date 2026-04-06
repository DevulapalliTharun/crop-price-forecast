"""
05_generate_tft_predictions.py
Load trained TFT and generate:
  1. Quantile predictions (q10/q50/q90) -> tft_predictions.csv
  2. Attention weights -> tft_attention.csv
  3. Variable importance (encoder + decoder) -> tft_variable_importance.csv
"""

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tft_utils import find_best_checkpoint, load_tft_from_checkpoint

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

print("Loading master dataset...")
df = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])

for col in ["commodity", "market", "admin1", "season"]:
    df[col] = df[col].astype(str)

df_train = df[df["date"] <= "2020-12-31"].copy()

max_encoder_length = 24
max_prediction_length = 6

training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target="log_price",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    min_encoder_length=12,
    max_prediction_length=max_prediction_length,
    min_prediction_length=1,
    static_categoricals=["commodity", "market", "admin1"],
    static_reals=[],
    time_varying_known_categoricals=["season"],
    time_varying_known_reals=[
        "time_idx", "year", "month", "month_sin", "month_cos", "covid_lockdown"
    ],
    time_varying_unknown_reals=[
        "log_price",
        "temperature_mean", "rainfall_monthly", "humidity_mean",
        "price_lag_1m", "price_lag_12m", "rolling_3m", "rolling_6m",
        "yoy_change",
        "rain_deficit", "rain_excess", "heat_stress", "cold_stress",
    ],
    target_normalizer=GroupNormalizer(
        groups=["series_id"],
        transformation="softplus",
    ),
    categorical_encoders={
        "commodity": NaNLabelEncoder(add_nan=True),
        "market": NaNLabelEncoder(add_nan=True),
        "admin1": NaNLabelEncoder(add_nan=True),
        "season": NaNLabelEncoder(add_nan=True),
    },
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

best_ckpt = find_best_checkpoint(MODELS)
if best_ckpt is None:
    print("ERROR: No TFT checkpoint found")
    raise SystemExit(1)

print(f"Loading TFT: {best_ckpt.name}")
best_tft, quantiles = load_tft_from_checkpoint(training, best_ckpt)
q10_idx = quantiles.index(0.1)
q50_idx = quantiles.index(0.5)
q90_idx = quantiles.index(0.9)


def predict_and_interpret(split_df, split_name, predict_mode=True):
    dataset = TimeSeriesDataSet.from_dataset(
        training,
        split_df,
        predict=predict_mode,
        stop_randomization=True,
    )
    dataloader = dataset.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=0,
        shuffle=False,
    )

    print(f"  Predicting on {split_name} ({len(dataset)} windows)...")
    quantile_preds = best_tft.predict(dataloader, mode="quantiles")
    raw = best_tft.predict(dataloader, mode="raw", return_x=True)
    decoded = dataset.decoded_index.reset_index(drop=True)

    pred_records = []
    for i in range(min(len(decoded), quantile_preds.shape[0])):
        row = decoded.iloc[i]
        sid = row["series_id"]
        first_pred_time = row["time_idx_first_prediction"]
        series_df = df[df["series_id"] == sid].sort_values("time_idx")

        for step in range(max_prediction_length):
            target_time = first_pred_time + step
            match = series_df[series_df["time_idx"] == target_time]
            if len(match) == 0:
                continue

            actual = match.iloc[0]
            pred_records.append({
                "series_id": sid,
                "date": actual["date"],
                "commodity": actual["commodity"],
                "market": actual["market"],
                "price": actual["price"],
                "tft_q10": np.expm1(quantile_preds[i, step, q10_idx].item()),
                "tft_q50": np.expm1(quantile_preds[i, step, q50_idx].item()),
                "tft_q90": np.expm1(quantile_preds[i, step, q90_idx].item()),
                "horizon": step + 1,
                "split": split_name,
            })

    attn_records = []
    enc_var_records = []
    dec_var_records = []

    if not predict_mode:
        return (
            pd.DataFrame(pred_records),
            pd.DataFrame(attn_records),
            pd.DataFrame(enc_var_records + dec_var_records),
        )

    try:
        interp = best_tft.interpret_output(raw.output, reduction="sum")
        raw_attn = raw.output[1].detach().cpu().numpy()
        raw_enc_vs = raw.output[4].detach().cpu().numpy()
        raw_dec_vs = raw.output[5].detach().cpu().numpy()

        per_sample_attn = raw_attn.mean(axis=(1, 2))
        per_sample_enc = raw_enc_vs.mean(axis=(1, 2))
        per_sample_dec = raw_dec_vs.mean(axis=(1, 2))

        for i in range(min(len(decoded), per_sample_attn.shape[0])):
            row = decoded.iloc[i]
            sid = row["series_id"]
            commodity = sid.split("_")[0]
            encoder_start = row["time_idx_first"]

            for step in range(per_sample_attn.shape[1]):
                past_time_idx = encoder_start + step
                past_match = df[
                    (df["series_id"] == sid) & (df["time_idx"] == past_time_idx)
                ]
                past_date = past_match["date"].iloc[0] if len(past_match) > 0 else None
                attn_records.append({
                    "series_id": sid,
                    "commodity": commodity,
                    "encoder_step": step,
                    "past_time_idx": past_time_idx,
                    "past_date": past_date,
                    "attention_weight": float(per_sample_attn[i, step]),
                    "split": split_name,
                })

            for j, var_name in enumerate(best_tft.encoder_variables):
                if j >= per_sample_enc.shape[-1]:
                    continue
                enc_var_records.append({
                    "series_id": sid,
                    "commodity": commodity,
                    "variable": var_name,
                    "importance": float(per_sample_enc[i, j]),
                    "type": "encoder",
                    "split": split_name,
                })

            for j, var_name in enumerate(best_tft.decoder_variables):
                if j >= per_sample_dec.shape[-1]:
                    continue
                dec_var_records.append({
                    "series_id": sid,
                    "commodity": commodity,
                    "variable": var_name,
                    "importance": float(per_sample_dec[i, j]),
                    "type": "decoder",
                    "split": split_name,
                })

        print(
            f"    Extracted attention ({len(attn_records)} records), "
            f"encoder vars ({len(enc_var_records)}), decoder vars ({len(dec_var_records)})"
        )
        _ = interp
    except Exception as exc:
        print(f"    WARNING: Could not extract interpretability: {exc}")

    return (
        pd.DataFrame(pred_records),
        pd.DataFrame(attn_records),
        pd.DataFrame(enc_var_records + dec_var_records),
    )


all_preds = []
all_attn = []
all_vars = []

p, a, v = predict_and_interpret(df_train, "train", predict_mode=True)
all_preds.append(p)
all_attn.append(a)
all_vars.append(v)

df_val_with_context = df[
    (df["date"] >= "2020-01-01") & (df["date"] <= "2022-12-31")
].copy()
p, a, v = predict_and_interpret(df_val_with_context, "val", predict_mode=False)
p = p[(p["date"] >= "2021-01-01") & (p["date"] <= "2022-12-31")]
all_preds.append(p)
all_attn.append(a)
all_vars.append(v)

df_test_with_context = df[df["date"] >= "2020-01-01"].copy()
if len(df_test_with_context) > 0:
    p, a, v = predict_and_interpret(df_test_with_context, "test", predict_mode=False)
    p = p[p["date"] >= "2023-01-01"]
    all_preds.append(p)
    all_attn.append(a)
    all_vars.append(v)

result_df = pd.concat(all_preds, ignore_index=True)
result_agg = result_df.groupby(
    ["series_id", "date", "commodity", "market", "price"]
).agg(
    tft_q10=("tft_q10", "mean"),
    tft_q50=("tft_q50", "mean"),
    tft_q90=("tft_q90", "mean"),
).reset_index()
result_agg["date"] = pd.to_datetime(result_agg["date"])
result_agg = result_agg.sort_values(["series_id", "date"]).reset_index(drop=True)
result_agg["band_width"] = result_agg["tft_q90"] - result_agg["tft_q10"]
result_agg.to_csv(PROCESSED / "tft_predictions.csv", index=False)

attn_df = pd.concat(all_attn, ignore_index=True)
if len(attn_df) > 0:
    attn_agg = attn_df.groupby(["commodity", "encoder_step"]).agg(
        attention_weight=("attention_weight", "mean"),
    ).reset_index()
    attn_agg.to_csv(PROCESSED / "tft_attention.csv", index=False)
    attn_df.to_csv(PROCESSED / "tft_attention_detail.csv", index=False)
    print(f"\n  Saved tft_attention.csv ({len(attn_agg)} rows)")
    print(f"  Saved tft_attention_detail.csv ({len(attn_df)} rows)")

vars_df = pd.concat(all_vars, ignore_index=True)
if len(vars_df) > 0:
    vars_agg = vars_df.groupby(["commodity", "variable", "type"]).agg(
        importance=("importance", "mean"),
    ).reset_index()
    vars_agg.to_csv(PROCESSED / "tft_variable_importance.csv", index=False)
    vars_df.to_csv(PROCESSED / "tft_variable_importance_detail.csv", index=False)
    print(f"  Saved tft_variable_importance.csv ({len(vars_agg)} rows)")

print(f"\n{'=' * 60}")
print("OUTPUTS:")
print(f"  {PROCESSED / 'tft_predictions.csv'} ({len(result_agg):,} rows)")
print(f"  {PROCESSED / 'tft_attention.csv'}")
print(f"  {PROCESSED / 'tft_attention_detail.csv'}")
print(f"  {PROCESSED / 'tft_variable_importance.csv'}")

for split_name, split_dates in [
    ("Val (2021-22)", ("2021-01-01", "2022-12-31")),
    ("Test (2023+)", ("2023-01-01", "2030-01-01")),
]:
    mask = (result_agg["date"] >= split_dates[0]) & (result_agg["date"] <= split_dates[1])
    split_pred = result_agg[mask]
    if len(split_pred) == 0:
        continue

    mae = np.mean(np.abs(split_pred["price"] - split_pred["tft_q50"]))
    mape = np.mean(np.abs((split_pred["price"] - split_pred["tft_q50"]) / split_pred["price"])) * 100
    coverage = np.mean(
        (split_pred["price"] >= split_pred["tft_q10"])
        & (split_pred["price"] <= split_pred["tft_q90"])
    ) * 100
    print(f"\n  {split_name}: MAE={mae:.2f} Rs/KG  MAPE={mape:.1f}%  Coverage={coverage:.1f}%")

print(f"{'=' * 60}")
