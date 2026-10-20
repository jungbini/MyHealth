"""LightGBM만으로 시간대별 조도(lux) 예측 모델을 학습·저장·예측·평가한다.

forecast.py의 비교 실험과 독립적으로 쓸 수 있는 단일 파일이다. 입력 피처는 forecast.py의
기본 LightGBM과 같다(예측 대상 날짜 D, 시각 h).

  lag_1 ~ lag_7         D-1 ~ D-7일의 h시 조도 (없으면 NaN)
  hist_mean, hist_std   D-1일까지 전체 과거의 h시 조도 평균·표준편차
  seg_morning/afternoon/evening
                        최근 관측일(D-1 우선, 없으면 7일 이내 가장 최근)의
                        06~12시 / 12~18시 / 18~24시 평균 조도
  hour, weekday         예측 대상의 시각·요일
조도는 10분(또는 5분) 측정치의 log(lux+1)을 시간별로 평균한 값이다.

사용법:
  # 학습: 마지막 관측일까지 전부 학습해 모델 저장
  python light_forecast/lightgbm_model.py train --data lux.csv --model model.txt
  python light_forecast/lightgbm_model.py train --data light_all.xlsx --sheet 116 --model model_116.txt

  # 예측: 저장된 모델로 특정 날짜(기본: 마지막 관측일 다음 날)의 0~23시 조도 예측
  python light_forecast/lightgbm_model.py predict --data lux.csv --model model.txt [--date 2026-09-21]

  # 평가: 첫 4주 이후 일 단위 rolling-origin 검증 (매일 그 전날까지로 다시 학습)
  python light_forecast/lightgbm_model.py evaluate --data lux.csv [--out eval.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HOURS = list(range(24))
N_LAGS = 7
INITIAL_TRAIN_DAYS = 28  # 평가 시 초기 학습 구간(첫 4주)
MIN_TRAIN_DAYS = 5  # 학습일이 이보다 적으면 학습하지 않는다
ONE_DAY = pd.Timedelta(days=1)
SEGMENTS = {"morning": range(6, 12), "afternoon": range(12, 18), "evening": range(18, 24)}
MIN_SEGMENT_HOURS = 3

FEATURES = (
    [f"lag_{k}" for k in range(1, N_LAGS + 1)]
    + ["hist_mean", "hist_std"]
    + [f"seg_{name}" for name in SEGMENTS]
    + ["hour", "weekday"]
)
PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=1,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def load_readings(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    """CSV 또는 엑셀 시트를 (timestamp, lux)로 읽는다.

    'timestamp' 열이 있으면 그 열과 'value'(또는 'light_value') 열을, 없으면 첫 두 열을 쓴다.
    같은 timestamp의 중복 행은 평균으로 합친다.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    else:
        raw = pd.read_csv(path)
    return readings_from_frame(raw)


def readings_from_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """이미 읽은 표(CSV 또는 엑셀 시트)를 (timestamp, lux)로 바꾼다. 규칙은 load_readings와 같다."""
    if "timestamp" in raw.columns:
        value_col = next((c for c in ("value", "light_value", "lux") if c in raw.columns), None)
        ts_col = "timestamp"
        if value_col is None:
            value_col = [c for c in raw.columns if c != ts_col][-1]
    else:
        ts_col, value_col = raw.columns[:2]
    readings = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[ts_col], errors="coerce"),
            "lux": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    ).dropna()
    return readings.groupby("timestamp", as_index=False)["lux"].mean()


def hourly_matrix(readings: pd.DataFrame) -> pd.DataFrame:
    """(연속 달력 날짜 x 0~23시) 행렬. 값은 그 시간 측정치들의 log(lux+1) 평균, 관측 없으면 NaN."""
    ts = readings["timestamp"]
    value = (
        readings.assign(log_lux=np.log1p(readings["lux"].clip(lower=0)), date=ts.dt.normalize(), hour=ts.dt.hour)
        .groupby(["date", "hour"])["log_lux"]
        .mean()
        .unstack("hour")
    )
    days = pd.date_range(value.index.min(), value.index.max(), freq="D", name="date")
    return value.reindex(index=days, columns=HOURS)


def build_features(visible: pd.DataFrame) -> pd.DataFrame:
    """모든 (날짜 d, 시각 h)의 입력 X. 각 행은 d보다 앞선 날짜의 값만으로 계산된다."""
    per_hour = {f"lag_{k}": visible.shift(k) for k in range(1, N_LAGS + 1)}
    per_hour["hist_mean"] = visible.expanding(min_periods=1).mean().shift(1)
    per_hour["hist_std"] = visible.expanding(min_periods=2).std().shift(1)
    features = {name: frame.to_numpy().ravel() for name, frame in per_hour.items()}
    for name, hours in SEGMENTS.items():
        block = visible[list(hours)]
        day_mean = block.mean(axis=1).where(block.notna().sum(axis=1) >= MIN_SEGMENT_HOURS)
        recent = day_mean.shift(1).ffill(limit=N_LAGS - 1)
        features[f"seg_{name}"] = np.repeat(recent.to_numpy(), len(HOURS))
    index = pd.MultiIndex.from_product([visible.index, HOURS], names=["date", "hour"])
    X = pd.DataFrame(features, index=index).reset_index()
    X["weekday"] = X["date"].dt.weekday
    return X


def _visible_until(hourly: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """target 전날까지만 남기고 target 당일 행은 비워 둔다 (예측 시점에 볼 수 있는 데이터)."""
    days = pd.date_range(hourly.index[0], target, freq="D", name="date")
    return hourly.loc[: target - ONE_DAY].reindex(days)


# ---------------------------------------------------------------------------
# 학습·예측
# ---------------------------------------------------------------------------
def training_set(hourly: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    """target 전날까지의 학습 사례 (lag 7개가 생기는 8일째부터, 실측이 있는 시각만)."""
    visible = _visible_until(hourly, target)
    X = build_features(visible)
    y = visible.to_numpy().ravel()
    keep = ((X["date"] >= hourly.index[0] + N_LAGS * ONE_DAY) & (X["date"] < target) & ~np.isnan(y)).to_numpy()
    return X[keep], y[keep]


def train(hourly: pd.DataFrame, target: pd.Timestamp) -> lgb.LGBMRegressor | None:
    """target 전날까지의 데이터로 학습한다. 학습일이 MIN_TRAIN_DAYS 미만이면 None."""
    X, y = training_set(hourly, target)
    if X["date"].nunique() < MIN_TRAIN_DAYS:
        return None
    return lgb.LGBMRegressor(**PARAMS).fit(X[FEATURES], y)


def predict_day(model, hourly: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """target 날짜 0~23시 예측 (log(lux+1)과 lux). model은 LGBMRegressor 또는 Booster."""
    X = build_features(_visible_until(hourly, target))
    X = X[X["date"] == target]
    log_pred = np.clip(model.predict(X[FEATURES]), 0, None)
    return pd.DataFrame(
        {"date": target.date(), "hour": HOURS, "pred_log_lux": log_pred, "pred_lux": np.expm1(log_pred)}
    )


def save_model(model: lgb.LGBMRegressor, path: Path, meta: dict) -> None:
    model.booster_.save_model(str(path))
    path.with_suffix(".json").write_text(json.dumps({**meta, "features": FEATURES, "params": PARAMS}, indent=2))


def load_model(path: Path) -> lgb.Booster:
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["features"] != FEATURES:
        raise ValueError("저장된 모델의 피처 목록이 현재 코드와 다릅니다.")
    return lgb.Booster(model_file=str(path))


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate(hourly: pd.DataFrame, initial_days: int = INITIAL_TRAIN_DAYS) -> pd.DataFrame:
    """첫 initial_days일 이후 하루씩 전진: 전날까지로 학습 → 예측 고정 → 실측과 비교.

    실측이 없는 날은 채점할 수 없으므로 건너뛴다.
    """
    rows = []
    for target in pd.date_range(hourly.index[0] + initial_days * ONE_DAY, hourly.index[-1], freq="D"):
        actual = hourly.loc[target]
        if actual.isna().all():
            continue
        model = train(hourly, target)
        if model is None:
            continue
        pred = predict_day(model, hourly, target)
        pred["actual_log_lux"] = actual.to_numpy()
        rows.append(pred)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metrics(pred: pd.DataFrame) -> dict:
    scored = pred.dropna(subset=["actual_log_lux"])
    err = scored["pred_log_lux"] - scored["actual_log_lux"]
    return {
        "days": scored["date"].nunique(),
        "hours": len(scored),
        "MAE": err.abs().mean(),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAE_lux": (scored["pred_lux"] - np.expm1(scored["actual_log_lux"])).abs().mean(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 시간대별 조도 예측")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="CSV(timestamp, 조도) 또는 엑셀")
        p.add_argument("--sheet", help="엑셀 시트 이름 (사용자 ID)")
    sub.choices["train"].add_argument("--model", required=True, help="저장할 모델 파일 (.txt)")
    sub.choices["predict"].add_argument("--model", required=True, help="저장된 모델 파일 (.txt)")
    sub.choices["predict"].add_argument("--date", help="예측 날짜 (기본: 마지막 관측일 다음 날)")
    sub.choices["predict"].add_argument("--out", help="예측 결과 CSV")
    sub.choices["evaluate"].add_argument("--out", help="날짜·시각별 예측과 실측 CSV")
    args = parser.parse_args()

    hourly = hourly_matrix(load_readings(args.data, args.sheet))
    next_day = hourly.index[-1] + ONE_DAY

    if args.command == "train":
        model = train(hourly, next_day)
        if model is None:
            raise SystemExit(f"학습일이 {MIN_TRAIN_DAYS}일 미만이라 학습할 수 없습니다.")
        n = len(training_set(hourly, next_day)[1])
        save_model(model, Path(args.model), {"data_end": str(hourly.index[-1].date()), "train_rows": n})
        print(f"학습 사례 {n}개 (~{hourly.index[-1].date()})로 학습해 {args.model}에 저장했습니다.")

    elif args.command == "predict":
        target = pd.Timestamp(args.date) if args.date else next_day
        if target > next_day:
            hourly = hourly.reindex(pd.date_range(hourly.index[0], target - ONE_DAY, freq="D", name="date"))
        pred = predict_day(load_model(Path(args.model)), hourly, target)
        if args.out:
            pred.to_csv(args.out, index=False)
        print(pred.round(2).to_string(index=False))

    else:
        pred = evaluate(hourly)
        if pred.empty:
            raise SystemExit("평가할 수 있는 날이 없습니다 (첫 4주 이후 관측 또는 학습일 부족).")
        if args.out:
            pred.to_csv(args.out, index=False)
        print(", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics(pred).items()))


if __name__ == "__main__":
    main()
