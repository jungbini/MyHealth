"""여러 사용자(엑셀 시트별)에 forecast.py의 모델과 rolling-origin 평가를 적용한다.

엑셀의 각 시트는 한 사용자이며 timestamp, value(조도) 열을 가진다.
시간값은 log(lux+1)의 시간 평균을 쓰고, 사용자마다 독립적으로 학습·평가한다.

사용법:
    python light_forecast/multi_user.py --xlsx path/to/light_all.xlsx --out outputs/multi_user
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from forecast import (
    COMPARISONS,
    INITIAL_TRAIN_DAYS,
    MODELS,
    ONE_DAY,
    REFERENCE,
    SEGMENT_MODES,
    WEEKEND,
    bootstrap_vs_reference,
    hourly_matrix,
    rolling_origin,
    score,
)

MIN_EVAL_DAYS = 10  # 채점된 평가일이 이보다 적으면 사용자 간 요약에서 제외
SUMMARY_SUBSETS = (
    "common",
    "b3_and_learners",
    "weekday_common",
    "weekend_common",
    "yesterday_observed",
    "yesterday_missing",
)


def sheet_readings(sheet: pd.DataFrame) -> pd.DataFrame:
    """시트를 forecast.py의 측정치 형식(timestamp, lux)으로 바꾼다. 중복 timestamp는 평균."""
    readings = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(sheet["timestamp"], errors="coerce"),
            "lux": pd.to_numeric(sheet["value"], errors="coerce"),
        }
    ).dropna()
    return readings.groupby("timestamp", as_index=False)["lux"].mean()


def evaluate_user(
    user: str, sheet: pd.DataFrame, out: Path, segments: str = "recent_day", daytype_variants: bool = True
) -> dict:
    warnings.filterwarnings("ignore")
    readings = sheet_readings(sheet)
    hourly = hourly_matrix(readings, "mean")
    info = {
        "user": user,
        "start": hourly.index[0].date(),
        "end": hourly.index[-1].date(),
        "calendar_days": len(hourly),
        "days_with_data": int(hourly.notna().any(axis=1).sum()),
        "hour_coverage": hourly.notna().to_numpy().mean(),
    }
    evaluated = hourly.loc[hourly.index[0] + INITIAL_TRAIN_DAYS * ONE_DAY :]
    info["eval_days"] = int(evaluated.notna().any(axis=1).sum())
    info["eval_hours"] = int(evaluated.notna().sum().sum())
    if info["eval_hours"] == 0:
        info["status"] = "skipped: 첫 4주 이후 관측 없음"
        return {"info": info}

    pred = rolling_origin(
        hourly, skip_unobserved=True, segments=segments, daytype_variants=daytype_variants
    )
    user_dir = out / "users" / user
    user_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(user_dir / "predictions.csv", index=False)
    metrics = score(pred).assign(user=user)
    weekend = pred["date"].dt.weekday.isin(WEEKEND)
    boots = []
    for days, rows in (("all", pred), ("weekday", pred[~weekend]), ("weekend", pred[weekend])):
        for model, reference in COMPARISONS:
            if rows[[model, reference]].isna().all().any():
                continue
            if rows[[model, reference, "actual"]].notna().all(axis=1).any():
                boots.append({**bootstrap_vs_reference(rows, model, reference), "days_type": days, "user": user})
    info["status"] = "ok" if info["eval_days"] >= MIN_EVAL_DAYS else f"평가일 {MIN_EVAL_DAYS}일 미만"
    return {"info": info, "metrics": metrics, "boots": pd.DataFrame(boots), "pred": pred.assign(user=user)}


def cross_user_summary(metrics: pd.DataFrame, users: list[str], subset: str) -> pd.DataFrame:
    """사용자별 MAE를 모아 모델별 평균·중앙값, 기준 모델(B3)보다 나은 사용자 수, Wilcoxon 검정."""
    m = metrics[(metrics["subset"] == subset) & metrics["user"].isin(users)]
    wide = m.pivot(index="user", columns="model", values="MAE")
    rows = []
    for model in [c for c in MODELS if c in wide]:
        diff = (wide[model] - wide[REFERENCE]).dropna()
        p = wilcoxon(diff).pvalue if model != REFERENCE and (diff != 0).any() else np.nan
        rows.append(
            {
                "model": model,
                "users": int(wide[model].notna().sum()),
                "mean_MAE": wide[model].mean(),
                "median_MAE": wide[model].median(),
                "mean_rank": wide.rank(axis=1)[model].mean(),
                f"better_than_{REFERENCE}": f"{int((diff < 0).sum())}/{len(diff)}",
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="사용자별 조도 예측 + rolling-origin 평가")
    parser.add_argument("--xlsx", required=True, help="시트별 사용자 조도 엑셀 (timestamp, value 열)")
    parser.add_argument("--out", default="outputs/multi_user")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--segments", choices=SEGMENT_MODES, default="recent_day",
        help="구간 평균 피처: recent_day(최근 관측일 하루, 06/12/18시 경계) 또는 sun7d(일출·일몰 4구간 7일 평균)",
    )
    parser.add_argument("--no-daytype", action="store_true", help="평일/휴일 구분 학습 모델 생략")
    parser.add_argument("--summary-only", action="store_true", help="저장된 사용자별 결과로 요약만 다시 계산")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(out)
        return

    sheets = pd.read_excel(args.xlsx, sheet_name=None)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # 큰 사용자부터 시작해 병렬 처리 시간을 줄인다
        order = sorted(sheets, key=lambda s: -len(sheets[s]))
        futures = {
            pool.submit(evaluate_user, str(s), sheets[s], out, args.segments, not args.no_daytype): s
            for s in order
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            i = res["info"]
            print(f"user {i['user']:>4}: {i['status']}, 평가일 {i['eval_days']}, 평가 시각 {i['eval_hours']}", flush=True)

    info = pd.DataFrame([r["info"] for r in results]).sort_values("user", key=lambda s: s.astype(int))
    pd.concat([r["metrics"] for r in results if "metrics" in r]).to_csv(out / "metrics_by_user.csv", index=False)
    pd.concat([r["boots"] for r in results if "boots" in r]).to_csv(out / "bootstrap_by_user.csv", index=False)
    info.to_csv(out / "users.csv", index=False)
    summarize(out)


def summarize(out: Path) -> None:
    """저장된 사용자별 결과로 사용자 간 요약과 전체 합산 지표를 만든다."""
    info = pd.read_csv(out / "users.csv", dtype={"user": str})
    metrics = pd.read_csv(out / "metrics_by_user.csv", dtype={"user": str})
    included = info.loc[info["status"] == "ok", "user"].tolist()
    summaries = []
    for subset in SUMMARY_SUBSETS:
        summaries.append(cross_user_summary(metrics, included, subset).assign(subset=subset))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "summary_across_users.csv", index=False)

    # 모든 사용자의 평가 시각을 합쳐 계산한 지표 (시각 수가 많은 사용자의 비중이 크다)
    pooled = pd.concat(
        [pd.read_csv(out / "users" / u / "predictions.csv", parse_dates=["date"]) for u in included],
        ignore_index=True,
    )
    pooled_metrics = score(pooled)
    pooled_metrics.to_csv(out / "metrics_pooled.csv", index=False)

    from plots import plot_users

    plot_users(metrics, included, out / "per_user_errors.png")

    pd.set_option("display.width", 200)
    print("\n== 사용자 ==")
    print(info.round(2).to_string(index=False))
    print(f"\n== 사용자 간 요약 (평가일 {MIN_EVAL_DAYS}일 이상 {len(included)}명, 사용자별 MAE 기준) ==")
    print(summary.round(3).to_string(index=False))
    print("\n== 전체 평가 시각 합산 ==")
    print(pooled_metrics[pooled_metrics["subset"].isin(["common", "all"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
