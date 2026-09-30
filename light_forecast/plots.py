"""forecast.py 결과 그림 (matplotlib, PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast import MODELS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# 모델마다 모든 그림에서 같은 색 (검증된 범주형 팔레트 1~5번 슬롯)
COLORS = {
    "Ridge": "#2a78d6",
    "LightGBM": "#eb6834",
    "B3_median7": "#1baf7a",
    "B1_yesterday": "#eda100",
    "B2_last_week": "#e87ba4",
}
LABELS = {
    "B1_yesterday": "B1 yesterday",
    "B2_last_week": "B2 same weekday last week",
    "B3_median7": "B3 7-day median",
    "Ridge": "Ridge",
    "LightGBM": "LightGBM",
}
LUX_TICKS = [0, 10, 100, 1000, 10000]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "font.size": 9,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)


def _clean(ax: plt.Axes, grid_axis: str = "y") -> None:
    """축선은 기준선 하나만, 격자는 값 축 방향의 가는 선만 남긴다."""
    baseline = "bottom" if grid_axis == "y" else "left"
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side == baseline)
    ax.grid(False)
    ax.grid(axis=grid_axis, visible=True)
    ax.tick_params(length=0)


def _lux_axis(ax: plt.Axes) -> None:
    """log(lux+1) 축에 lux 값으로 눈금을 단다 (축은 하나)."""
    ax.set_yticks(np.log1p(LUX_TICKS))
    ax.set_yticklabels([f"{v:,}" for v in LUX_TICKS])
    ax.set_ylabel("lux (log scale)")
    ax.set_ylim(-0.2, np.log1p(12000))


def _hour_axis(ax: plt.Axes) -> None:
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("hour of day")


def plot_model_errors(metrics: pd.DataFrame, path: Path, agg: str = "mean") -> None:
    common = metrics[(metrics["agg"] == agg) & (metrics["subset"] == "common")].set_index("model")
    n = int(common["n"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    order = MODELS[::-1]
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        values = common.loc[order, metric]
        ax.barh(
            [LABELS[m] for m in order],
            values,
            color=[COLORS[m] for m in order],
            height=0.62,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for i, v in enumerate(values):
            ax.text(v + values.max() * 0.015, i, f"{v:.3f}", va="center", color=INK_2, fontsize=8.5)
        ax.set_xlim(0, values.max() * 1.18)
        ax.set_title(f"{metric}  (log(lux+1), lower is better)")
        _clean(ax, grid_axis="x")
    fig.suptitle(
        f"Common evaluation set: {n} hours where all five models can forecast "
        f"(hourly {agg} of log(lux+1))",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _label_line_ends(ax: plt.Axes, x: float, ends: dict[str, float], min_gap: float) -> None:
    """선 끝 직접 라벨. 값이 가까우면 min_gap 이상 벌려 겹치지 않게 한다."""
    placed: list[float] = []
    for name, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(
            LABELS[name], (x, y), xytext=(x + 0.5, y_label), textcoords="data",
            va="center", color=INK_2, fontsize=8.5,
        )


def plot_error_by_hour(pred: pd.DataFrame, path: Path) -> None:
    models = ["B3_median7", "Ridge", "LightGBM"]  # 모든 평가 시각을 예측하는 모델
    obs = pred[pred["actual"].notna()]
    mae = obs[models].sub(obs["actual"], axis=0).abs().groupby(obs["hour"]).mean()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for m in models:
        ax.plot(mae.index, mae[m], color=COLORS[m], lw=2, marker="o", ms=4, label=LABELS[m])
    top = mae[models].to_numpy().max() * 1.15
    _label_line_ends(ax, 23, {m: mae[m].iloc[-1] for m in models}, min_gap=top * 0.06)
    _hour_axis(ax)
    ax.set_xlim(-0.5, 28)
    ax.set_ylabel("MAE, log(lux+1)")
    ax.set_ylim(0, top)
    ax.set_title(f"MAE by hour of day, all {len(obs)} evaluated hours")
    ax.legend(loc="upper left", ncol=3)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_by_day(pred: pd.DataFrame, path: Path) -> None:
    models = ["B3_median7", "Ridge", "LightGBM"]
    obs = pred[pred["actual"].notna()]
    daily = obs[models].sub(obs["actual"], axis=0).abs().groupby(obs["date"]).mean()
    calendar = pd.date_range(pred["date"].min(), pred["date"].max(), freq="D")
    daily = daily.reindex(calendar)  # 관측 없는 날은 선을 끊는다
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for m in models:
        ax.plot(daily.index, daily[m], color=COLORS[m], lw=1.5, marker="o", ms=4, label=LABELS[m])
    ax.set_ylabel("daily MAE, log(lux+1)")
    ax.set_ylim(bottom=0)
    ax.set_title("Daily MAE through the rolling-origin evaluation (gaps = days without data)")
    ax.legend(loc="upper left", ncol=3)
    fig.autofmt_xdate()
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example_days(pred: pd.DataFrame, path: Path, n_days: int = 3) -> None:
    models = ["B1_yesterday", "B3_median7", "Ridge", "LightGBM"]
    counts = pred.groupby("date")["actual"].count()
    full_days = counts[counts == 24].index[-n_days:]
    fig, axes = plt.subplots(1, len(full_days), figsize=(4 * len(full_days), 3.6), sharey=True)
    for ax, day in zip(np.atleast_1d(axes), full_days):
        d = pred[pred["date"] == day]
        ax.plot(d["hour"], d["actual"], color=INK, lw=2.4, label="actual")
        for m in models:
            ax.plot(d["hour"], d[m], color=COLORS[m], lw=1.6, label=LABELS[m])
        ax.set_title(pd.Timestamp(day).strftime("%Y-%m-%d (%a)"))
        _hour_axis(ax)
        _lux_axis(ax)
        _clean(ax)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7.5)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")
    fig.suptitle("Frozen forecasts vs. actual on the last fully observed evaluation days",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_next_day(next_day: pd.DataFrame, path: Path) -> None:
    day = pd.Timestamp(next_day["date"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for m in MODELS:
        ax.plot(next_day["hour"], next_day[m], color=COLORS[m], lw=2, marker="o", ms=3.5, label=LABELS[m])
    _hour_axis(ax)
    _lux_axis(ax)
    ax.set_title(f"Next-day forecast for {day:%Y-%m-%d (%a)} using data up to the day before")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_all_plots(preds: dict, metrics: pd.DataFrame, next_days: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plot_model_errors(metrics, out / "fig1_model_errors_common.png", "mean")
    plot_model_errors(metrics, out / "fig1b_model_errors_common_median.png", "median")
    plot_error_by_hour(preds["mean"], out / "fig2_mae_by_hour.png")
    plot_error_by_day(preds["mean"], out / "fig3_mae_by_day.png")
    plot_example_days(preds["mean"], out / "fig4_example_days.png")
    plot_next_day(next_days["mean"], out / "fig5_next_day_forecast.png")
