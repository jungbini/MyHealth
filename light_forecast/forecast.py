"""시간대별 조도(lux) 예측과 일 단위 rolling-origin 평가.

10분 간격 조도 측정치를 시간 단위로 요약한 뒤(log(lux+1)의 평균, 비교군으로 중앙값)
"어제까지의 데이터로 오늘 0~23시 조도를 예측"하는 다섯 모델을 비교한다.

  B1_yesterday   어제 같은 시각 값
  B2_last_week   지난주 같은 요일·같은 시각 값
  B3_median7     어제까지 최근 7일 같은 시각 값의 중앙값
  Ridge, LightGBM  아래 입력 X로 학습한 회귀 모델 (시간대 전체를 하나의 모델로 학습)

입력 X (예측 대상 날짜 D, 시각 h)
  lag_1 ~ lag_7        D-1 ~ D-7일의 h시 조도
  hist_mean, hist_std  D-1일까지 전체 과거의 h시 조도 평균·표준편차
  seg_morning/afternoon/evening
                       최근 날짜(D-1 우선, 없으면 7일 이내 가장 최근 관측일)의
                       오전(06-12시)·오후(12-18시)·저녁(18-24시) 평균 조도
  hour, weekday        예측 대상의 시각·요일
정답 y: D일 h시의 실제 조도 (log(lux+1) 척도)

평가: 첫 4주를 초기 학습 구간으로 두고 그 다음 날부터 하루씩 전진한다(expanding window).
각 날짜 D의 예측은 D 이전 관측만으로 만들어 고정하고, D의 실측이 공개된 뒤 오차를 계산한다.

사용법:
    python light_forecast/forecast.py --csv path/to/lux.csv --out outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from solar import SEOUL, sun_times

HOURS = list(range(24))
N_LAGS = 7
INITIAL_TRAIN_DAYS = 28  # 첫 4주
ONE_DAY = pd.Timedelta(days=1)
SEGMENTS = {"morning": range(6, 12), "afternoon": range(12, 18), "evening": range(18, 24)}
MIN_SEGMENT_HOURS = 3  # 구간(6시간) 중 절반 이상 관측된 날만 구간 평균으로 인정

LAG_COLS = [f"lag_{k}" for k in range(1, N_LAGS + 1)]
SEG_COLS = [f"seg_{name}" for name in SEGMENTS]
NUMERIC_COLS = LAG_COLS + ["hist_mean", "hist_std"] + SEG_COLS
# 일출·일몰 기준 4구간의 최근 7일 평균 (segments="sun7d")
#   밤: 일출-7시간 ~ 일출, 오전: 일출 ~ 12시, 오후: 12시 ~ 일몰, 저녁: 일몰 ~ 다음날 일출-7시간
SUN_SEGMENTS = ["night", "morning", "afternoon", "evening"]
SUN_COLS = [f"sun7_{name}" for name in SUN_SEGMENTS]
NIGHT_HOURS_BEFORE_SUNRISE = 7
MIN_SUN_SEGMENT_HOURS = 3  # 7일 동안 그 구간에 관측된 시간이 이보다 적으면 NaN
LOCATION = SEOUL
CALENDAR_COLS = ["hour", "weekday"]
LGBM_COLS = NUMERIC_COLS + CALENDAR_COLS
SEGMENT_MODES = ("recent_day", "sun7d")


def numeric_cols(segments: str = "recent_day") -> list[str]:
    """segments='recent_day'는 최근 관측일 하루의 06-12/12-18/18-24시 평균, 'sun7d'는 일출·일몰 4구간 7일 평균."""
    return LAG_COLS + ["hist_mean", "hist_std"] + (SUN_COLS if segments == "sun7d" else SEG_COLS)
# 평일/휴일 구분: 평일은 어제(D-1), 휴일(토·일)은 지난주 같은 요일(D-7)을 기준값으로 쓴다
WEEKEND = (5, 6)
DAYTYPE_COLS = ["is_weekend", "daytype_ref"]

BASELINES = ["B1_yesterday", "B2_last_week", "B3_median7", "B4_daytype"]
LEARNERS = ["Ridge", "LightGBM", "Ridge_daytype", "LightGBM_daytype"]
MODELS = BASELINES + LEARNERS
REFERENCE = "B3_median7"  # 모든 평가 시각에서 예측 가능한 단순 모델
# 부트스트랩으로 차이를 검정할 (model, reference) 쌍. 두 모델이 모두 예측한 시각에서 비교한다.
COMPARISONS = [(m, REFERENCE) for m in MODELS if m != REFERENCE] + [
    ("Ridge", "B1_yesterday"),
    ("LightGBM", "B1_yesterday"),
    ("B4_daytype", "B1_yesterday"),
    ("Ridge_daytype", "Ridge"),
    ("LightGBM_daytype", "LightGBM"),
]

# 평가 구간 결과를 보고 고르지 않도록, 작은 표본에 맞춘 보수적인 값으로 미리 고정
LGBM_PARAMS = dict(
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
RIDGE_ALPHAS = np.logspace(-2, 4, 13)
RIDGE_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def load_readings(path: str | Path) -> pd.DataFrame:
    """CSV(첫 열 timestamp, 둘째 열 조도)를 읽는다. 같은 timestamp의 중복 행은 평균으로 합친다."""
    raw = pd.read_csv(path)
    ts_col, lux_col = raw.columns[:2]
    readings = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[ts_col]),
            "lux": pd.to_numeric(raw[lux_col], errors="coerce"),
        }
    ).dropna()
    return readings.groupby("timestamp", as_index=False)["lux"].mean()


def hourly_matrix(readings: pd.DataFrame, agg: str, min_readings: int = 1) -> pd.DataFrame:
    """(연속 달력 날짜 x 0~23시) 행렬. 값은 그 시간 측정치들의 log(lux+1)을 agg로 요약한 것.

    agg='mean'은 log(lux+1)의 평균, agg='median'은 log(lux+1)의 중앙값(비교군).
    측정치가 min_readings개 미만인 시간과 측정이 없는 날짜는 NaN으로 둔다.
    """
    ts = readings["timestamp"]
    grouped = readings.assign(
        log_lux=np.log1p(readings["lux"].clip(lower=0)),
        date=ts.dt.normalize(),
        hour=ts.dt.hour,
    ).groupby(["date", "hour"])["log_lux"]
    value = grouped.agg(agg).where(grouped.size() >= min_readings)
    matrix = value.unstack("hour")
    days = pd.date_range(matrix.index.min(), matrix.index.max(), freq="D", name="date")
    return matrix.reindex(index=days, columns=HOURS)


# ---------------------------------------------------------------------------
# 입력 X
# ---------------------------------------------------------------------------
def build_features(visible: pd.DataFrame) -> pd.DataFrame:
    """visible(연속 달력 x 24시간)의 모든 (날짜 d, 시각 h)에 대해 입력 X를 만든다.

    모든 피처는 1일 이상 shift해서 계산하므로 d행의 피처에는 d보다 앞선 날짜의 값만 들어간다.
    """
    per_hour = {f"lag_{k}": visible.shift(k) for k in range(1, N_LAGS + 1)}
    per_hour["hist_mean"] = visible.expanding(min_periods=1).mean().shift(1)
    per_hour["hist_std"] = visible.expanding(min_periods=2).std().shift(1)
    features = {name: frame.to_numpy().ravel() for name, frame in per_hour.items()}

    for name, hours in SEGMENTS.items():
        block = visible[list(hours)]
        day_mean = block.mean(axis=1).where(block.notna().sum(axis=1) >= MIN_SEGMENT_HOURS)
        # d-1일 값을 우선 쓰고, 없으면 d-7일까지 거슬러 올라가 가장 최근 관측일 값을 쓴다
        recent = day_mean.shift(1).ffill(limit=N_LAGS - 1)
        features[f"seg_{name}"] = np.repeat(recent.to_numpy(), len(HOURS))

    for name, recent in sun_segment_means(visible).items():
        features[f"sun7_{name}"] = np.repeat(recent.to_numpy(), len(HOURS))

    index = pd.MultiIndex.from_product([visible.index, HOURS], names=["date", "hour"])
    X = pd.DataFrame(features, index=index).reset_index()
    X["weekday"] = X["date"].dt.weekday
    X["is_weekend"] = X["weekday"].isin(WEEKEND).astype(float)
    X["daytype_ref"] = X["lag_1"].where(X["is_weekend"] == 0, X["lag_7"])
    return X


def sun_segment_labels(dates: pd.DatetimeIndex, location: dict = LOCATION) -> np.ndarray:
    """(날짜 x 24시간) 각 시간 칸이 속한 구간 이름. 칸의 위치는 그 시간의 중앙(h+0.5시)으로 판단한다."""
    sun = sun_times(dates, **location)
    rise = sun["sunrise"].to_numpy()[:, None]
    set_ = sun["sunset"].to_numpy()[:, None]
    rise_next = sun_times(dates + ONE_DAY, **location)["sunrise"].to_numpy()[:, None]
    t = np.arange(24) + 0.5
    labels = np.full((len(dates), 24), "evening", dtype=object)
    labels[(t >= rise) & (t < 12)] = "morning"
    labels[(t >= 12) & (t < set_)] = "afternoon"
    # 그날 일출 전 7시간, 또는 다음날 일출 7시간 전부터 자정까지는 밤
    labels[(t < rise) & (t >= rise - NIGHT_HOURS_BEFORE_SUNRISE)] = "night"
    labels[t >= 24 + rise_next - NIGHT_HOURS_BEFORE_SUNRISE] = "night"
    return labels


def sun_segment_means(visible: pd.DataFrame) -> dict[str, pd.Series]:
    """d행에 d-7 ~ d-1일(자정 기준 7일, d일 이전 칸만)의 구간별 평균을 둔다."""
    labels = sun_segment_labels(visible.index)
    values = visible.to_numpy()
    out = {}
    for name in SUN_SEGMENTS:
        in_seg = (labels == name) & ~np.isnan(values)
        daily_sum = pd.Series(np.where(in_seg, values, 0).sum(axis=1), index=visible.index)
        daily_n = pd.Series(in_seg.sum(axis=1), index=visible.index)
        total = daily_sum.rolling(N_LAGS, min_periods=1).sum().shift(1)
        n = daily_n.rolling(N_LAGS, min_periods=1).sum().shift(1)
        out[name] = (total / n).where(n >= MIN_SUN_SEGMENT_HOURS)
    return out


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def baseline_predictions(X: pd.DataFrame) -> pd.DataFrame:
    """단순 대체 모델. 필요한 과거 값이 없으면 NaN(예측 불가)으로 둔다.

    B4_daytype은 평일이면 어제 값, 휴일(토·일)이면 지난주 같은 요일 값이다.
    """
    return pd.DataFrame(
        {
            "B1_yesterday": X["lag_1"],
            "B2_last_week": X["lag_7"],
            "B3_median7": X[LAG_COLS].median(axis=1, skipna=True),
            "B4_daytype": X["daytype_ref"],
        },
        index=X.index,
    )


def _ref_cols(daytype: bool) -> list[str]:
    """결측을 hist_mean으로 채우고 결측 지시변수를 붙일 기준값 열."""
    return LAG_COLS + (["daytype_ref"] if daytype else [])


def ridge_inputs(X: pd.DataFrame, daytype: bool = False, segments: str = "recent_day") -> pd.DataFrame:
    """Ridge용 입력: 빈 lag는 같은 시각의 과거 평균(hist_mean)으로 채우고 결측 지시변수를 붙인다."""
    Z = X[numeric_cols(segments) + CALENDAR_COLS + (DAYTYPE_COLS if daytype else [])].copy()
    for col in _ref_cols(daytype):
        Z[f"{col}_missing"] = Z[col].isna().astype(float)
        Z[col] = Z[col].fillna(Z["hist_mean"])
    return Z


def make_ridge(daytype: bool = False, segments: str = "recent_day") -> Pipeline:
    numeric = (
        numeric_cols(segments)
        + (DAYTYPE_COLS if daytype else [])
        + [f"{col}_missing" for col in _ref_cols(daytype)]
    )
    preprocess = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="mean", add_indicator=True, keep_empty_features=True),
                    StandardScaler(),
                ),
                numeric,
            ),
            ("cal", OneHotEncoder(handle_unknown="ignore"), CALENDAR_COLS),
        ]
    )
    return make_pipeline(preprocess, Ridge())


def _one_standard_error_rule(cv_results: dict) -> int:
    """CV 점수가 최고점에서 1 표준오차 이내인 alpha 중 가장 큰 값(가장 강한 규제)을 고른다."""
    score = np.asarray(cv_results["mean_test_score"])
    se = np.asarray(cv_results["std_test_score"]) / np.sqrt(RIDGE_CV_FOLDS)
    alphas = np.asarray(cv_results["param_ridge__alpha"], dtype=float)
    best = int(np.argmax(score))
    candidates = np.flatnonzero(score >= score[best] - se[best])
    return int(candidates[np.argmax(alphas[candidates])])


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alpha_rule: str = "min",
    daytype: bool = False,
    segments: str = "recent_day",
) -> Pipeline:
    """alpha는 학습 구간 안에서 날짜 단위 GroupKFold로 고른다.

    같은 날의 24개 행은 날짜 단위 피처(lag 결측 패턴, 구간 평균)를 공유하므로 행 단위
    leave-one-out은 지나치게 낙관적이어서 alpha를 너무 작게 고르고 외삽이 폭주한다.
    alpha_rule='min'은 CV 오차가 가장 작은 alpha, '1se'는 1-SE 규칙(민감도 분석용).
    """
    search = GridSearchCV(
        make_ridge(daytype, segments),
        {"ridge__alpha": RIDGE_ALPHAS},
        cv=GroupKFold(n_splits=RIDGE_CV_FOLDS),
        scoring="neg_mean_squared_error",
        refit=_one_standard_error_rule if alpha_rule == "1se" else True,
    )
    search.fit(ridge_inputs(X_train, daytype, segments), y_train, groups=X_train["date"].to_numpy())
    return search.best_estimator_


def forecast_day(
    hourly: pd.DataFrame,
    target: pd.Timestamp,
    ridge_alpha_rule: str = "min",
    segments: str = "recent_day",
    daytype_variants: bool = True,
) -> pd.DataFrame:
    """target 날짜의 0~23시 조도를 예측한다. target 이전(< target) 관측만 사용한다.

    segments는 구간 평균 피처의 정의(numeric_cols 참고). daytype_variants=False면
    평일/휴일 구분 학습 모델(_daytype)은 계산하지 않고 NaN으로 둔다.

    학습 사례는 (d, h) 쌍: 입력 = d 이전 날짜로 만든 X, 정답 = d일 h시 실측.
    d는 lag 7개가 정의되는 첫 날(시작일 + 7일)부터 target 전날까지이다.
    """
    start = hourly.index[0]
    # target 이후 관측은 잘라내고 target 당일 행은 비워 둔 '예측 시점의 데이터'
    visible = hourly.loc[: target - ONE_DAY].reindex(
        pd.date_range(start, target, freq="D", name="date")
    )
    X = build_features(visible)
    y = visible.to_numpy().ravel()

    is_train = (
        (X["date"] >= start + N_LAGS * ONE_DAY) & (X["date"] < target) & ~np.isnan(y)
    ).to_numpy()
    X_train, y_train = X[is_train], y[is_train]
    X_test = X[X["date"] == target]

    pred = pd.concat([X_test[["date", "hour"]], baseline_predictions(X_test)], axis=1)
    pred["n_train"] = len(y_train)
    if X_train["date"].nunique() < RIDGE_CV_FOLDS:
        # 학습일이 날짜 단위 CV fold 수보다 적으면 학습 모델은 예측하지 않는다
        pred[LEARNERS] = np.nan
        pred["ridge_alpha"] = np.nan
        return pred.reset_index(drop=True)
    # 기존 모델과 평일/휴일 구분 모델(_daytype)을 같은 학습 사례로 학습한다
    variants = ((False, ""), (True, "_daytype")) if daytype_variants else ((False, ""),)
    pred[LEARNERS] = np.nan
    for daytype, suffix in variants:
        ridge = fit_ridge(X_train, y_train, ridge_alpha_rule, daytype, segments)
        pred[f"Ridge{suffix}"] = ridge.predict(ridge_inputs(X_test, daytype, segments))
        cols = numeric_cols(segments) + CALENDAR_COLS + (DAYTYPE_COLS if daytype else [])
        lgbm = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X_train[cols], y_train)
        pred[f"LightGBM{suffix}"] = lgbm.predict(X_test[cols])
        if not daytype:
            pred["ridge_alpha"] = ridge[-1].alpha
    pred[LEARNERS] = pred[LEARNERS].clip(lower=0)  # log(lux+1) >= 0
    return pred.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def rolling_origin(
    hourly: pd.DataFrame,
    initial_days: int = INITIAL_TRAIN_DAYS,
    ridge_alpha_rule: str = "min",
    skip_unobserved: bool = False,
    segments: str = "recent_day",
    daytype_variants: bool = True,
) -> pd.DataFrame:
    """첫 initial_days일 이후 하루씩 전진하는 순차 검증 결과(날짜 x 시각 x 모델 예측 + 실측).

    skip_unobserved=True면 실측이 하나도 없는 날(채점할 수 없는 날)은 예측을 생략한다.
    다른 날의 예측에는 영향이 없다(각 날의 학습은 그 이전 관측만으로 매번 새로 한다).
    """
    start, end = hourly.index[0], hourly.index[-1]
    results = []
    for target in pd.date_range(start + initial_days * ONE_DAY, end, freq="D"):
        if skip_unobserved and hourly.loc[target].isna().all():
            continue
        pred = forecast_day(hourly, target, ridge_alpha_rule, segments, daytype_variants)  # 1) target의 실측을 보기 전에 예측을 고정
        pred["actual"] = hourly.loc[target].to_numpy()  # 2) 실측이 공개된 뒤 비교
        results.append(pred)
    return pd.concat(results, ignore_index=True)


def evaluation_subsets(pred: pd.DataFrame) -> dict[str, pd.Series]:
    """실측이 있는 시각을 비교 목적에 따라 나눈다.

    common             모든 모델이 예측 가능한 시각 (모델 공정 비교)
    yesterday_observed 어제 같은 시각 값이 있는 시각 (B1과 학습 모델을 같은 행에서 비교)
    yesterday_missing  어제 값이 없어 B1이 예측할 수 없는 시각
    all                실측이 있는 모든 시각 (각 모델은 예측 가능한 시각만, coverage로 표시)
    weekday_*/weekend_* 위 common·all을 평일/휴일(토·일)로 나눈 것
    """
    observed = pred["actual"].notna()
    has_yesterday = pred["B1_yesterday"].notna()
    computed = [m for m in MODELS if pred[m].notna().any()]  # 계산하지 않은 모델은 제외
    common = observed & pred[computed].notna().all(axis=1)
    weekend = pred["date"].dt.weekday.isin(WEEKEND)
    return {
        "common": common,
        "yesterday_observed": observed & has_yesterday,
        "yesterday_missing": observed & ~has_yesterday,
        "all": observed,
        # 과거 특정일 값이 필요 없는 모델(B3, 학습 모델)이 모두 예측 가능한 시각
        "b3_and_learners": observed
        & pred[[REFERENCE] + [m for m in LEARNERS if m in computed]].notna().all(axis=1),
        "weekday_common": common & ~weekend,
        "weekend_common": common & weekend,
        "weekday_all": observed & ~weekend,
        "weekend_all": observed & weekend,
    }


def score(pred: pd.DataFrame) -> pd.DataFrame:
    """모델별 MAE·RMSE(log(lux+1) 척도)와 lux 척도 MAE.

    '...all' 이외의 부분집합에서는 그 부분집합 전체를 예측할 수 있는 모델만 집계해
    모든 모델이 같은 행에서 비교되도록 한다.
    """
    rows = []
    for subset, mask in evaluation_subsets(pred).items():
        for model in MODELS:
            ok = mask & pred[model].notna()
            if not subset.endswith("all") and ok.sum() < mask.sum():
                continue
            err = pred.loc[ok, model] - pred.loc[ok, "actual"]
            lux_err = np.expm1(pred.loc[ok, model]) - np.expm1(pred.loc[ok, "actual"])
            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "n": int(ok.sum()),
                    "coverage": ok.sum() / mask.sum(),
                    "MAE": err.abs().mean(),
                    "RMSE": np.sqrt((err**2).mean()),
                    "MAE_lux": lux_err.abs().mean(),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_vs_reference(
    pred: pd.DataFrame, model: str, reference: str = REFERENCE, n_boot: int = 5000, seed: int = 0
) -> dict:
    """평가일 단위 블록 부트스트랩으로 본 MAE·RMSE 차이(model - reference)와 95% 구간.

    두 모델이 모두 예측한 시각만 쓰고, 날짜를 복원추출한다(같은 날 시각들 간 상관 반영).
    차이가 음수면 model이 더 정확하다.
    """
    ok = pred["actual"].notna() & pred[model].notna() & pred[reference].notna()
    d = pred[ok]
    err_m = d[model] - d["actual"]
    err_r = d[reference] - d["actual"]
    per_day = (
        pd.DataFrame(
            {
                "date": d["date"],
                "abs_m": err_m.abs(),
                "abs_r": err_r.abs(),
                "sq_m": err_m**2,
                "sq_r": err_r**2,
            }
        )
        .groupby("date")
        .agg(["sum", "size"])
    )
    sums = per_day.xs("sum", axis=1, level=1)[["abs_m", "abs_r", "sq_m", "sq_r"]].to_numpy()
    counts = per_day[("abs_m", "size")].to_numpy()

    def diffs(s: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mae = (s[..., 0] - s[..., 1]) / n
        rmse = np.sqrt(s[..., 2] / n) - np.sqrt(s[..., 3] / n)
        return mae, rmse

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot_mae, boot_rmse = diffs(sums[idx].sum(axis=1), counts[idx].sum(axis=1))
    mae, rmse = diffs(sums.sum(axis=0), counts.sum())
    return {
        "model": model,
        "reference": reference,
        "n": int(counts.sum()),
        "days": len(sums),
        "MAE_diff": mae,
        "MAE_ci_low": np.percentile(boot_mae, 2.5),
        "MAE_ci_high": np.percentile(boot_mae, 97.5),
        "RMSE_diff": rmse,
        "RMSE_ci_low": np.percentile(boot_rmse, 2.5),
        "RMSE_ci_high": np.percentile(boot_rmse, 97.5),
    }


def error_by(pred: pd.DataFrame, key: str) -> pd.DataFrame:
    """key('hour' 또는 'date')별 모델 MAE. 각 모델은 예측 가능한 시각만 집계한다."""
    observed = pred[pred["actual"].notna()]
    err = observed[MODELS].sub(observed["actual"], axis=0).abs()
    err[key] = observed[key]
    table = err.groupby(key)[MODELS].mean()
    table["n"] = err.groupby(key).size()
    return table


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def describe_data(readings: pd.DataFrame, n_raw: int, hourly: pd.DataFrame) -> dict:
    start, end = hourly.index[0], hourly.index[-1]
    eval_start = start + INITIAL_TRAIN_DAYS * ONE_DAY
    evaluated = hourly.loc[eval_start:]
    return {
        "raw_rows": n_raw,
        "readings_after_dedup": len(readings),
        "period": f"{start.date()} ~ {end.date()}",
        "calendar_days": len(hourly),
        "days_with_data": int(hourly.notna().any(axis=1).sum()),
        "hour_cells_observed": int(hourly.notna().sum().sum()),
        "hour_cells_total": hourly.size,
        "initial_train": f"{start.date()} ~ {(eval_start - ONE_DAY).date()}",
        "evaluation": f"{eval_start.date()} ~ {end.date()}",
        "eval_days_with_data": int(evaluated.notna().any(axis=1).sum()),
        "eval_hours_observed": int(evaluated.notna().sum().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="시간대별 조도 예측 + rolling-origin 평가")
    parser.add_argument("--csv", required=True, help="timestamp, 조도 두 열로 된 CSV")
    parser.add_argument("--out", default="outputs", help="결과 저장 폴더")
    parser.add_argument("--min-readings", type=int, default=1, help="시간값 산출에 필요한 최소 측정 수")
    parser.add_argument(
        "--ridge-alpha-rule",
        choices=["min", "1se"],
        default="min",
        help="Ridge alpha 선택: CV 최소 오차(min) 또는 1-SE 규칙(1se, 민감도 분석용)",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_raw = len(pd.read_csv(args.csv))
    readings = load_readings(args.csv)

    metrics, boots, preds, next_days, info = [], [], {}, {}, None
    for agg in ("mean", "median"):
        hourly = hourly_matrix(readings, agg, args.min_readings)
        if info is None:
            info = describe_data(readings, n_raw, hourly)
        pred = rolling_origin(hourly, ridge_alpha_rule=args.ridge_alpha_rule)
        pred.to_csv(out / f"predictions_{agg}.csv", index=False)
        preds[agg] = pred

        m = score(pred)
        m.insert(0, "agg", agg)
        metrics.append(m)
        weekend = pred["date"].dt.weekday.isin(WEEKEND)
        for days, rows in (("all", pred), ("weekday", pred[~weekend]), ("weekend", pred[weekend])):
            for model, reference in COMPARISONS:
                b = bootstrap_vs_reference(rows, model, reference)
                b["agg"], b["days_type"] = agg, days
                boots.append(b)
        error_by(pred, "hour").to_csv(out / f"mae_by_hour_{agg}.csv")
        error_by(pred, "date").to_csv(out / f"mae_by_date_{agg}.csv")

        # '실제 예측': 마지막 관측일까지의 데이터로 다음 날 0~23시를 예측
        next_day = forecast_day(hourly, hourly.index[-1] + ONE_DAY, args.ridge_alpha_rule)
        next_day.to_csv(out / f"next_day_forecast_{agg}.csv", index=False)
        next_days[agg] = next_day

    metrics = pd.concat(metrics, ignore_index=True)
    boots = pd.DataFrame(boots)
    boots = boots[["agg", "days_type"] + [c for c in boots.columns if c not in ("agg", "days_type")]]
    metrics.to_csv(out / "metrics.csv", index=False)
    boots.to_csv(out / "bootstrap_vs_reference.csv", index=False)
    pd.Series(info).to_csv(out / "data_summary.csv", header=["value"])

    pd.set_option("display.width", 160)
    print("== 데이터 요약 ==")
    for k, v in info.items():
        print(f"  {k:>22}: {v}")
    first = preds["mean"].groupby("date")["n_train"].first()
    print(f"  {'train_rows (first→last)':>22}: {first.iloc[0]} → {first.iloc[-1]}")
    print("\n== 평가 지표 (log(lux+1) 척도, MAE_lux는 lux 척도) ==")
    print(metrics.round(3).to_string(index=False))
    print("\n== 오차 차이 (model - reference, 음수면 model 우수), 평가일 블록 부트스트랩 95% 구간 ==")
    print(boots.round(3).to_string(index=False))

    if not args.no_plots:
        from plots import make_all_plots

        make_all_plots(preds, metrics, next_days, out / "figures")
        print(f"\n그림 저장: {out / 'figures'}")


if __name__ == "__main__":
    main()
