"""forecast.py 검증: 시간 집계, 피처 정의, 미래 데이터 누출 여부 (합성 데이터 사용)."""
import numpy as np
import pandas as pd
import pytest

from forecast import (
    HOURS,
    build_features,
    forecast_day,
    hourly_matrix,
    rolling_origin,
)


def synthetic_hourly(days: int = 40, missing: float = 0.3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D", name="date")
    profile = np.clip(np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None) * 5
    values = np.clip(profile + rng.normal(0, 0.5, (days, 24)), 0, None)
    values[rng.random((days, 24)) < missing] = np.nan
    return pd.DataFrame(values, index=dates, columns=HOURS)


def test_hourly_matrix_mean_and_median_of_log_lux():
    readings = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 10:10", "2026-01-01 10:20", "2026-01-03 00:00"]
            ),
            "lux": [0.0, 9.0, 99.0, 0.0],
        }
    )
    mean = hourly_matrix(readings, "mean")
    median = hourly_matrix(readings, "median")

    assert mean.loc["2026-01-01", 10] == pytest.approx((np.log(1) + np.log(10) + np.log(100)) / 3)
    assert median.loc["2026-01-01", 10] == pytest.approx(np.log(10))
    # 측정이 없는 날짜(1/2)도 달력에 NaN 행으로 남는다
    assert list(mean.index.strftime("%m-%d")) == ["01-01", "01-02", "01-03"]
    assert mean.loc["2026-01-02"].isna().all()
    assert np.isnan(mean.loc["2026-01-01", 11])


def test_features_follow_calendar_days():
    hourly = synthetic_hourly(missing=0.0)
    hourly.loc["2026-01-15"] = np.nan  # 결측일
    X = build_features(hourly).set_index(["date", "hour"])
    d, h = pd.Timestamp("2026-01-20"), 9

    for k in range(1, 8):
        assert X.loc[(d, h), f"lag_{k}"] == pytest.approx(hourly.loc[d - pd.Timedelta(days=k), h], nan_ok=True)
    assert np.isnan(X.loc[(d, h), "lag_5"])  # 1/15는 결측
    past = hourly.loc[: d - pd.Timedelta(days=1), h]
    assert X.loc[(d, h), "hist_mean"] == pytest.approx(past.mean())
    assert X.loc[(d, h), "hist_std"] == pytest.approx(past.std())
    yesterday = hourly.loc[d - pd.Timedelta(days=1)]
    assert X.loc[(d, h), "seg_morning"] == pytest.approx(yesterday[6:12].mean())
    assert X.loc[(d, 0), "weekday"] == d.weekday()


def test_segment_mean_falls_back_to_most_recent_day_within_week():
    hourly = synthetic_hourly(missing=0.0)
    hourly.loc["2026-01-17":"2026-01-19"] = np.nan
    X = build_features(hourly).set_index(["date", "hour"])
    # 1/20 기준 어제(1/19)~1/17이 비어 있으므로 1/16 값을 사용
    assert X.loc[(pd.Timestamp("2026-01-20"), 0), "seg_evening"] == pytest.approx(
        hourly.loc["2026-01-16", 18:23].mean()
    )
    hourly.loc["2026-01-10":"2026-01-19"] = np.nan
    X = build_features(hourly).set_index(["date", "hour"])
    assert np.isnan(X.loc[(pd.Timestamp("2026-01-20"), 0), "seg_evening"])  # 7일 안에 관측 없음


def test_baselines():
    hourly = synthetic_hourly()
    target = hourly.index[30]
    pred = forecast_day(hourly, target).set_index("hour")
    window = hourly.loc[target - pd.Timedelta(days=7) : target - pd.Timedelta(days=1)]
    for h in HOURS:
        assert pred.loc[h, "B1_yesterday"] == pytest.approx(window.iloc[-1][h], nan_ok=True)
        assert pred.loc[h, "B2_last_week"] == pytest.approx(window.iloc[0][h], nan_ok=True)
        assert pred.loc[h, "B3_median7"] == pytest.approx(window[h].median(), nan_ok=True)
        # 평일은 어제 값, 휴일(토·일)은 지난주 같은 요일 값
        expected = window.iloc[0][h] if target.weekday() >= 5 else window.iloc[-1][h]
        assert pred.loc[h, "B4_daytype"] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("offset", range(7))
def test_daytype_reference_by_weekday(offset):
    hourly = synthetic_hourly(missing=0.0)
    X = build_features(hourly).set_index(["date", "hour"])
    d = pd.Timestamp("2026-01-19") + pd.Timedelta(days=offset)
    k = 7 if d.weekday() >= 5 else 1
    assert X.loc[(d, 8), "is_weekend"] == float(d.weekday() >= 5)
    assert X.loc[(d, 8), "daytype_ref"] == pytest.approx(hourly.loc[d - pd.Timedelta(days=k), 8])


def test_forecast_uses_only_observations_before_target():
    hourly = synthetic_hourly()
    target = hourly.index[30]
    expected = forecast_day(hourly, target)

    tampered = hourly.copy()
    tampered.loc[target:] = 99.0  # 예측일 당일과 이후 값을 전부 바꿔도
    pd.testing.assert_frame_equal(forecast_day(tampered, target), expected)
    truncated = hourly.loc[: target - pd.Timedelta(days=1)]  # 아예 없어도 예측은 같아야 한다
    pd.testing.assert_frame_equal(forecast_day(truncated, target), expected)


def test_rolling_origin_starts_after_four_weeks():
    hourly = synthetic_hourly(days=31)
    pred = rolling_origin(hourly)
    first_target = hourly.index[0] + pd.Timedelta(days=28)

    assert pred["date"].min() == first_target
    first = pred[pred["date"] == first_target]
    # 학습 사례: 8일째(lag 7개가 생기는 날)부터 28일째까지의 관측
    assert (first["n_train"] == hourly.iloc[7:28].notna().sum().sum()).all()
    np.testing.assert_array_equal(first["actual"].to_numpy(), hourly.loc[first_target].to_numpy())


def test_sun_segment_labels_follow_sunrise_and_sunset():
    from forecast import sun_segment_labels

    summer, winter = pd.Timestamp("2025-06-21"), pd.Timestamp("2025-12-21")
    labels = sun_segment_labels(pd.DatetimeIndex([summer, winter]))
    # 서울 하지: 일출 05:11, 일몰 19:56 / 동지: 일출 07:43, 일몰 17:17
    # 하지 다음날 일출 05:11 -> 밤은 22:11부터
    assert list(labels[0, [0, 4, 5, 11, 12, 19, 20, 21, 22, 23]]) == [
        "night", "night", "morning", "morning", "afternoon", "afternoon", "evening", "evening", "night", "night",
    ]
    # 동지 00시대(중앙 00:30)는 일출-7시간(00:43) 이전이라 전날 저녁, 그 뒤 일출 전까지 밤
    assert list(labels[1, [0, 1, 6, 7, 8, 16, 17, 23]]) == [
        "evening", "night", "night", "night", "morning", "afternoon", "evening", "evening",
    ]


def test_sun_segment_means_use_previous_seven_days_only():
    hourly = synthetic_hourly(missing=0.0)
    X = build_features(hourly).set_index(["date", "hour"])
    d = pd.Timestamp("2026-01-20")
    from forecast import sun_segment_labels

    window = hourly.loc[d - pd.Timedelta(days=7) : d - pd.Timedelta(days=1)]
    labels = sun_segment_labels(hourly.index)[hourly.index.get_indexer(window.index)]
    for name in ["night", "morning", "afternoon", "evening"]:
        expected = window.to_numpy()[labels == name].mean()
        assert X.loc[(d, 5), f"sun7_{name}"] == pytest.approx(expected)
