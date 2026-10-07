"""lightgbm_model.py가 forecast.py의 LightGBM과 같은 결과를 내는지, 저장·불러오기가 맞는지 확인한다."""
import numpy as np
import pandas as pd

import forecast
import lightgbm_model as L
from test_forecast import synthetic_hourly


def test_features_match_forecast_module():
    hourly = synthetic_hourly()
    ours = L.build_features(hourly)[["date", "hour"] + L.FEATURES]
    theirs = forecast.build_features(hourly)[["date", "hour"] + L.FEATURES]
    pd.testing.assert_frame_equal(ours, theirs)


def test_evaluate_matches_forecast_rolling_origin():
    hourly = synthetic_hourly(days=33)
    ours = L.evaluate(hourly).dropna(subset=["actual_log_lux"])
    theirs = forecast.rolling_origin(hourly)
    theirs = theirs[theirs["actual"].notna()]
    merged = ours.assign(date=pd.to_datetime(ours["date"])).merge(theirs, on=["date", "hour"])
    assert len(merged) == len(ours) == len(theirs)
    np.testing.assert_allclose(merged["pred_log_lux"], merged["LightGBM"])


def test_saved_model_predicts_the_same(tmp_path):
    hourly = synthetic_hourly()
    target = hourly.index[-1] + pd.Timedelta(days=1)
    model = L.train(hourly, target)
    path = tmp_path / "model.txt"
    L.save_model(model, path, {"data_end": str(hourly.index[-1].date())})
    expected = L.predict_day(model, hourly, target)
    pd.testing.assert_frame_equal(L.predict_day(L.load_model(path), hourly, target), expected)
