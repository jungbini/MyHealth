"""viewer.py 검증: 예측값의 배열 위치, 상태 구분, HTML 안전한 데이터 삽입."""
import json
import re

import numpy as np
import pandas as pd

import lightgbm_model as L
import viewer
from test_forecast import synthetic_hourly


def test_user_record_places_predictions_at_their_hour():
    hourly = synthetic_hourly(days=33)
    pred = L.evaluate(hourly)
    record = viewer.user_record("7", hourly, pred)

    assert len(record["actual"]) == len(record["pred"]) == len(hourly) * 24
    start = hourly.index[0]
    for row in pred.sample(10, random_state=0).itertuples():
        i = (pd.Timestamp(row.date) - start).days * 24 + row.hour
        assert record["pred"][i] == round(row.pred_log_lux, 3)
    # 첫 4주는 예측이 없다
    assert all(v is None for v in record["pred"][: L.INITIAL_TRAIN_DAYS * 24])
    observed = hourly.to_numpy().ravel()
    assert [v is None for v in record["actual"]] == list(np.isnan(observed))
    assert record["status"] == "few"  # 평가일 5일 (< 10일)
    assert record["metrics"]["hours"] == int(pred["actual_log_lux"].notna().sum())


def test_user_without_predictions():
    hourly = synthetic_hourly(days=20)
    record = viewer.user_record("8", hourly, pd.DataFrame(columns=viewer.PRED_COLUMNS))
    assert record["status"] == "none" and record["metrics"] is None
    assert all(v is None for v in record["pred"])


def test_render_html_keeps_script_tags_intact():
    hourly = synthetic_hourly(days=10)
    record = viewer.user_record("</script><b>x&y", hourly, pd.DataFrame(columns=viewer.PRED_COLUMNS))
    html = viewer.render_html([record], "data.xlsx")
    template = viewer.TEMPLATE.read_text(encoding="utf-8")
    assert html.count("</script>") == template.count("</script>")
    payload = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(payload)["users"][0]["id"] == "</script><b>x&y"
