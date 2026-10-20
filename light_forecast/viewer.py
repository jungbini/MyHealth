"""사용자별 LightGBM 예측값과 실측 조도를 한 그래프에서 비교하는 HTML 뷰어를 만든다.

엑셀의 각 시트(사용자)에 lightgbm_model.py의 rolling-origin 평가(첫 4주 이후 매일 전날까지
데이터로 다시 학습해 그날 0~23시를 예측)를 적용하고, 실측과 예측을 HTML 파일 하나에 넣는다.
결과 파일은 외부 라이브러리 없이 브라우저에서 열린다. 개인 건강 데이터가 들어 있으므로
저장소에는 올리지 않는다(outputs/는 .gitignore 대상).

사용법:
    python light_forecast/viewer.py --xlsx light_all.xlsx --out outputs/viewer/lightgbm_viewer.html

사용자별 예측은 --out 옆의 cache/ 폴더에 저장해 다시 실행할 때 재사용한다(--refresh로 다시 계산).
"""
from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm_model as L

TEMPLATE = Path(__file__).with_name("viewer_template.html")
PLACEHOLDER = "__VIEWER_DATA__"
MIN_EVAL_DAYS = 10  # 이보다 평가일이 적으면 뷰어에 '평가일 적음'으로 표시
PRED_COLUMNS = ["date", "hour", "pred_log_lux", "pred_lux", "actual_log_lux"]


def user_predictions(user: str, sheet: pd.DataFrame, cache_dir: Path, refresh: bool) -> dict:
    """한 사용자의 시간 행렬과 rolling-origin 예측(캐시 사용)을 뷰어용 레코드로 만든다."""
    warnings.filterwarnings("ignore")
    hourly = L.hourly_matrix(L.readings_from_frame(sheet))
    cache = cache_dir / f"{user}.csv"
    if cache.exists() and not refresh:
        pred = pd.read_csv(cache)
    else:
        pred = L.evaluate(hourly)
        if pred.empty:
            pred = pd.DataFrame(columns=PRED_COLUMNS)
        pred.to_csv(cache, index=False)
    return user_record(user, hourly, pred)


def _rounded(values: np.ndarray) -> list:
    return [None if np.isnan(v) else round(float(v), 3) for v in values]


def user_record(user: str, hourly: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """실측·예측을 (날짜 x 24시간)을 펼친 같은 길이의 배열로 담는다. 값은 log(lux+1), 없으면 null."""
    start = hourly.index[0]
    actual = hourly.to_numpy().ravel()
    predicted = np.full(actual.shape, np.nan)
    metrics = None
    if not pred.empty:
        day = (pd.to_datetime(pred["date"]) - start).dt.days.to_numpy()
        predicted[day * 24 + pred["hour"].to_numpy()] = pred["pred_log_lux"].to_numpy()
        m = L.metrics(pred)
        metrics = {k: (int(v) if k in ("days", "hours") else round(float(v), 4)) for k, v in m.items()}
    if metrics is None or metrics["hours"] == 0:
        status = "none"
    elif metrics["days"] < MIN_EVAL_DAYS:
        status = "few"
    else:
        status = "ok"
    return {
        "id": user,
        "start": start.date().isoformat(),
        "days": len(hourly),
        "status": status,
        "metrics": metrics,
        "actual": _rounded(actual),
        "pred": _rounded(predicted),
    }


def sheet_order(names) -> list[str]:
    """시트 이름이 숫자면 숫자 순서로, 아니면 이름 순서로 정렬한다."""
    return sorted(map(str, names), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))


def compute_records(xlsx: Path, cache_dir: Path, refresh: bool, workers: int) -> list[dict]:
    sheets = {str(k): v for k, v in pd.read_excel(xlsx, sheet_name=None).items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # 큰 사용자부터 제출해 병렬 처리 시간을 줄이고, 결과는 시트 순서로 모은다
        futures = {
            name: pool.submit(user_predictions, name, sheets[name], cache_dir, refresh)
            for name in sorted(sheets, key=lambda s: -len(sheets[s]))
        }
        return [futures[name].result() for name in sheet_order(sheets)]


def render_html(records: list[dict], source: str) -> str:
    payload = {
        "generated": date.today().isoformat(),
        "source": source,
        "initialTrainDays": L.INITIAL_TRAIN_DAYS,
        "users": records,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # <script> 안에 넣으므로 태그로 해석될 수 있는 문자를 이스케이프한다
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="사용자별 LightGBM 예측 vs 실측 HTML 뷰어 생성")
    parser.add_argument("--xlsx", required=True, help="시트별 사용자 조도 엑셀 (timestamp, value 열)")
    parser.add_argument("--out", default="outputs/viewer/lightgbm_viewer.html")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true", help="캐시를 무시하고 예측을 다시 계산")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = compute_records(Path(args.xlsx), out.parent / "cache", args.refresh, args.workers)
    out.write_text(render_html(records, Path(args.xlsx).name), encoding="utf-8")
    for r in records:
        m = r["metrics"]
        summary = f"평가일 {m['days']}, MAE {m['MAE']:.3f}, RMSE {m['RMSE']:.3f}" if m else "예측 없음"
        print(f"user {r['id']:>4}: {r['days']}일, {summary}")
    print(f"\n뷰어 저장: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
