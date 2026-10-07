"""날짜별 일출·일몰 시각(현지 시각, 소수 시간)을 NOAA 근사식으로 계산한다. 오차는 수 분 이내."""
from __future__ import annotations

import numpy as np
import pandas as pd

# 위치 정보가 없어 서울을 기본값으로 쓴다
SEOUL = {"lat": 37.5665, "lon": 126.9780, "utc_offset": 9.0}


def sun_times(dates: pd.DatetimeIndex, lat: float, lon: float, utc_offset: float) -> pd.DataFrame:
    """dates의 각 날짜에 대해 sunrise, sunset 열(현지 시각, 0~24 소수 시간)을 돌려준다."""
    gamma = 2 * np.pi / 365 * (dates.dayofyear.to_numpy() - 1)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    phi = np.radians(lat)
    # 태양 중심이 지평선 아래 0.833도(대기 굴절·태양 반지름 보정)일 때의 시간각
    cos_ha = np.cos(np.radians(90.833)) / (np.cos(phi) * np.cos(decl)) - np.tan(phi) * np.tan(decl)
    ha = np.degrees(np.arccos(np.clip(cos_ha, -1, 1)))
    to_local = utc_offset * 60
    sunrise = (720 - 4 * (lon + ha) - eqtime + to_local) / 60
    sunset = (720 - 4 * (lon - ha) - eqtime + to_local) / 60
    return pd.DataFrame({"sunrise": sunrise, "sunset": sunset}, index=dates)
