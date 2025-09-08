# app.py (발췌) — 유효일수(가중 영업일) 분석 화면 추가

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# (기존 코드 상단 그대로) 한글 폰트 설정 등은 생략

# ------------------------------------------------------------
# 공통 유틸(추가)
def _to_date(s):
    try:
        # 20210101 같은 int/str도 처리
        s = str(s).strip()
        if len(s) == 8 and s.isdigit():
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def _detect_supply_col(df: pd.DataFrame):
    for c in df.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # 없으면 None
    return None

def _normalize_calendar(df: pd.DataFrame):
    """필수 컬럼 표준화 + 카테고리 라벨 생성"""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜/연/월/요일/구분 추출
    date_col = None
    for c in d.columns:
        if str(c).lower() in ["날짜", "일자", "date"]:
            date_col = c; break
    if date_col is None:
        for c in d.columns:
            if pd.api.types.is_integer_dtype(d[c]) or pd.api.types.is_object_dtype(d[c]):
                # yyyymmdd 후보
                if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.8 and d[c].astype(str).str.len().mode().iloc[0] in [7,8]:
                    date_col = c; break
    if date_col is None:
        raise ValueError("엑셀에서 날짜 열을 찾지 못했습니다. (예: 날짜/일자/date)")

    d["날짜"] = d[date_col].map(_to_date)
    d = d.dropna(subset=["날짜"]).copy()
    if "연" not in d.columns: d["연"] = d["날짜"].dt.year
    if "월" not in d.columns: d["월"] = d["날짜"].dt.month

    # 요일
    if "요일" not in d.columns:
        # 월=0..일=6
        yo_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
        d["요일"] = d["날짜"].dt.dayofweek.map(yo_map)

    # 구분(문자)
    if "구분" not in d.columns:
        d["구분"] = ""

    # 불리언 플래그 보정
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else:
            d[col] = np.nan

    # 명절 세부 추정: '구분'에 설/추 포함 → 그대로, 없으면 월로 추정
    def _infer_festival(row):
        gubun = str(row.get("구분",""))
        mon = int(row["월"])
        if "설" in gubun: return "명절_설날"
        if "추" in gubun: return "명절_추석"
        if str(row.get("명절여부","")).upper() == "TRUE":
            if mon in (1,2): return "명절_설날"
            if mon in (9,10): return "명절_추석"
            return "명절_기타"
        return None

    # 카테고리 매핑
    def _category(row):
        y = row["요일"]; g = str(row["구분"])
        # 공휴일/대체공휴일
        if ("공휴" in g) or (str(row.get("공휴일여부","")).upper() == "TRUE") or ("대체" in g):
            return "공휴일_대체"
        # 명절(우선 분기)
        fest = _infer_festival(row)
        if fest in ["명절_설날","명절_추석"]:
            return fest
        # 주말
        if y == "토": return "토요일"
        if y == "일": return "일요일"
        # 평일
        if y in ["화","수","목"]: return "평일_1"
        if y in ["월","금"]: return "평일_2"
        # fallback
        return "평일_1"

    d["카테고리"] = d.apply(_category, axis=1).astype("category")
    return d

# ------------------------------------------------------------
# 사이드바 라디오에 새 메뉴 추가 (기존 메뉴에 이어서)
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio(
        "선택",
        ["공급량 분석", "판매량 분석(냉방용)", "모델 비교(정확도 리더보드)", "유효일수 분석"],
        index=0
    )

# ============================================================
# 유효일수 분석 화면
if mode == "유효일수 분석":
    st.header("유효일수(가중 영업일) 분석")
    st.caption("카테고리: 평일_1(화·수·목), 평일_2(월·금), 토요일, 일요일, 공휴일/대체, 명절_설날, 명절_추석")

    # ─ 파일 입력
    with st.sidebar:
        st.subheader("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    c1, c2 = st.columns(2)
    if src == "Repo 내 파일 사용":
        # 기본 파일명 예: data/effective_days_calendar.xlsx
        default_path = Path("data") / "effective_days_calendar.xlsx"
        if default_path.exists():
            st.success(f"레포 파일 사용: {default_path.name}")
            file = open(default_path, "rb")
        else:
            file = c1.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
    else:
        file = c1.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

    if file is None:
        st.info("엑셀 파일을 업로드하거나 data 폴더에 배치하세요.")
        st.stop()

    # ─ 데이터 로드/정규화
    try:
        raw = pd.read_excel(file, engine="openpyxl")
    except Exception:
        st.error("엑셀을 읽는 중 문제가 발생했습니다. 형식을 확인하세요.")
        st.stop()

    try:
        df = _normalize_calendar(raw)
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        st.stop()

    # ─ 공급량 열 탐지(가중치 계산용, 없으면 수동입력)
    supply_col = _detect_supply_col(df)

    st.markdown("### 1) 카테고리 가중치 산정 (기본=평일_1 = 1.0000)")
    if supply_col is None:
        st.warning("일별 공급량 열(예: '공급량(MJ)')을 찾지 못해 가중치를 수동으로 입력합니다.")
        base_mean = 1.0
        auto_weights = {
            "평일_1": 1.0, "평일_2": 0.95, "토요일": 0.75, "일요일": 0.60,
            "공휴일_대체": 0.55, "명절_설날": 0.50, "명절_추석": 0.50
        }
    else:
        g = df.groupby("카테고리")[supply_col].mean().rename("mean")
        base_mean = g.get("평일_1", np.nan)
        if pd.isna(base_mean) or base_mean == 0:
            st.warning("평일_1 평균 공급량을 찾지 못했습니다. 가중치를 수동으로 입력하세요.")
            auto_weights = {
                "평일_1": 1.0, "평일_2": 0.95, "토요일": 0.75, "일요일": 0.60,
                "공휴일_대체": 0.55, "명절_설날": 0.50, "명절_추석": 0.50
            }
        else:
            auto_weights = {k: float(g.get(k, base_mean) / base_mean) for k in
                            ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]}

    # ─ 가중치 조정 UI
    cols = st.columns(7)
    cats = ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]
    weights = {}
    for i, cat in enumerate(cats):
        default_v = round(float(auto_weights.get(cat, 1.0)), 4)
        weights[cat] = cols[i].number_input(cat, value=default_v, step=0.0001, format="%.4f", key=f"w_{cat}")

    wdf = pd.DataFrame({"카테고리":cats, "가중치(평일_1=1.0000 기준)": [round(weights[c],4) for c in cats]})
    st.dataframe(wdf, use_container_width=True)

    st.markdown("### 2) 월별 유효일수 계산")
    # 월별 일수·카운트
    df["연"] = df["연"].astype(int); df["월"] = df["월"].astype(int)
    counts = df.pivot_table(index=["연","월"], columns="카테고리", values="날짜", aggfunc="count").fillna(0)
    counts = counts.reindex(columns=cats, fill_value=0)
    counts = counts.astype(int)

    # 월일수
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")

    # 유효일수(가중 합)
    eff = counts.copy().astype(float)
    for c in cats:
        eff[c] = eff[c] * float(weights[c])
    eff["유효일수합"] = eff.sum(axis=1)

    result = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_")], axis=1).reset_index()
    result["적용_비율(유효/월일수)"] = (result["적용_유효일수합"] / result["월일수"]).round(4)

    # 표 보여주기 (요약)
    show_cols = ["연","월","월일수","일수_평일_1","일수_평일_2","일수_토요일","일수_일요일","일수_공휴일_대체","일수_명절_설날","일수_명절_추석",
                 "적용_유효일수합","적용_비율(유효/월일수)"]
    st.dataframe(result[show_cols].sort_values(["연","월"]), use_container_width=True)

    # 상세(적용일수 분해)
    st.markdown("#### 적용 일수(유효) 상세")
    detail_cols = ["연","월"] + [f"적용_{c}" for c in cats] + ["적용_유효일수합","적용_비율(유효/월일수)"]
    st.dataframe(result[detail_cols].sort_values(["연","월"]), use_container_width=True)

    # CSV 다운로드
    st.download_button(
        "월별 유효일수 결과 CSV 다운로드",
        data=result.sort_values(["연","월"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="effective_days_by_month.csv",
        mime="text/csv"
    )

    st.info(
        "적용 예시: 1월 총일수=31, 1월 유효일수합=26.7이라면, "
        "**조정된 1월 공급량 = 원예측량 × (26.7 / 31)** 방식으로 적용."
    )
