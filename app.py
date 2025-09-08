# app.py — Effective Days (유효일수 분석 전용)
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 기본 세팅
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ─────────────────────────────────────────────────────────────
# 한글 폰트 (가능하면 나눔/맑은고딕 사용)
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic.ttf",
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.family"] = [fam]
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ─────────────────────────────────────────────────────────────
# 유틸/상수
CATS = ["평일_1", "평일_2", "토요일", "일요일", "공휴일_대체", "명절_설날", "명절_추석"]
CAT_SHORT = {"평일_1": "평1", "평일_2": "평2", "토요일": "토", "일요일": "일", "공휴일_대체": "휴", "명절_설날": "설", "명절_추석": "추"}
PALETTE = {
    "평일_1": "#7DC3C1",     # teal light
    "평일_2": "#3DA4AB",     # teal
    "토요일": "#5D6D7E",     # slate
    "일요일": "#34495E",     # deep slate
    "공휴일_대체": "#E57373", # soft red
    "명절_설날": "#F5C04A",   # warm gold
    "명절_추석": "#F39C12",   # amber
}
DEFAULT_WEIGHTS = {
    "평일_1": 1.0,
    "평일_2": 0.952,
    "토요일": 0.85,
    "일요일": 0.60,
    "공휴일_대체": 0.799,
    "명절_설날": 0.842,
    "명절_추석": 0.799,
}

def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """엑셀 원본을 표준 스키마로 정규화하고 (DataFrame, 공급량컬럼명 or None) 반환"""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 열 찾기
    date_col = None
    for c in d.columns:
        if str(c).lower() in ["날짜", "일자", "date"]:
            date_col = c
            break
    if date_col is None:
        for c in d.columns:
            try:
                if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.9:
                    date_col = c
                    break
            except Exception:
                pass
    if date_col is None:
        raise ValueError("날짜 열을 찾지 못했습니다. (예: 날짜/일자/date/yyyymmdd)")

    d["날짜"] = d[date_col].map(to_date)
    d = d.dropna(subset=["날짜"]).copy()
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    d["일"] = d["날짜"].dt.day.astype(int)

    # 요일
    if "요일" not in d.columns:
        yo_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        d["요일"] = d["날짜"].dt.dayofweek.map(yo_map)

    # 분류 도움 컬럼(문자열)
    gcol = "구분" if "구분" in d.columns else None

    # 공급량 컬럼 추정(없으면 None)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c
            break

    # 명절 분류(월 기준: 1-2월 설, 9-10월 추석)
    def festival_by_month(m: int) -> Optional[str]:
        if m in (1, 2):
            return "명절_설날"
        if m in (9, 10):
            return "명절_추석"
        return None

    # 카테고리 매핑
    def map_category(row):
        y = row["요일"]
        m = int(row["월"])
        g = str(row.get(gcol, ""))

        # 공휴일/대체공휴일
        if ("공휴" in g) or ("대체" in g):
            return "공휴일_대체"

        # 명절 처리 (엑셀에 '명절(설·추석)'처럼 적혀 있어도 월로 구분)
        if "명절" in g or "설" in g or "추석" in g:
            f = festival_by_month(m)
            if f:
                return f

        # 요일 분류
        if y == "토":
            return "토요일"
        if y == "일":
            return "일요일"
        if y in ["화", "수", "목"]:
            return "평일_1"
        if y in ["월", "금"]:
            return "평일_2"
        return "평일_1"

    d["카테고리"] = d.apply(map_category, axis=1)
    d["카테고리"] = pd.Categorical(d["카테고리"], categories=CATS, ordered=False)
    return d, supply_col

def compute_weights_monthly(
    df: pd.DataFrame, supply_col: Optional[str], base_cat: str = "평일_1", cap_holiday: float = 0.95
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    월별 가중치: 같은 '월'에서 base_cat(평일_1)의 '공급량' 중앙값을 기준으로
    각 카테고리 중앙값 비율(=가중치)을 산정. 데이터 부족은 전체 중앙값/DEFAULT로 보강.
    반환: (월별가중치 DataFrame(index=월), 전역가중치 dict)
    """
    W = []
    for m in range(1, 13):
        sub = df[df["월"] == m]
        if sub.empty:
            W.append(pd.Series({c: np.nan for c in CATS}, name=m))
            continue

        if (supply_col is None) or sub[sub["카테고리"] == base_cat].empty:
            row = {c: (1.0 if c == base_cat else np.nan) for c in CATS}
            W.append(pd.Series(row, name=m))
            continue

        base_med = sub.loc[sub["카테고리"] == base_cat, supply_col].median()
        row = {}
        for c in CATS:
            if c == base_cat:
                row[c] = 1.0
            else:
                s = sub.loc[sub["카테고리"] == c, supply_col]
                row[c] = float(s.median() / base_med) if (len(s) > 0 and base_med > 0) else np.nan
        W.append(pd.Series(row, name=m))
    W = pd.DataFrame(W)  # index=월

    # 전체 중앙값으로 보강 + 휴일/명절 상한
    global_med = {c: (np.nanmedian(W[c].values) if c in W else np.nan) for c in CATS}
    for c in CATS:
        if np.isnan(global_med[c]):
            global_med[c] = DEFAULT_WEIGHTS[c]
    for c in ["공휴일_대체", "명절_설날", "명절_추석"]:
        global_med[c] = min(global_med[c], cap_holiday)

    W_filled = W.fillna(pd.Series(global_med))
    global_w = {c: float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 카테고리 일수와 가중 유효일수 합계를 계산 + 비고 생성"""
    counts = (
        df.pivot_table(index=["연", "월"], columns="카테고리", values="날짜", aggfunc="count")
        .reindex(columns=CATS, fill_value=0)
        .astype(int)
    )
    # 월별 가중치 적용
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values

    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연", "월"])["날짜"].nunique().rename("월일수")

    out = pd.concat([month_days, counts.add_prefix("일수_"), eff_sum], axis=1)
    out = out.reset_index()

    # 비고 생성(설/추석/대체공휴일)
    note = []
    for _, r in out.iterrows():
        items = []
        if r.get("일수_명절_설날", 0) > 0:
            items.append(f"설연휴 {int(r['일수_명절_설날'])}일")
        if r.get("일수_명절_추석", 0) > 0:
            items.append(f"추석연휴 {int(r['일수_명절_추석'])}일")
        if r.get("일수_공휴일_대체", 0) > 0:
            items.append(f"대체공휴일 {int(r['일수_공휴일_대체'])}일")
        note.append(" · ".join(items) if items else "")
    out["비고"] = note

    # 적용 비율
    out["적용_비율(유효/월일수)"] = out["유효일수합"] / out["월일수"]
    return out

def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str, float]):
    """12x31 매트릭스 캘린더(월=열, 일=행)"""
    months = range(1, 13)
    days = range(1, 32)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 31)
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i + 0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

    for x in range(13):
        ax.plot([x, x], [0, 31], color="#D0D5DB", lw=0.8)
    for y in range(32):
        ax.plot([0, 12], [y, y], color="#D0D5DB", lw=0.8)

    for j, m in enumerate(months):
        for i, d in enumerate(days):
            try:
                row = df_year[(df_year["월"] == m) & (df_year["일"] == d)].iloc[0]
            except Exception:
                continue
            cat = row["카테고리"]
            color = PALETTE.get(cat, "#EEEEEE")
            rect = mpl.patches.Rectangle((j, i), 1, 1, color=color, alpha=0.95)
            ax.add_patch(rect)
            label = CAT_SHORT.get(cat, "")
            ax.text(
                j + 0.5,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cat in ["일요일", "공휴일_대체", "명절_설날", "명절_추석"] else "black",
                fontweight="bold",
            )

    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{c}") for c in CATS]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="카테고리")
    plt.tight_layout()
    return fig

# 중앙 배치 + 가운데 정렬로 표 렌더
def show_center_table(df: pd.DataFrame, int_cols: list[str] | None = None, float4_cols: list[str] | None = None, width_ratio=(1, 6, 1)):
    if int_cols is None:
        int_cols = []
    if float4_cols is None:
        float4_cols = []

    fmt: Dict[str, str] = {}
    for c in int_cols:
        if c in df.columns:
            fmt[c] = "{:,.0f}"
    for c in float4_cols:
        if c in df.columns:
            fmt[c] = "{:.4f}"

    sty = (
        df.style.set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center"), ("font-weight", "600")]}])
        .format(fmt)
        .hide(axis="index")
    )

    c1, c2, c3 = st.columns(width_ratio)
    with c2:
        st.table(sty)

# ─────────────────────────────────────────────────────────────
# 사이드바(이전 방식 유지: 좌측에서 컨트롤 + 버튼)
with st.sidebar:
    st.header("예측 기간")
    years = list(range(2026, 2031))
    start_y = st.selectbox("예측 시작(연)", years, index=0)
    start_m = st.selectbox("예측 시작(월)", list(range(1, 13)), index=0)

    end_y = st.selectbox("예측 종료(연)", years, index=1 if len(years) > 1 else 0)
    end_m = st.selectbox("예측 종료(월)", list(range(1, 13)), index=11)

    matrix_year = st.selectbox("매트릭스 표시 연도", years, index=0)

    st.header("데이터 소스")
    file_mode = st.radio("파일 선택", ["Repo 내 엑셀 사용", "파일 업로드"], horizontal=False)
    default_path = Path("data") / "effective_days_calendar.xlsx"
    upload = None
    if file_mode == "Repo 내 엑셀 사용":
        if default_path.exists():
            st.success(f"레포 파일 사용: {default_path.name}")
        else:
            st.warning("레포에 data/effective_days_calendar.xlsx 가 없습니다. 아래에서 업로드 해주세요.")
            upload = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
    else:
        upload = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

    run = st.button("분석 시작", type="primary")

st.title("📅 Effective Days — 유효일수 분석")
st.caption("월별 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 같은 달의 ‘평일_1(화·수·목)’ 중앙값 대비 각 카테고리 중앙값 비율로 산정합니다. (명절/공휴일 가중치는 상한 0.95 적용)")

if not run:
    st.stop()

# ─ 데이터 로드
file_obj = None
if upload is not None:
    file_obj = upload
elif default_path.exists():
    file_obj = open(default_path, "rb")

if file_obj is None:
    st.error("엑셀을 업로드하거나 data/effective_days_calendar.xlsx 를 레포에 넣어주세요.")
    st.stop()

try:
    raw = pd.read_excel(file_obj, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀을 읽는 중 문제가 발생했습니다: {e}")
    st.stop()

# ─ 전처리/가중치
try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# ─ 기간 필터
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts = pd.Timestamp(int(end_y), int(end_m), 1) + pd.offsets.MonthEnd(0)
mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts)
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 예측 구간에 해당하는 날짜가 엑셀에 없습니다. (엑셀에 2026~2030 데이터가 포함되어 있는지 확인)")
    st.stop()

# ─ 히트맵(맨 위)
years_in_range = sorted(pred_df["연"].unique().tolist())
if matrix_year not in years_in_range:
    matrix_year = years_in_range[0]
fig = draw_calendar_matrix(matrix_year, pred_df[pred_df["연"] == matrix_year], W_global)
st.pyplot(fig, clear_figure=True)

# ─ 가중치 요약(가운데 배치)
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({"카테고리": CATS, "전역 가중치(중앙값)": [round(W_global[c], 4) for c in CATS]})
show_center_table(w_show, int_cols=[], float4_cols=["전역 가중치(중앙값)"], width_ratio=(1, 5, 1))

# ─ 월별 유효일수 요약(표)
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly).sort_values(["연", "월"]).reset_index(drop=True)

# 정수/소수 포맷 지정
int_cols = ["연", "월", "월일수"] + [c for c in eff_tbl.columns if c.startswith("일수_")]
float4_cols = ["유효일수합", "적용_비율(유효/월일수)"]
show_center_table(eff_tbl, int_cols=int_cols, float4_cols=float4_cols, width_ratio=(1, 10, 1))

# 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl.to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv",
)
