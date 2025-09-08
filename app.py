# app.py — Effective Days (유효일수 분석 전용 · 2026~2030)
# by hanyoub + ChatGPT

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 기본 세팅
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 약간의 CSS (표/레이아웃 정갈하게 + 가운데 정렬)
st.markdown(
    """
    <style>
    /* 본문 폭을 너무 넓지 않게 조정 */
    .block-container {max-width: 1280px;}
    /* 모든 표 글자 가운데 정렬 */
    table.dataframe, .tbl-wrap table {margin: 0 auto; }
    table.dataframe th, table.dataframe td {text-align:center !important;}
    .tbl-wrap {max-width: 1100px; margin: 0 auto;}
    /* select 크기 살짝 컴팩트 */
    div[data-baseweb="select"] {min-width: 160px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 한글 폰트(가능하면 나눔/맑은고딕)
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
CAT_DESC = {
    "평일_1": "화·수·목",
    "평일_2": "월·금",
    "토요일": "토",
    "일요일": "일",
    "공휴일_대체": "법정 공휴일/대체휴일",
    "명절_설날": "설 연휴",
    "명절_추석": "추석 연휴",
}
CAT_SHORT = {"평일_1": "평1", "평일_2": "평2", "토요일": "토", "일요일": "일", "공휴일_대체": "휴", "명절_설날": "설", "명절_추석": "추"}
PALETTE = {
    "평일_1": "#7DC3C1",
    "평일_2": "#3DA4AB",
    "토요일": "#5D6D7E",
    "일요일": "#34495E",
    "공휴일_대체": "#E57373",
    "명절_설날": "#F5C04A",
    "명절_추석": "#F39C12",
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

def show_table(df: pd.DataFrame, note: str | None = None):
    """가운데 정렬 HTML로 출력"""
    sty = (
        df.style
        .set_table_styles([dict(selector="th", props="text-align:center;")])
        .set_properties(**{"text-align": "center"})
        .format(precision=4)
    )
    html = f"<div class='tbl-wrap'>{sty.to_html(index=False)}</div>"
    st.markdown(html, unsafe_allow_html=True)
    if note:
        st.caption(note)

def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame):
    """엑셀 원본을 표준 스키마로 정규화하고 (DataFrame, 공급량 컬럼명 or None) 반환"""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 컬럼 찾기
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

    if "요일" not in d.columns:
        yo_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        d["요일"] = d["날짜"].dt.dayofweek.map(yo_map)

    for col in ["주중여부", "주말여부", "공휴일여부", "명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
        else:
            d[col] = np.nan

    # 공급량(있다면) 추정
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c
            break

    # 카테고리 분류
    def infer_festival(row):
        g = str(row.get("구분", ""))
        mon = int(row["월"])
        # 파일에 "명절(설·추석)" 식으로 합쳐져 있어도 월로 분기
        if "설" in g:
            return "명절_설날"
        if "추" in g:
            return "명절_추석"
        if str(row.get("명절여부", "")).upper() == "TRUE":
            if mon in (1, 2):
                return "명절_설날"
            if mon in (9, 10):
                return "명절_추석"
        return None

    def map_category(row):
        g, y = str(row.get("구분", "")), row["요일"]
        if ("공휴" in g) or ("대체" in g) or (str(row.get("공휴일여부", "")).upper() == "TRUE"):
            return "공휴일_대체"
        fest = infer_festival(row)
        if fest:
            return fest
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

def apply_special_overrides(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], str]]:
    """
    예외적으로 '명절로 다 묶어야 하는 연휴'를 강제 지정.
    - 2026-02-14 ~ 2026-02-18 → 설
    - 2026-09-24 ~ 2026-09-27 → 추석
    반환: (수정된 df, { (연,월): "비고" })
    """
    note: Dict[Tuple[int, int], str] = {}

    def mark_range(start: str, end: str, cat: str, label: str):
        nonlocal note, df
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        m = df["날짜"].between(s, e)
        if m.any():
            df.loc[m, "카테고리"] = cat
            key = (int(s.year), int(s.month))
            added = int(m.sum())
            prev = note.get(key, "")
            add_txt = f"{label} {added}일 반영"
            note[key] = (prev + ("; " if prev else "") + add_txt)

    # 필요에 따라 여기에 케이스 추가 가능
    mark_range("2026-02-14", "2026-02-18", "명절_설날", "설연휴")
    mark_range("2026-09-24", "2026-09-27", "명절_추석", "추석연휴")

    return df, note

def compute_weights_monthly(
    df: pd.DataFrame, supply_col: Optional[str], base_cat: str = "평일_1", cap_holiday: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    같은 '월'에서 base_cat(평일_1)의 공급량 중앙값을 기준으로 카테고리 중앙값 비율을 가중치로 산정.
    공급량이 없으면 DEFAULT 사용.
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
    W = pd.DataFrame(W)

    global_med = {c: (np.nanmedian(W[c].values) if c in W else np.nan) for c in CATS}
    for c in CATS:
        if np.isnan(global_med[c]):
            global_med[c] = DEFAULT_WEIGHTS[c]
    for c in ["공휴일_대체", "명절_설날", "명절_추석"]:
        global_med[c] = min(global_med[c], cap_holiday)

    W_filled = W.fillna(pd.Series(global_med))
    global_w = {c: float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame, month_notes: Dict[Tuple[int,int], str]) -> pd.DataFrame:
    """월별 카테고리 일수·유효일수·비고 계산"""
    counts = (
        df.pivot_table(index=["연", "월"], columns="카테고리", values="날짜", aggfunc="count")
        .reindex(columns=CATS, fill_value=0)
        .astype(int)
    )
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values

    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연", "월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"] / out["월일수"]).round(4)

    # 비고 추가
    out = out.reset_index()
    notes = []
    for _, r in out[["연", "월"]].iterrows():
        notes.append(month_notes.get((int(r["연"]), int(r["월"])), ""))
    out["비고"] = notes
    return out

def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str, float]):
    """12x31 라벨 매트릭스"""
    months = range(1, 13)
    days = range(1, 32)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 31)
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=8)
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
                fontsize=8.5,
                color="white" if cat in ["일요일", "공휴일_대체", "명절_설날", "명절_추석"] else "black",
                fontweight="bold",
            )

    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{c} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# 상단 컨트롤 (예측 범위: 2026~2030)
st.title("📅 Effective Days — 유효일수 분석")
st.caption(
    "월별 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 **같은 달의 ‘평일_1(화·수·목)’ 중앙값 대비** 각 카테고리 중앙값 비율로 산정합니다. "
    "(명절/공휴일 가중치 상한 0.95 적용)"
)

cols = st.columns([1, 1, 1, 1, 1])
with cols[0]:
    start_year = st.selectbox("시작 연", list(range(2026, 2031)), index=0)
with cols[1]:
    start_month = st.selectbox("시작 월", list(range(1, 13)), index=0)
with cols[2]:
    end_year = st.selectbox("종료 연", list(range(2026, 2031)), index=0)
with cols[3]:
    end_month = st.selectbox("종료 월", list(range(1, 13)), index=11)
with cols[4]:
    matrix_year = st.selectbox("매트릭스 표시 연도", list(range(2026, 2031)), index=0)

# 데이터 로드(레포 파일 기본, 없으면 업로드)
default_path = Path("data") / "effective_days_calendar.xlsx"
file = None
if default_path.exists():
    file = open(default_path, "rb")
else:
    st.info("레포에 data/effective_days_calendar.xlsx 가 없으면 여기로 업로드하세요.")
    file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

if file is None:
    st.stop()

try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception:
    st.error("엑셀을 읽는 중 문제가 발생했습니다.")
    st.stop()

# 전처리/학습
try:
    df, supply_col = normalize_calendar(raw)
    df, month_notes_override = apply_special_overrides(df)  # 특수 연휴 강제 반영
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

W_monthly, W_global = compute_weights_monthly(df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 상단에 가중치 요약(설명 포함)
w_table = pd.DataFrame(
    {
        "카테고리": [f"{c} ({CAT_DESC[c]})" if c in CAT_DESC else c for c in CATS],
        "전역 가중치(중앙값)": [round(W_global[c], 4) for c in CATS],
    }
)
st.subheader("카테고리 가중치 요약")
show_table(
    w_table,
    note="※ 가중치는 **같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값** 대비 각 카테고리 중앙값 비율입니다. "
         "데이터가 부족한 달은 전체 중앙값/기본값으로 보강되며, 명절·공휴일 가중치는 0.95 상한을 둡니다."
)

# 선택 구간 필터
start_ts = pd.Timestamp(int(start_year), int(start_month), 1)
end_ts = pd.Timestamp(int(end_year), int(end_month), 1)
if end_ts < start_ts:
    st.error("종료가 시작보다 이전입니다. 범위를 다시 선택하세요.")
    st.stop()

mask = (df["날짜"] >= start_ts) & (df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 예측 구간에 해당하는 날짜가 엑셀에 없습니다. (미래 연도 2026~2030 포함 여부 확인)")
    st.stop()

# 월별 유효일수 표
eff_tbl = effective_days_by_month(pred_df, W_monthly, month_notes_override)
order_cols = (
    ["연", "월", "월일수"]
    + [f"일수_{c}" for c in CATS]
    + ["유효일수합", "적용_비율(유효/월일수)", "비고"]
)
eff_tbl = eff_tbl[order_cols].sort_values(["연", "월"])

st.subheader("월별 유효일수 요약")
show_table(
    eff_tbl,
    note="비고 예시) ‘설연휴 5일 반영’, ‘추석연휴 4일 반영’ 등. "
         "연휴가 주말과 겹치더라도 본 도구에서는 **명절 기간 전체를 보수적으로 명절 가중치**로 계산합니다."
)

st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl.to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv",
)

# 매트릭스(상단으로 노출)
st.subheader("유효일수 카테고리 매트릭스")
if matrix_year not in pred_df["연"].unique():
    st.info(f"{matrix_year}년은 현재 선택한 예측 구간에 포함되지 않습니다. (매트릭스는 선택 구간 내 연도만 시각화)")
else:
    fig = draw_calendar_matrix(matrix_year, pred_df[pred_df["연"] == matrix_year], W_global)
    st.pyplot(fig, clear_figure=True)

# 계산 로직 설명(간단)
with st.expander("가중치·연휴 처리 간단 설명"):
    st.markdown(
        """
        - **가중치 산정**: 같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값을 기준(=1.0)으로 삼고,  
          각 카테고리의 중앙값/기준 중앙값 비율을 가중치로 사용합니다.  
          공급량 데이터가 부족한 달은 전체 중앙값/기본값으로 보정하며, 명절·공휴일은 0.95 상한을 둡니다.
        - **명절 특수 처리(보수적)**: 2026년 **2/14~2/18은 설**, **9/24~9/27은 추석**으로 강제 분류하여  
          주말이 끼어 있어도 해당 기간 전체를 명절 가중치로 계산합니다.  
          (필요 시 `apply_special_overrides()`에 날짜 구간을 추가하면 됩니다.)
        - **표시 예시**: 2026년 2월 ‘설연휴 5일 반영’, 9월 ‘추석연휴 4일 반영’ 같은 식으로 비고에 기입합니다.
        """
    )
