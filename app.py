# -*- coding: utf-8 -*-
# Effective Days — 유효일수 분석 전용

import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st


# ─────────────────────────────────────────────
# 기본 세팅
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


# ─────────────────────────────────────────────
# 한글 폰트 세팅 (Matplotlib)
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic.ttf",
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.family"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()

# ─────────────────────────────────────────────
# 상수/팔레트
CATS = ["평일_1", "평일_2", "토요일", "일요일", "공휴일_대체", "명절_설날", "명절_추석"]
CAT_SHORT = {"평일_1": "평1", "평일_2": "평2", "토요일": "토", "일요일": "일",
             "공휴일_대체": "휴", "명절_설날": "설", "명절_추석": "추"}

PALETTE = {
    "평일_1": "#7DC3C1",      # teal light
    "평일_2": "#3DA4AB",      # teal
    "토요일": "#5D6D7E",      # slate
    "일요일": "#34495E",      # deep slate
    "공휴일_대체": "#E57373",  # soft red
    "명절_설날": "#F5C04A",    # warm gold
    "명절_추석": "#F39C12",    # amber
}

DEFAULT_WEIGHTS = {
    "평일_1": 1.0, "평일_2": 0.952, "토요일": 0.85, "일요일": 0.60,
    "공휴일_대체": 0.799, "명절_설날": 0.842, "명절_추석": 0.799,
}


# ─────────────────────────────────────────────
# 공용 함수(표 스타일 → 항상 가운데 정렬 & 중앙 배치)
def _fmt4(v):
    if isinstance(v, (int, float, np.floating)) and pd.notna(v):
        return f"{v:.4f}"
    return v if v is not None else ""


def center_table(df: pd.DataFrame, width_px: int = 900, hide_index: bool = True):
    styler = (
        df.style
        .format(_fmt4)
        .set_table_styles([
            {"selector": "th", "props": "text-align:center; font-weight:600;"},
            {"selector": "td", "props": "text-align:center;"},
            {"selector": "table", "props": f"margin-left:auto; margin-right:auto; width:{width_px}px; border-collapse:collapse;"},
        ])
    )
    if hide_index:
        styler = styler.hide(axis="index")
    # 중요: CSS가 텍스트로 보이지 않게 unsafe_allow_html=True 로 렌더
    st.markdown(styler.to_html(), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 전처리
def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")


def normalize_calendar(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 열 찾기
    date_col = None
    for c in d.columns:
        if str(c).lower() in ["날짜", "일자", "date", "yyyymmdd"]:
            date_col = c
            break
    if date_col is None:
        for c in d.columns:
            try:
                # 8자리 yyyymmdd 숫자 비율이 큰 열 찾기
                if (d[c].astype(str).str.match(r"^\d{8}$", na=False)).mean() > 0.7:
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

    # 요일 없으면 생성
    if "요일" not in d.columns:
        yo_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        d["요일"] = d["날짜"].dt.dayofweek.map(yo_map)

    # 공급량 컬럼(있으면 사용)
    supply_col = None
    for c in d.columns:
        if ("공급" in c) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c
            break

    # 카테고리 매핑
    def infer_festival(row):
        g = str(row.get("구분", ""))
        m = int(row["월"])
        if "설" in g:
            return "명절_설날"
        if "추" in g:
            return "명절_추석"
        if "명절" in g or "설·추석" in g or "설추석" in g:
            if m in (1, 2):
                return "명절_설날"
            if m in (9, 10):
                return "명절_추석"
            # 그 외는 일단 추석 처리
            return "명절_추석"
        return None

    def map_category(row):
        g = str(row.get("구분", ""))
        y = row["요일"]
        if ("공휴" in g) or ("대체" in g):
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


# ─────────────────────────────────────────────
# 가중치 계산
def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    같은 '월' 안에서 base_cat(평일_1)의 '공급량' 중앙값을 기준으로
    각 카테고리 중앙값 비율(=가중치)을 산정.
    데이터 부족은 전체 중앙값/기본값으로 보강. 공휴/명절 상한 0.95 적용.
    """
    W_rows = []
    for m in range(1, 13):
        sub = df[df["월"] == m]
        if sub.empty:
            W_rows.append(pd.Series({c: np.nan for c in CATS}, name=m))
            continue

        if (supply_col is None) or sub[sub["카테고리"] == base_cat].empty:
            row = {c: (1.0 if c == base_cat else np.nan) for c in CATS}
            W_rows.append(pd.Series(row, name=m))
            continue

        base_med = sub.loc[sub["카테고리"] == base_cat, supply_col].median()
        r = {}
        for c in CATS:
            if c == base_cat:
                r[c] = 1.0
            else:
                s = sub.loc[sub["카테고리"] == c, supply_col]
                r[c] = float(s.median() / base_med) if (len(s) > 0 and base_med > 0) else np.nan
        W_rows.append(pd.Series(r, name=m))

    W = pd.DataFrame(W_rows)  # index=월
    # 전체 중앙값으로 보강
    fill = {}
    for c in CATS:
        med = np.nanmedian(W[c].values) if c in W else np.nan
        if np.isnan(med):
            med = DEFAULT_WEIGHTS[c]
        # 상한 적용
        if c in ["공휴일_대체", "명절_설날", "명절_추석"]:
            med = min(med, cap_holiday)
        fill[c] = float(med)

    W_filled = W.fillna(pd.Series(fill))
    global_w = {c: float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w


# ─────────────────────────────────────────────
# 유효일수 계산
def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    # 월별 카테고리 일수(카운트)
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

    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)

    # 비고(설/추석/대체공휴일 요약)
    notes = []
    for (y, m), row in counts.iterrows():
        parts = []
        if row.get("명절_설날", 0) > 0:
            parts.append(f"설연휴 {int(row['명절_설날'])}일 반영")
        if row.get("명절_추석", 0) > 0:
            parts.append(f"추석연휴 {int(row['명절_추석'])}일 반영")
        if row.get("공휴일_대체", 0) > 0:
            parts.append(f"대체공휴일 {int(row['공휴일_대체'])}일")
        notes.append("; ".join(parts) if parts else "")
    out["비고"] = notes

    out["적용_비율(유효/월일수)"] = (out["유효일수합"] / out["월일수"]).round(4)
    return out.reset_index()


# ─────────────────────────────────────────────
# 캘린더 매트릭스 (월=열, 일=행)
def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str, float]):
    months = range(1, 13)
    days = range(1, 32)

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 31)
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=18, pad=12)

    # 격자
    for x in range(13):
        ax.plot([x, x], [0, 31], color="#D0D5DB", lw=0.8)
    for y in range(32):
        ax.plot([0, 12], [y, y], color="#D0D5DB", lw=0.8)

    # 칠하기 + 라벨
    for j, m in enumerate(months):
        for i, d in enumerate(days):
            row = df_year[(df_year["월"] == m) & (df_year["일"] == d)]
            if row.empty:
                continue
            row = row.iloc[0]
            cat = row["카테고리"]
            color = PALETTE.get(cat, "#EEEEEE")
            rect = mpl.patches.Rectangle((j, i), 1, 1, color=color, alpha=0.96)
            ax.add_patch(rect)
            label = CAT_SHORT.get(cat, "")
            ax.text(
                j + 0.5, i + 0.5, label, ha="center", va="center",
                fontsize=10,
                color=("white" if cat in ["일요일", "공휴일_대체", "명절_설날", "명절_추석"] else "black"),
                fontweight="bold",
            )

    # 범례(전역 가중치 표기)
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{c} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="카테고리 (가중치)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 사이드바 (입력)
with st.sidebar:
    st.header("예측 기간")
    years = list(range(2026, 2031))  # 2026~2030
    start_y = st.selectbox("예측 시작(연)", years, index=0, key="start_y")
    start_m = st.selectbox("예측 시작(월)", list(range(1, 13)), index=0, key="start_m")
    end_y = st.selectbox("예측 종료(연)", years, index=1, key="end_y")
    end_m = st.selectbox("예측 종료(월)", list(range(1, 13)), index=11, key="end_m")
    view_y = st.selectbox("매트릭스 표시 연도", years, index=0, key="view_y")

    # 데이터 가져오기
    st.divider()
    st.caption("데이터 소스")
    src = st.radio("파일 선택", ["Repo 내 엑셀 사용", "파일 업로드"], index=0, horizontal=False)
    default_path = Path("data") / "effective_days_calendar.xlsx"
    if src == "Repo 내 엑셀 사용":
        if default_path.exists():
            file = open(default_path, "rb")
            st.success(f"레포 파일 사용: {default_path.name}")
        else:
            file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
    else:
        file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

    run = st.button("분석 시작", type="primary")


# ─────────────────────────────────────────────
# 본문
st.title("📅 Effective Days — 유효일수 분석")
st.caption(
    "월별 유효일수 = Σ(해당일 카테고리 가중치). "
    "가중치는 같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값 대비 각 카테고리 중앙값 비율로 산정합니다. "
    "데이터가 부족하면 전역 중앙값(기본값)으로 보강하며 공휴/명절 가중치는 상한 0.95를 둡니다."
)

if not run:
    st.stop()

if file is None:
    st.warning("엑셀을 업로드하거나 data/effective_days_calendar.xlsx 를 레포에 넣어주세요.")
    st.stop()

# 데이터 읽기/정규화
try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception:
    st.error("엑셀 파일을 읽는 중 문제가 발생했습니다.")
    st.stop()

try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# 기간 검사 & 필터
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts = pd.Timestamp(int(end_y), int(end_m), 1)
if end_ts < start_ts:
    st.error("예측 종료가 시작보다 빠릅니다.")
    st.stop()

mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 예측 구간에 해당하는 날짜가 엑셀에 없습니다. (엑셀에 2026~2030 데이터 포함 확인)")
    st.stop()

# 가중치 계산
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 매트릭스(상단)
if view_y not in pred_df["연"].unique():
    st.info(f"선택한 매트릭스 연도({view_y})가 예측 구간에 없습니다. 가장 가까운 연도로 표시합니다.")
    cand = sorted(pred_df["연"].unique())
    view_year = cand[0]
else:
    view_year = view_y

fig = draw_calendar_matrix(view_year, base_df[base_df["연"] == view_year], W_global)
st.pyplot(fig, clear_figure=True)

# 전역 가중치 요약 표
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({"카테고리": CATS, "전역 가중치(중앙값)": [W_global[c] for c in CATS]})
center_table(w_show, width_px=620, hide_index=True)

# 월별 유효일수 표
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly)

# 표시 컬럼 (숫자/텍스트 모두 가운데 정렬)
show_cols = (
    ["연", "월", "월일수"]
    + [f"일수_{c}" for c in CATS]
    + ["유효일수합", "적용_비율(유효/월일수)", "비고"]
)
view_df = eff_tbl[show_cols].sort_values(["연", "월"]).reset_index(drop=True)

center_table(view_df, width_px=1180, hide_index=True)

# 파일 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=view_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv",
)
