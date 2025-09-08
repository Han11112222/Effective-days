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
import streamlit.components.v1 as components


# ─────────────────────────────────────────────────────────────
# 기본 세팅
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")


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
# 상수·팔레트
CATS = ["평일_1", "평일_2", "토요일", "일요일", "공휴일_대체", "명절_설날", "명절_추석"]
CAT_LABEL = {
    "평일_1": "평일_1(화·수·목)",
    "평일_2": "평일_2(월·금)",
    "토요일": "토요일",
    "일요일": "일요일",
    "공휴일_대체": "공휴일·대체",
    "명절_설날": "명절(설)",
    "명절_추석": "명절(추석)",
}
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
    "평일_1": 1.00,
    "평일_2": 0.952,
    "토요일": 0.85,
    "일요일": 0.60,
    "공휴일_대체": 0.799,
    "명절_설날": 0.842,
    "명절_추석": 0.799,
}


# ─────────────────────────────────────────────────────────────
# 유틸

def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")


def normalize_calendar(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    • 날짜(연/월/일/요일) 만들고
    • '구분'을 읽어 카테고리 분류
    • 공급량 컬럼 추정 (없어도 동작)
    """
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

    # 불리언 힌트 표준화 (있으면 이용)
    for col in ["공휴일여부", "명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
        else:
            d[col] = np.nan

    # 공급량 컬럼 추정
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c
            break

    # 명절 분류 보조
    def infer_festival(row):
        g = str(row.get("구분", "")).replace(" ", "")
        mon = int(row["월"])
        # 엑셀에 '명절(설•추석)' 같은 패턴이 있으므로 월로 보정
        if "명절" in g or str(row.get("명절여부", "")).upper() == "TRUE":
            if mon in (1, 2):  # 보통 설
                return "명절_설날"
            if mon in (9, 10):  # 보통 추석
                return "명절_추석"
        if "설" in g:
            return "명절_설날"
        if "추석" in g:
            return "명절_추석"
        return None

    # 카테고리 매핑
    def map_category(row):
        g = str(row.get("구분", "")).replace(" ", "")
        y = row["요일"]

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


def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    월별 가중치: 같은 '월' 내 base_cat(평일_1)의 공급량 중앙값을 기준으로
    각 카테고리 중앙값 비율(=가중치)을 산정.
    데이터가 부족하면 전체 중앙값/DEFAULT로 보강하며,
    공휴/명절 가중치는 최대 cap_holiday로 캡.
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
        row = {}
        for c in CATS:
            if c == base_cat:
                row[c] = 1.0
            else:
                s = sub.loc[sub["카테고리"] == c, supply_col]
                row[c] = float(s.median() / base_med) if (len(s) > 0 and base_med > 0) else np.nan
        W_rows.append(pd.Series(row, name=m))

    W = pd.DataFrame(W_rows)  # index=월

    # 보강 + 상한
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
    """
    월별 카테고리 일수, 적용 가중치, 유효일수합, 비고(설/추석/대체공휴일 설명) 계산
    """
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
    out["적용_비율(유효/월일수)"] = (out["유효일수합"] / out["월일수"]).round(4)

    # 비고
    notes = []
    for (yy, mm), row in counts.iterrows():
        note_parts = []
        if row["명절_설날"] > 0:
            note_parts.append(f"설연휴 {int(row['명절_설날'])}일 반영")
        if row["명절_추석"] > 0:
            note_parts.append(f"추석연휴 {int(row['명절_추석'])}일 반영")
        if row["공휴일_대체"] > 0:
            note_parts.append(f"대체공휴일 {int(row['공휴일_대체'])}일")
        notes.append("; ".join(note_parts) if note_parts else "")
    out["비고"] = notes

    out = out.reset_index()
    return out


def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str, float]):
    """월=열 / 일=행 매트릭스 라벨 + 범례"""
    months = range(1, 13)
    days = range(1, 32)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 31)
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=18, pad=10)

    # 그리드
    for x in range(13):
        ax.plot([x, x], [0, 31], color="#D0D5DB", lw=0.8)
    for y in range(32):
        ax.plot([0, 12], [y, y], color="#D0D5DB", lw=0.8)

    short = {"평일_1": "평1", "평일_2": "평2", "토요일": "토", "일요일": "일", "공휴일_대체": "휴", "명절_설날": "설", "명절_추석": "추"}

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
            label = short.get(cat, "")
            ax.text(
                j + 0.5,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cat in ["일요일", "공휴일_대체", "명절_설날", "명절_추석"] else "black",
                fontweight="bold",
            )

    # 범례
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_LABEL[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False, title="카테고리(가중치)")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 표 렌더링(가운데 정렬, components.html 사용) — markdown에서 스타일이 노출되는 이슈 방지
def center_table(df: pd.DataFrame, width_px: int = 1100, height_px: Optional[int] = None):
    styler = (
        df.style.set_table_styles(
            [
                {"selector": "th", "props": "text-align:center; padding:6px;"},
                {"selector": "td", "props": "text-align:center; padding:6px;"},
            ]
        ).hide_index()
    )
    html_table = styler.to_html()
    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          .tbl-wrap {{
            max-width: {width_px}px;
            margin: 0 auto;
          }}
        </style>
      </head>
      <body>
        <div class="tbl-wrap">{html_table}</div>
      </body>
    </html>
    """
    if height_px is None:
        height_px = min(700, 42 * (len(df) + 3))
    components.html(html, height=height_px, scrolling=True)


# ─────────────────────────────────────────────────────────────
# 사이드바 — 기간 선택 + 분석 시작
with st.sidebar:
    st.subheader("예측 기간")
    years = list(range(2026, 2031))  # 2026~2030
    c1, c2 = st.columns(2)
    with c1:
        start_y = st.selectbox("예측 시작(연)", years, index=0)
    with c2:
        start_m = st.selectbox("예측 시작(월)", list(range(1, 13)), index=0)

    c3, c4 = st.columns(2)
    with c3:
        end_y = st.selectbox("예측 종료(연)", years, index=1 if len(years) > 1 else 0)
    with c4:
        end_m = st.selectbox("예측 종료(월)", list(range(1, 13)), index=11)

    matrix_year = st.selectbox("매트릭스 표시 연도", years, index=0)
    run_btn = st.button("분석 시작", type="primary")

st.title("📅 Effective Days — 유효일수 분석")
st.caption(
    "월별 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 **같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값** 대비 각 카테고리 중앙값 비율로 산정합니다. "
    "데이터가 부족하면 전체 중앙값/기본값으로 보강하며 공휴/명절 가중치는 0.95 상한을 둡니다."
)

if not run_btn:
    st.stop()

# 데이터 로드
default_path = Path("data") / "effective_days_calendar.xlsx"
file = None
if default_path.exists():
    file = open(default_path, "rb")
else:
    file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

if file is None:
    st.warning("엑셀을 업로드하거나 data/effective_days_calendar.xlsx 를 레포에 넣어주세요.")
    st.stop()

try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀 읽기 오류: {e}")
    st.stop()

# 전처리
try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# 가중치 산정
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 예측 기간 필터
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts = pd.Timestamp(int(end_y), int(end_m), 1)
if end_ts < start_ts:
    st.error("예측 종료가 시작보다 빠릅니다.")
    st.stop()

mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택 기간에 해당하는 날짜가 엑셀에 없습니다. (2026~ 이후 데이터 포함 확인)")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 1) 매트릭스 (맨 위)
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame(
    {"카테고리": [CAT_LABEL[c] for c in CATS], "전역 가중치(중앙값)": [round(W_global[c], 4) for c in CATS]}
)
center_table(w_show, width_px=700, height_px=360)

st.markdown(
    f"**가중치 산정 메모** — 예: ‘명절(추석) {W_global['명절_추석']:.3f}’은 **같은 달**의 평일_1 중앙값 대비 "
    "명절(추석) 중앙값 비율로 계산됩니다(데이터 부족 시 전체 중앙값/기본값 보강). 따라서 특정 해의 명절 위치가 달라져도, "
    "가중치는 ‘월 효과’를 반영합니다."
)

st.subheader(f"{matrix_year}년 매트릭스")
fig = draw_calendar_matrix(matrix_year, base_df[base_df["연"] == matrix_year], W_global)
st.pyplot(fig, clear_figure=True)

# ─────────────────────────────────────────────────────────────
# 2) 월별 유효일수 테이블
eff_tbl = effective_days_by_month(pred_df, W_monthly)
# 보기용 컬럼 순서/라벨
show_cols = (
    ["연", "월", "월일수"]
    + [f"일수_{c}" for c in CATS]
    + ["유효일수합", "적용_비율(유효/월일수)", "비고"]
)
eff_show = eff_tbl[show_cols].copy()
eff_show.columns = (
    ["연", "월", "월일수"]
    + [f"일수_{CAT_LABEL[c]}" for c in CATS]
    + ["유효일수합", "적용_비율(유효/월일수)", "비고"]
)

st.subheader("월별 유효일수 요약")
center_table(eff_show, width_px=1250)

# 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl.sort_values(["연", "월"]).to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv",
)
