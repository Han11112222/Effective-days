# app.py — Effective Days (유효일수 분석 전용)

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
st.set_page_config(
    page_title="Effective Days · 유효일수 분석",
    page_icon="📅",
    layout="wide",
)
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
# 카테고리 / 색상 / 라벨
CATS = ["평일_1", "평일_2", "토요일", "일요일", "공휴일_대체", "명절_설날", "명절_추석"]
CAT_LABEL = {
    "평일_1": "평일_1(화·수·목)",
    "평일_2": "평일_2(월·금)",
    "토요일": "토요일",
    "일요일": "일요일",
    "공휴일_대체": "공휴일·대체",
    "명절_설날": "명절_설날",
    "명절_추석": "명절_추석",
}
CAT_SHORT = {
    "평일_1": "평1", "평일_2": "평2", "토요일": "토", "일요일": "일",
    "공휴일_대체": "휴", "명절_설날": "설", "명절_추석": "추"
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
    "평일_1": 1.0, "평일_2": 0.952, "토요일": 0.85, "일요일": 0.60,
    "공휴일_대체": 0.799, "명절_설날": 0.842, "명절_추석": 0.799,
}

# ─────────────────────────────────────────────────────────────
# 유틸
def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")


def normalize_calendar(df: pd.DataFrame):
    """엑셀 원본을 표준 스키마로 정규화하고 (DataFrame, 공급량컬럼명 or None) 반환"""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜
    date_col = None
    for c in d.columns:
        if str(c).lower() in ["날짜", "일자", "date"]:
            date_col = c; break
    if date_col is None:
        for c in d.columns:
            try:
                if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.9:
                    date_col = c; break
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

    # 불리언 힌트 표준화
    for col in ["주중여부", "주말여부", "공휴일여부", "명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
        else:
            d[col] = np.nan

    # 공급량 컬럼 추정(없으면 None)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 카테고리 분류
    def infer_festival(row):
        g = str(row.get("구분", "")).strip()
        mon = int(row["월"])
        # 엑셀에 "명절(설+추석)"처럼 섞여있는 경우 월로 판단
        if "설" in g:
            return "명절_설날"
        if "추" in g:
            return "명절_추석"
        if "명절" in g or str(row.get("명절여부", "")).upper() == "TRUE":
            if mon in (1, 2, 3):   # 상반기 ⇒ 설날로 해석
                return "명절_설날"
            if mon in (8, 9, 10):  # 하반기 ⇒ 추석으로 해석
                return "명절_추석"
        return None

    def map_category(row):
        g, y = str(row.get("구분", "")), row["요일"]
        if ("공휴" in g) or ("대체" in g) or (str(row.get("공휴일여부", "")).upper() == "TRUE"):
            return "공휴일_대체"
        fest = infer_festival(row)
        if fest: return fest
        if y == "토": return "토요일"
        if y == "일": return "일요일"
        if y in ["화", "수", "목"]: return "평일_1"
        if y in ["월", "금"]: return "평일_2"
        return "평일_1"

    d["카테고리"] = d.apply(map_category, axis=1)
    d["카테고리"] = pd.Categorical(d["카테고리"], categories=CATS, ordered=False)
    return d, supply_col


def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    월별 가중치: 같은 '월'에서 base_cat(평일_1)의 '공급량' 중앙값을 기준으로
    각 카테고리 중앙값 비율(=가중치)을 산정. 데이터 부족은 전체 중앙값/DEFAULT로 보강.
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
    """월별 카테고리 일수와 가중 유효일수 합계를 계산"""
    counts = df.pivot_table(
        index=["연", "월"], columns="카테고리", values="날짜",
        aggfunc="count"
    ).reindex(columns=CATS, fill_value=0).astype(int)

    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values

    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연", "월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"] / out["월일수"]).round(4)

    out = out.reset_index()

    # 비고(설/추/대체공휴일) 자동 설명
    def note(row):
        msgs = []
        if row.get("일수_명절_설날", 0) > 0:
            msgs.append(f"설연휴 {int(row['일수_명절_설날'])}일 반영")
        if row.get("일수_명절_추석", 0) > 0:
            msgs.append(f"추석연휴 {int(row['일수_명절_추석'])}일 반영")
        if row.get("일수_공휴일_대체", 0) > 0:
            msgs.append(f"대체공휴일 {int(row['일수_공휴일_대체'])}일")
        return " · ".join(msgs)

    out["비고"] = out.apply(note, axis=1)
    return out


def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str, float]):
    """12x31 매트릭스 캘린더(월=열, 일=행)"""
    months = range(1, 13)
    days = range(1, 32)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 31)
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i + 0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=12)

    # 그리드
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
                j + 0.5, i + 0.5, label, ha="center", va="center",
                fontsize=9,
                color="white" if cat in ["일요일", "공휴일_대체", "명절_설날", "명절_추석"] else "black",
                fontweight="bold"
            )

    # 범례(가중치 함께)
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_LABEL[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 상단 컨트롤(2026~2030)
st.title("Effective Days — 유효일수 분석")
st.caption("월별 유효일수 = Σ(해당일 카테고리 가중치). "
           "가중치는 같은 달의 ‘평일_1(화·수·목)’ 중앙값 대비 각 카테고리 중앙값 **비율**로 산정합니다. "
           "(명절/공휴일 가중치 상한 0.95 적용)")

years_opts = list(range(2026, 2031))
months_opts = list(range(1, 13))

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
with c1:
    start_y = st.selectbox("시작 연", years_opts, index=0, key="start_y")
with c2:
    start_m = st.selectbox("시작 월", months_opts, index=0, key="start_m")
with c3:
    end_y = st.selectbox("종료 연", years_opts, index=1 if len(years_opts) > 1 else 0, key="end_y")
with c4:
    end_m = st.selectbox("종료 월", months_opts, index=11, key="end_m")
with c5:
    view_year = st.selectbox("매트릭스 표시 연도", years_opts, index=0, key="view_year")

# 데이터 로드
default_path = Path("data") / "effective_days_calendar.xlsx"
file = default_path if default_path.exists() else None
if file is None:
    file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
if file is None:
    st.warning("엑셀을 업로드하거나 repo의 data/effective_days_calendar.xlsx를 넣어주세요.")
    st.stop()

try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception:
    st.error("엑셀을 읽는 중 문제가 발생했습니다.")
    st.stop()

try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# 가중치 계산(학습 데이터 전체에서 산정)
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# ─ 매트릭스 (맨 위에 표시)
year_df = base_df[base_df["연"] == int(view_year)]
if year_df.empty:
    st.info(f"{view_year}년 데이터가 엑셀에 없습니다.")
else:
    fig = draw_calendar_matrix(view_year, year_df, W_global)
    st.pyplot(fig, clear_figure=True)

# ─ 가중치 요약(텍스트)
weights_line = "  ".join([f"{CAT_LABEL[c]} {W_global[c]:.4f}" for c in CATS])
st.markdown(
    f"**카테고리 전역 가중치(중앙값)**  {weights_line}  \n"
    "※ 가중치는 해당 달 ‘평일_1’ 대비 각 카테고리 중앙값 비율을 다시 중앙값으로 취합한 값입니다. "
    "데이터가 부족한 달은 기본값/전체중앙값으로 보강되며, 명절/공휴일은 보수적으로 0.95 상한을 둡니다."
)

# ─ 예측 구간 필터 (종료연 반영 확실히)
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts = (pd.Timestamp(int(end_y), int(end_m), 1) + pd.offsets.MonthEnd(1))
mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts)
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 예측 구간에 해당하는 날짜가 엑셀에 없습니다. (2026~2030 범위로 선택하세요)")
    st.stop()

# ─ 월별 유효일수 테이블
eff_tbl = effective_days_by_month(pred_df, W_monthly)

# 표시용 컬럼 정리(인덱스 0~11 제거 = hide_index=True)
show_cols = (
    ["연", "월", "월일수"]
    + [f"일수_{c}" for c in CATS]
    + ["유효일수합", "적용_비율(유효/월일수)", "비고"]
)
st.subheader("월별 유효일수 요약")
st.dataframe(
    eff_tbl[show_cols].sort_values(["연", "월"]).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# ─ CSV 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl[show_cols].sort_values(["연", "월"]).to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv",
)

# ─ 설명 블록(계산 방식 Q&A)
with st.expander("가중치/명절 처리 방식(간단 설명)", expanded=False):
    st.markdown(
        """
- **가중치 산정**  
  같은 **월** 안에서 ‘평일_1(화·수·목)’의 공급량 **중앙값**을 1.0으로 두고, 다른 날 유형의 중앙값을 나누어 **비율**로 만듭니다.  
  이렇게 얻은 월별 비율들을 다시 중앙값으로 요약하여 전역 가중치로 사용합니다. (명절/공휴일은 **0.95 상한**)

- **표의 예: 추석 가중치 0.48**  
  이는 *추석이 있는 달(주로 9~10월)*에서 ‘평일_1’ **대비** 추석 태그 날의 **중앙값 비율**을 모아 중앙값으로 요약한 값입니다.  
  데이터가 부족한 달은 기본/전체중앙값으로 보강됩니다.

- **명절 연휴(설/추석) 처리**  
  엑셀에 `명절(설+추석)`처럼 기재되어도, **1~3월 ⇒ 설날**, **8~10월 ⇒ 추석**으로 자동 인식합니다.  
  예: 2026-02 **2/14~2/18**은 ‘설연휴 5일’로 처리, 2026-09 **9/24~9/27**은 ‘추석연휴 4일’로 처리합니다.  
  연휴가 주말과 겹치더라도, 예측의 보수성 확보를 위해 **명절 태그 전체를 명절 가중치**로 반영합니다.
        """
    )
