# -*- coding: utf-8 -*-
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
st.set_page_config(
    page_title="Effective Days · 유효일수 분석",
    page_icon="📅",
    layout="wide",
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ─────────────────────────────────────────────────────────────
# 한글 폰트 (가능하면 나눔/맑은고딕 사용, 실패 시 DejaVu)
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
# 상수/팔레트
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
CAT_SHORT = {
    "평일_1": "평1",
    "평일_2": "평2",
    "토요일": "토",
    "일요일": "일",
    "공휴일_대체": "휴",
    "명절_설날": "설",
    "명절_추석": "추",
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
    "평일_1": 1.0,
    "평일_2": 0.952,
    "토요일": 0.85,
    "일요일": 0.60,
    "공휴일_대체": 0.799,
    "명절_설날": 0.842,
    "명절_추석": 0.799,
}

# ─────────────────────────────────────────────────────────────
# 도우미
def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")


def normalize_calendar(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    원본 엑셀을 표준 스키마로 정규화하여 (DataFrame, 공급량_컬럼명 또는 None)을 반환
    필요한 컬럼:
      - 날짜/일자/date/yyyymmdd 중 하나
      - 요일(없으면 자동 생성)
      - 구분/공휴일여부/명절여부 등 힌트(있으면 활용)
      - (선택) '공급'이 들어간 수치형 컬럼 → 가중치 산정에 사용
    """
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
            if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.9:
                date_col = c
                break
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

    # 부울 힌트 표준화
    for col in ["주중여부", "주말여부", "공휴일여부", "명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
        else:
            d[col] = np.nan

    # 공급량 컬럼
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c
            break

    # 카테고리 매핑
    def infer_festival(row):
        g = str(row.get("구분", ""))
        mon = int(row["월"])
        if "설" in g:
            return "명절_설날"
        if "추" in g:
            return "명절_추석"
        if str(row.get("명절여부", "")).upper() == "TRUE":
            if mon in (1, 2):
                return "명절_설날"
            if mon in (9, 10):
                return "명절_추석"
            # 모호하면 추석으로 처리
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


def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    같은 월에서 base_cat(평일_1) 공급량 중앙값 대비 각 카테고리 중앙값 비율(=가중치)
    부족한 데이터는 전체 중앙값/기본값으로 보강. 공휴/명절은 상한(cap_holiday) 적용.
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

    # 전체 중앙값으로 보강 + 상한
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
    """월별 카테고리 일수와 가중 유효일수 합계"""
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
    return out.reset_index()


def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str, float]):
    """12x31 캘린더(월=열, 일=행)"""
    months = range(1, 13)
    days = range(1, 32)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 31)
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i + 0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

    # 그리드
    for x in range(13):
        ax.plot([x, x], [0, 31], color="#D0D5DB", lw=0.8)
    for y in range(32):
        ax.plot([0, 12], [y, y], color="#D0D5DB", lw=0.8)

    for j, m in enumerate(months):
        for i, d in enumerate(days):
            row = df_year[(df_year["월"] == m) & (df_year["일"] == d)]
            if row.empty:
                continue
            cat = row.iloc[0]["카테고리"]
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

    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_LABEL[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig


def render_center_table(df: pd.DataFrame, width_px: int = 1000, caption: Optional[str] = None):
    """
    DataFrame을 HTML로 만들어 가운데 정렬. index 숨김. 숫자 포맷은 df에 들어온 그대로 사용.
    """
    html = df.to_html(index=False, border=0, escape=False)
    style = f"""
    <style>
      .tbl-wrap {{ width:100%; }}
      .tbl-wrap table {{
          margin-left:auto; margin-right:auto;
          width:{width_px}px; border-collapse:collapse;
      }}
      .tbl-wrap th, .tbl-wrap td {{ text-align:center; padding:6px; }}
      .tbl-wrap th {{ font-weight:600; }}
    </style>
    """
    if caption:
        st.caption(caption)
    st.markdown(style + f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 사이드바 UI (분석 시작 버튼 눌러야 실행)
with st.sidebar:
    st.header("예측 기간")

    years = list(range(2026, 2031))  # 2026~2030
    months = list(range(1, 13))

    with st.form("controls", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            start_y = st.selectbox("예측 시작(연)", years, index=0, key="start_y")
        with col2:
            start_m = st.selectbox("예측 시작(월)", months, index=0, key="start_m")

        col3, col4 = st.columns(2)
        with col3:
            end_y = st.selectbox("예측 종료(연)", years, index=1, key="end_y")
        with col4:
            end_m = st.selectbox("예측 종료(월)", months, index=11, key="end_m")

        matrix_year = st.selectbox("매트릭스 표시 연도", years, index=0)

        st.markdown("---")
        st.subheader("데이터 소스")
        file_src = st.radio("파일 선택", ["Repo 내 엑셀 사용", "파일 업로드"], index=0, horizontal=False)
        repo_file = Path("data") / "effective_days_calendar.xlsx"
        upload = None
        if file_src == "Repo 내 엑셀 사용":
            if repo_file.exists():
                st.success(f"레포 파일 사용: {repo_file.name}")
            else:
                st.warning("레포에 data/effective_days_calendar.xlsx 가 없습니다. 업로드를 사용하세요.")
        else:
            upload = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

        submitted = st.form_submit_button("분석 시작", type="primary")

st.title("📅 Effective Days — 유효일수 분석")
st.caption(
    "월별 유효일수 = Σ(해당일 카테고리 가중치). "
    "가중치는 같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값 대비 각 카테고리 중앙값 비율로 산정합니다. "
    "데이터가 부족하면 전역 중앙값(기본값)으로 보강하며 공휴/명절 가중치는 상한 0.95를 둡니다."
)

if not submitted:
    st.stop()

# ─ 데이터 로드
file = None
if file_src == "Repo 내 엑셀 사용" and repo_file.exists():
    file = open(repo_file, "rb")
elif upload is not None:
    file = upload

if file is None:
    st.error("엑셀 파일이 없습니다. 레포에 파일을 넣거나 업로드하세요.")
    st.stop()

try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀을 읽는 중 오류: {e}")
    st.stop()

# 전처리
try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# 가중치 계산
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 기간 필터
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts = pd.Timestamp(int(end_y), int(end_m), 1) + pd.offsets.MonthEnd(0)
if end_ts < start_ts:
    st.error("예측 종료가 시작보다 빠릅니다.")
    st.stop()

pred_df = base_df[(base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts)].copy()
if pred_df.empty:
    st.error("선택한 구간에 해당하는 날짜가 없습니다. 엑셀에 2026~2030 데이터가 있는지 확인하세요.")
    st.stop()

# ─ 1) 캘린더 매트릭스 (맨 위)
st.pyplot(draw_calendar_matrix(int(matrix_year), pred_df[pred_df["연"] == int(matrix_year)], W_global), clear_figure=True)

# ─ 2) 카테고리 가중치 요약
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame(
    {"카테고리": [CAT_LABEL[c] for c in CATS], "전역 가중치(중앙값)": [round(W_global[c], 4) for c in CATS]}
)
render_center_table(w_show, width_px=620)

st.caption(
    "※ 가중치는 달별 ‘평일_1(화·수·목)’ 대비 각 카테고리 중앙값 비율을 다시 중앙값으로 취합한 값입니다. "
    "명절/공휴 가중치는 상한(0.95)을 둡니다. 데이터가 부족하면 기본 가중치로 보강합니다."
)

# ─ 3) 월별 유효일수 표
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly)

# 열 라벨 보기 좋게
rename_map = {
    "일수_평일_1": "일수_평일_1(화·수·목)",
    "일수_평일_2": "일수_평일_2(월·금)",
}
eff_tbl = eff_tbl.rename(columns=rename_map)

# 비고(명절 연휴 안내)
eff_tbl["비고"] = ""
if "일수_명절_설날" in eff_tbl.columns:
    eff_tbl.loc[eff_tbl["일수_명절_설날"] > 0, "비고"] = eff_tbl["비고"] + eff_tbl["일수_명절_설날"].astype(int).astype(str).radd("설연휴 ")
if "일수_명절_추석" in eff_tbl.columns:
    has_chuseok = eff_tbl["일수_명절_추석"] > 0
    eff_tbl.loc[has_chuseok, "비고"] = (
        eff_tbl.loc[has_chuseok, "비고"].str.strip() + " "
        + eff_tbl.loc[has_chuseok, "일수_명절_추석"].astype(int).astype(str).radd("추석연휴 ")
    )
eff_tbl["비고"] = eff_tbl["비고"].str.strip().replace("", np.nan).fillna("")

# 숫자 포맷: 개수(일수)는 정수, 유효일수합/비율만 소수 4자리
int_cols = [c for c in eff_tbl.columns if c.startswith("일수_")] + ["월일수"]
for c in int_cols:
    eff_tbl[c] = eff_tbl[c].astype(int)

eff_tbl["유효일수합"] = eff_tbl["유효일수합"].round(4)
eff_tbl["적용_비율(유효/월일수)"] = eff_tbl["적용_비율(유효/월일수)"].round(4)

# 표 가운데 렌더링
order_cols = (
    ["연", "월", "월일수"]
    + [c for c in eff_tbl.columns if c.startswith("일수_")]
    + ["유효일수합", "적용_비율(유효/월일수)", "비고"]
)
eff_show = eff_tbl[order_cols].sort_values(["연", "월"]).reset_index(drop=True)
render_center_table(
    eff_show,
    width_px=1180,
    caption="비고 예시) ‘설연휴 5일 반영’, ‘추석연휴 4일 반영’ 등. 연휴가 주말과 겹치더라도 본 도구에서는 명절 기간 전체를 보수적으로 명절 가중치로 계산합니다.",
)

# CSV 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_show.to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv",
)
