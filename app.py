# app.py — Effective Days (유효일수 분석 전용·최종)
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
# 상수/팔레트
CATS = ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]
CAT_LABEL = {
    "평일_1":"평일_1(화·수·목)",
    "평일_2":"평일_2(월·금)",
    "토요일":"토요일",
    "일요일":"일요일",
    "공휴일_대체":"공휴일·대체",
    "명절_설날":"명절(설)",
    "명절_추석":"명절(추석)",
}
CAT_SHORT = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}

PALETTE = {
    "평일_1":"#7DC3C1",
    "평일_2":"#3DA4AB",
    "토요일":"#5D6D7E",
    "일요일":"#34495E",
    "공휴일_대체":"#E57373",
    "명절_설날":"#F5C04A",
    "명절_추석":"#F39C12",
}
DEFAULT_WEIGHTS = {
    "평일_1":1.0, "평일_2":0.952, "토요일":0.85, "일요일":0.60,
    "공휴일_대체":0.799, "명절_설날":0.842, "명절_추석":0.799
}

# ─────────────────────────────────────────────────────────────
# 공통 유틸
def to_date(x):
    s = str(x).strip()
    if len(s)==8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """엑셀 원본을 표준 스키마로 정규화하고 (DataFrame, 공급량컬럼명 or None) 반환"""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 열
    date_col = None
    for c in d.columns:
        if str(c).lower() in ["날짜","일자","date"]:
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
        yo_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
        d["요일"] = d["날짜"].dt.dayofweek.map(yo_map)

    # 불리언 힌트 정규화
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else:
            d[col] = np.nan

    # 공급량 컬럼 추정
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 카테고리 분류
    def infer_festival(row):
        g = str(row.get("구분",""))
        mon = int(row["월"])
        # 명시 문자열 우선
        if "설" in g: return "명절_설날"
        if "추" in g: return "명절_추석"
        # 불리언 힌트
        if str(row.get("명절여부","")).upper() == "TRUE":
            if mon in (1,2): return "명절_설날"
            if mon in (9,10): return "명절_추석"
            return "명절_추석"
        return None

    def map_category(row):
        g, y = str(row.get("구분","")), row["요일"]
        if ("공휴" in g) or ("대체" in g) or (str(row.get("공휴일여부","")).upper()=="TRUE"):
            return "공휴일_대체"
        fest = infer_festival(row)
        if fest: return fest
        if y=="토": return "토요일"
        if y=="일": return "일요일"
        if y in ["화","수","목"]: return "평일_1"
        if y in ["월","금"]: return "평일_2"
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
    반환: (월별가중치 DataFrame(index=월), 전역가중치 dict)
    """
    W = []
    for m in range(1,13):
        sub = df[df["월"]==m]
        if sub.empty:
            W.append(pd.Series({c: np.nan for c in CATS}, name=m))
            continue
        if (supply_col is None) or sub[sub["카테고리"]==base_cat].empty:
            row = {c: (1.0 if c==base_cat else np.nan) for c in CATS}
            W.append(pd.Series(row, name=m))
            continue
        base_med = sub.loc[sub["카테고리"]==base_cat, supply_col].median()
        row = {}
        for c in CATS:
            if c==base_cat:
                row[c] = 1.0
            else:
                s = sub.loc[sub["카테고리"]==c, supply_col]
                row[c] = float(s.median()/base_med) if (len(s)>0 and base_med>0) else np.nan
        W.append(pd.Series(row, name=m))
    W = pd.DataFrame(W)  # index=월

    # 전체 중앙값으로 보강 + 휴일/명절 상한
    global_med = {c: (np.nanmedian(W[c].values) if c in W else np.nan) for c in CATS}
    for c in CATS:
        if np.isnan(global_med[c]):
            global_med[c] = DEFAULT_WEIGHTS[c]
    for c in ["공휴일_대체","명절_설날","명절_추석"]:
        global_med[c] = min(global_med[c], cap_holiday)

    W_filled = W.fillna(pd.Series(global_med))
    global_w = {c: float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 카테고리 일수와 가중 유효일수 합계를 계산"""
    counts = df.pivot_table(
        index=["연","월"], columns="카테고리", values="날짜", aggfunc="count"
    ).reindex(columns=CATS, fill_value=0).astype(int)

    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values

    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)

    # 비고(명절/대체공휴일 요약)
    remarks = []
    tmp = counts.reset_index()
    for _, r in tmp.iterrows():
        info = []
        if r.get("명절_설날",0)>0: info.append(f"설연휴 {int(r['명절_설날'])}일 반영")
        if r.get("명절_추석",0)>0: info.append(f"추석연휴 {int(r['명절_추석'])}일 반영")
        if r.get("공휴일_대체",0)>0: info.append(f"대체공휴일 {int(r['공휴일_대체'])}일")
        remarks.append("; ".join(info))
    out = out.reset_index()
    out["비고"] = remarks
    return out

# ─────────────────────────────────────────────────────────────
# 테이블 가운데 정렬 출력 (Styler → HTML)
def render_center_table(df: pd.DataFrame, width_px: int = 900, height_px: int = 360):
    sty = (
        df.style
        .format(precision=4)
        .set_table_styles([
            {"selector":"th","props":"text-align:center; font-weight:600;"},
            {"selector":"td","props":"text-align:center;"},
            {"selector":"table","props":f"margin-left:auto; margin-right:auto; width:{width_px}px; border-collapse:collapse;"},
        ])
    )
    try:
        sty = sty.hide(axis="index")
    except Exception:
        pass
    html = f"""
    <style>
    .tbl-wrap {{
        display:block; margin: 8px auto 16px auto; max-width:{width_px}px;
    }}
    thead th, tbody td {{
        border: 1px solid #e5e7eb; padding: 6px 8px;
    }}
    </style>
    <div class="tbl-wrap">{sty.to_html()}</div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 캘린더 매트릭스
def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    months = range(1,13)
    days = range(1,32)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 31)
    ax.set_xticks([i+0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

    for x in range(13):
        ax.plot([x,x],[0,31], color="#D0D5DB", lw=0.6)

    for j, m in enumerate(months):
        for i, d in enumerate(days):
            rows = df_year[(df_year["월"]==m) & (df_year["일"]==d)]
            if rows.empty: continue
            cat = rows.iloc[0]["카테고리"]
            color = PALETTE.get(cat, "#EEEEEE")
            rect = mpl.patches.Rectangle((j, i), 1, 1, color=color, alpha=0.95)
            ax.add_patch(rect)
    # 범례
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_LABEL[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# 사이드바 (좌측·가로배치 + 버튼)
with st.sidebar:
    st.header("예측 기간")
    years = list(range(2026, 2031))
    col1, col2 = st.columns(2, gap="small")
    with col1:
        sy = st.selectbox("예측 시작(연)", years, index=0, key="sy")
    with col2:
        sm = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="sm")
    col3, col4 = st.columns(2, gap="small")
    with col3:
        ey = st.selectbox("예측 종료(연)", years, index=1, key="ey")
    with col4:
        em = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="em")

    my = st.selectbox("매트릭스 표시 연도", years, index=0, key="my")

    run_btn = st.button("분석 시작", type="primary")

# ─────────────────────────────────────────────────────────────
# 본문
st.markdown(
    "<h1 style='margin-top:-8px'>📅 Effective Days — 유효일수 분석</h1>",
    unsafe_allow_html=True,
)
st.caption("월별 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값 대비 각 카테고리 중앙값 비율로 산정합니다. "
           "데이터가 부족하면 전역 중앙값(기본값)으로 보강하며 공휴/명절 가중치는 상한 0.95를 둡니다.")

if not run_btn:
    st.info("좌측에서 기간을 선택한 뒤 **분석 시작**을 눌러주세요.")
    st.stop()

# 데이터 로드
default_path = Path("data") / "effective_days_calendar.xlsx"
file_ok = default_path.exists()
if not file_ok:
    st.error("레포지토리의 `data/effective_days_calendar.xlsx` 가 없어요. 파일을 업로드해 주세요.")
    st.stop()

try:
    raw = pd.read_excel(default_path, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀을 읽는 중 오류: {e}")
    st.stop()

# 전처리
try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# 가중치 계산(전체 데이터 기준)
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 예측 구간 필터
start_ts = pd.Timestamp(int(sy), int(sm), 1)
end_ts   = pd.Timestamp(int(ey), int(em), 1) + pd.offsets.MonthEnd(0)
if end_ts < start_ts:
    st.error("예측 종료가 시작보다 빠릅니다.")
    st.stop()

pred_df = base_df[(base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts)].copy()
if pred_df.empty:
    st.warning("선택한 구간에 해당하는 날짜가 없습니다. `data/effective_days_calendar.xlsx`의 범위를 확인하세요.")
    st.stop()

# ─ 출력: 상단 매트릭스
st.pyplot(draw_calendar_matrix(int(my), base_df[base_df["연"]==int(my)], W_global), clear_figure=True)

# ─ 전역 가중치 표
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({
    "카테고리":[CAT_LABEL[c] for c in CATS],
    "전역 가중치(중앙값)":[W_global[c] for c in CATS]
})
render_center_table(w_show, width_px=620, height_px=230)

# ─ 월별 유효일수 표
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly)
show_cols = (
    ["연","월","월일수"] +
    [f"일수_{c}" for c in CATS] +
    ["유효일수합","적용_비율(유효/월일수)","비고"]
)
eff_sorted = eff_tbl[show_cols].sort_values(["연","월"], ignore_index=True)
# 열 이름 보기 좋게
nice_cols = {
    "연":"연", "월":"월", "월일수":"월일수",
    "일수_평일_1":"일수_평일_1(화·수·목)",
    "일수_평일_2":"일수_평일_2(월·금)",
    "일수_토요일":"일수_토요일",
    "일수_일요일":"일수_일요일",
    "일수_공휴일_대체":"일수_공휴일·대체",
    "일수_명절_설날":"일수_명절_설",
    "일수_명절_추석":"일수_명절_추석",
    "유효일수합":"유효일수합",
    "적용_비율(유효/월일수)":"적용_비율(유효/월일수)",
    "비고":"비고",
}
eff_sorted = eff_sorted.rename(columns=nice_cols)
render_center_table(eff_sorted, width_px=1180, height_px=420)

# 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_sorted.to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv"
)

# 계산 로직 설명
with st.expander("계산 방법(요약)"):
    st.markdown("""
- **가중치 산정**: 같은 달의 ‘평일_1(화·수·목)’ 공급량 **중앙값**을 기준으로 각 카테고리의 **중앙값 비율**을 가중치로 사용합니다.  
- **휴일/명절 상한**: 공휴일·대체, 명절(설/추석)의 가중치는 **최대 0.95**로 제한합니다.  
- **전역 가중치**: 월별 가중치가 부족할 경우, **전 구간 중앙값**(없으면 기본값)을 사용합니다.  
- **명절 분류**: `구분` 값이 `명절(설·추석)`처럼 혼합 표기이면 **1–2월은 설, 9–10월은 추석**으로 분리합니다.  
- **비고**: 월별 일수에서 **설/추석/대체공휴일** 건수를 읽어 `설연휴 n일 반영`처럼 간단히 요약합니다.
""")
