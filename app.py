# app.py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

# ───────────────────────── 기본 설정 ─────────────────────────
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

TITLE = "Effective Days — 유효일수 분석"
DESC = (
    "월별 유효일수 = Σ(해당일 카테고리 가중치). "
    "가중치는 같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값 대비 각 카테고리 중앙값 비율로 산정합니다. "
    "(데이터가 부족하면 전역 중앙값/기본값으로 보강하며 공휴/명절 가중치는 상한 0.95 적용)"
)

CATS: List[str] = ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]
CAT_SHORT: Dict[str, str] = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}
PALETTE = {
    "평일_1":"#7DC3C1","평일_2":"#3DA4AB","토요일":"#5D6D7E","일요일":"#34495E",
    "공휴일_대체":"#E57373","명절_설날":"#F5C04A","명절_추석":"#F39C12",
}
DEFAULT_WEIGHTS = {"평일_1":1.0,"평일_2":0.952,"토요일":0.85,"일요일":0.60,"공휴일_대체":0.799,"명절_설날":0.842,"명절_추석":0.799}

# ───────────────────────── 한글 폰트 ─────────────────────────
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

# ───────────────────────── 유틸 ─────────────────────────
def to_date(x):
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame):
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 열 찾기
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
    d["요일"] = d["날짜"].dt.dayofweek.map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"})

    # 불리언 힌트 통일
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else:
            d[col] = np.nan

    # 공급량 열 추정(없어도 동작)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 명절/대체/요일 기반 카테고리
    def infer_festival(row):
        g = str(row.get("구분",""))
        mon = int(row["월"])
        if "설" in g: return "명절_설날"
        if "추" in g: return "명절_추석"
        if str(row.get("명절여부","")).upper() == "TRUE":
            if mon in (1,2):  return "명절_설날"
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
        if y in ["화","수","목"]: return "평일_1"     # Tue-Thu
        if y in ["월","금"]: return "평일_2"         # Mon, Fri
        return "평일_1"

    d["카테고리"] = d.apply(map_category, axis=1)
    d["카테고리"] = pd.Categorical(d["카테고리"], categories=CATS)
    return d, supply_col

def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str,float]]:
    W = []
    for m in range(1,13):
        sub = df[df["월"]==m]
        if sub.empty:
            W.append(pd.Series({c: np.nan for c in CATS}, name=m)); continue

        if (supply_col is None) or sub[sub["카테고리"]==base_cat].empty:
            row = {c: (1.0 if c==base_cat else np.nan) for c in CATS}
            W.append(pd.Series(row, name=m)); continue

        base_med = sub.loc[sub["카테고리"]==base_cat, supply_col].median()
        row = {}
        for c in CATS:
            if c==base_cat: row[c]=1.0; continue
            s = sub.loc[sub["카테고리"]==c, supply_col]
            row[c] = float(s.median()/base_med) if (len(s)>0 and base_med>0) else np.nan
        W.append(pd.Series(row, name=m))
    W = pd.DataFrame(W)  # index=월

    # 전체 중앙값으로 보강 + 휴일 상한
    global_med = {c: (np.nanmedian(W[c].values) if c in W else np.nan) for c in CATS}
    for c in CATS:
        if np.isnan(global_med[c]): global_med[c] = DEFAULT_WEIGHTS[c]
    for c in ["공휴일_대체","명절_설날","명절_추석"]:
        global_med[c] = min(global_med[c], cap_holiday)

    W_filled = W.fillna(pd.Series(global_med))
    global_w = {c: float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    counts = df.pivot_table(index=["연","월"], columns="카테고리", values="날짜", aggfunc="count")\
              .reindex(columns=CATS, fill_value=0).astype(int)
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values
    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)
    return out.reset_index()

def remark_row(row: pd.Series) -> str:
    notes = []
    if row.get("일수_공휴일_대체",0) > 0:
        notes.append(f"대체공휴일 {int(row['일수_공휴일_대체'])}일")
    if row.get("일수_명절_설날",0) > 0:
        notes.append(f"설연휴 {int(row['일수_명절_설날'])}일 반영")
    if row.get("일수_명절_추석",0) > 0:
        notes.append(f"추석연휴 {int(row['일수_명절_추석'])}일 반영")
    return " · ".join(notes)

def center_html(df: pd.DataFrame, width_px: int = 1100, height_px: int = 420,
                float4: Optional[List[str]] = None, int_cols: Optional[List[str]] = None) -> str:
    """모든 셀 가운데 정렬 + 일부 컬럼만 포맷."""
    float4 = float4 or []
    int_cols = int_cols or []
    sty = df.style.set_table_styles([
        {"selector":"th","props":"text-align:center; font-weight:600;"},
        {"selector":"td","props":"text-align:center;"},
        {"selector":"table","props":f"margin-left:auto; margin-right:auto; width:{width_px}px; border-collapse:collapse;"},
    ])
    # pandas 2.2: hide_index는 deprecated → hide(axis="index")
    sty = sty.hide(axis="index")

    for c in float4:
        if c in df.columns: sty = sty.format({c: "{:.4f}"})
    for c in int_cols:
        if c in df.columns: sty = sty.format({c: "{:.0f}"})
    return sty.to_html()

def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    months = range(1,13); days = range(1,32)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 31)
    ax.set_xticks([i+0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i+0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

    for x in range(13):
        ax.plot([x,x],[0,31], color="#D0D5DB", lw=0.8)
    for y in range(32):
        ax.plot([0,12],[y,y], color="#D0D5DB", lw=0.8)

    for j, m in enumerate(months):
        for i, d in enumerate(days):
            row = df_year[(df_year["월"]==m) & (df_year["일"]==d)]
            if row.empty: continue
            cat = row.iloc[0]["카테고리"]
            color = PALETTE.get(cat, "#EEEEEE")
            rect = mpl.patches.Rectangle((j, i), 1, 1, color=color, alpha=0.95)
            ax.add_patch(rect)
            label = CAT_SHORT.get(cat, "")
            ax.text(j+0.5, i+0.5, label, ha="center", va="center",
                    fontsize=9,
                    color="white" if cat in ["일요일","공휴일_대체","명절_설날","명절_추석"] else "black",
                    fontweight="bold")
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{c} ({weights.get(c,1):.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

# ───────────────────────── UI ─────────────────────────
st.title(TITLE)
st.caption(DESC)

with st.sidebar:
    st.header("예측 기간")
    years = list(range(2026, 2031))  # 2026~2030
    colA, colB = st.columns(2)
    with colA: y_start = st.selectbox("예측 시작(연)", years, index=0, key="ys")
    with colB: m_start = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="ms")
    colC, colD = st.columns(2)
    with colC: y_end = st.selectbox("예측 종료(연)", years, index=1, key="ye")
    with colD: m_end = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="me")
    show_year = st.selectbox("매트릭스 표시 연도", years, index=0, key="viewy")

    st.markdown("---")
    st.subheader("데이터 소스")
    src = st.radio("파일 선택", ["Repo 내 엑셀 사용","파일 업로드"], index=0)
    default_path = Path("data") / "effective_days_calendar.xlsx"
    if src == "Repo 내 엑셀 사용":
        if default_path.exists():
            st.success(f"레포 파일 사용: {default_path.name}")
            file = open(default_path, "rb")
        else:
            file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
    else:
        file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

    run_btn = st.button("분석 시작", type="primary")

if not run_btn:
    st.stop()

# ───────────────────────── 데이터 로드 & 전처리 ─────────────────────────
if file is None:
    st.warning("엑셀을 업로드하거나 data/effective_days_calendar.xlsx 를 레포에 넣어주세요.")
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

# 가중치 계산
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 기간 필터
start_ts = pd.Timestamp(int(y_start), int(m_start), 1)
end_ts   = pd.Timestamp(int(y_end),   int(m_end),   1)
if end_ts < start_ts:
    st.error("예측 종료가 시작보다 빠릅니다.")
    st.stop()

mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 예측 구간에 해당하는 날짜가 엑셀에 없습니다.")
    st.stop()

# ───────────────────────── 매트릭스(맨 위) ─────────────────────────
years_in_range = sorted(pred_df["연"].unique().tolist())
if show_year not in years_in_range:
    st.info(f"선택 연도 {show_year} 는 현재 구간에 데이터가 없어, 가장 이른 연도({years_in_range[0]})로 대체합니다.")
    show_year = years_in_range[0]
fig = draw_calendar_matrix(show_year, pred_df[pred_df["연"]==show_year], W_global)
st.pyplot(fig, clear_figure=True)

# ───────────────────────── 가중치 요약 ─────────────────────────
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({"카테고리": CATS, "전역 가중치(중앙값)": [round(W_global[c],4) for c in CATS]})
html = center_html(w_show, width_px=620, float4=["전역 가중치(중앙값)"])
st.markdown(html, unsafe_allow_html=True)

# ───────────────────────── 월별 유효일수 ─────────────────────────
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly)

# 비고 생성(명절/대체)
eff_tbl["비고"] = eff_tbl.apply(remark_row, axis=1)

# 보여줄 컬럼 구성 & 포맷
show_cols = (["연","월","월일수"]
             + [f"일수_{c}" for c in CATS]
             + ["유효일수합","적용_비율(유효/월일수)","비고"])
eff_show = eff_tbl[show_cols].sort_values(["연","월"]).reset_index(drop=True)

float4_cols = ["유효일수합","적용_비율(유효/월일수)"]
int_cols = [c for c in eff_show.columns if c not in float4_cols+["비고"]]
html2 = center_html(eff_show, width_px=1180, float4=float4_cols, int_cols=int_cols)
st.markdown(html2, unsafe_allow_html=True)

st.caption("비고 예시) ‘설연휴 5일 반영’, ‘추석연휴 4일 반영’ 등. 연휴가 주말과 겹치더라도 본 도구에서는 명절 기간 전체를 보수적으로 명절 가중치로 계산합니다.")
