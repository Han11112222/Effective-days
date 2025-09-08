# app.py  — Effective Days · 유효일수 분석
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
CAT_SHORT = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}
PALETTE = {
    "평일_1":"#7DC3C1", "평일_2":"#3DA4AB", "토요일":"#5D6D7E", "일요일":"#34495E",
    "공휴일_대체":"#E57373", "명절_설날":"#F5C04A", "명절_추석":"#F39C12"
}
DEFAULT_WEIGHTS = {
    "평일_1":1.0, "평일_2":0.9713, "토요일":0.8566, "일요일":0.7651,
    "공휴일_대체":0.8410, "명절_설날":0.8420, "명절_추석":0.7990
}

# ─────────────────────────────────────────────────────────────
# 유틸 및 전처리
def to_date(x):
    s = str(x).strip()
    if len(s)==8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame):
    """
    입력 엑셀을 표준 스키마로 정리:
    - 날짜/연/월/일/요일
    - 카테고리(평1/평2/토/일/공휴/명절_설/명절_추)
    - 공급량 컬럼 자동 탐색 (없으면 None)
    """
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
                if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.9 and d[c].astype(str).str.len().mode().iat[0] in [7,8]:
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

    # 불리언 힌트 표준화
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else:
            d[col] = np.nan

    # 공급량 컬럼(있으면)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 명절 판별
    def infer_festival(row):
        g = str(row.get("구분",""))
        mon = int(row["월"])
        # 텍스트 안에 '설'/'추'가 들어가면 바로 매핑
        if "설" in g: return "명절_설날"
        if "추" in g: return "명절_추석"
        if str(row.get("명절여부","")).upper() == "TRUE":
            if mon in (1,2):  return "명절_설날"
            if mon in (9,10): return "명절_추석"
        return None

    def map_category(row):
        # ⚠️ 명절을 최우선 분류 → 그 다음 공휴/대체 → 요일
        fest = infer_festival(row)
        if fest:
            return fest
        g, y = str(row.get("구분","")), row["요일"]
        if ("공휴" in g) or ("대체" in g) or (str(row.get("공휴일여부","")).upper()=="TRUE"):
            return "공휴일_대체"
        if y=="토": return "토요일"
        if y=="일": return "일요일"
        if y in ["화","수","목"]: return "평일_1"
        if y in ["월","금"]:     return "평일_2"
        return "평일_1"

    d["카테고리"] = d.apply(map_category, axis=1)
    d["카테고리"] = pd.Categorical(d["카테고리"], categories=CATS, ordered=False)
    return d, supply_col

# ─────────────────────────────────────────────────────────────
# 가중치/유효일수
def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    같은 월 내 '평일_1' 중앙값 대비 각 카테고리 중앙값 비율(가중치) 산정.
    데이터 부족 시 전체 중앙값→DEFAULT로 보강. 설/추/공휴 가중치 상한 0.95.
    """
    W = []
    for m in range(1,13):
        sub = df[df["월"]==m]
        if sub.empty:
            W.append(pd.Series({c: np.nan for c in CATS}, name=m)); continue
        if (supply_col is None) or sub[sub["카테고리"]==base_cat].empty:
            W.append(pd.Series({c:(1.0 if c==base_cat else np.nan) for c in CATS}, name=m)); continue
        base_med = sub.loc[sub["카테고리"]==base_cat, supply_col].median()
        row = {}
        for c in CATS:
            if c==base_cat: row[c]=1.0
            else:
                s = sub.loc[sub["카테고리"]==c, supply_col]
                row[c] = float(s.median()/base_med) if (len(s)>0 and base_med>0) else np.nan
        W.append(pd.Series(row, name=m))
    W = pd.DataFrame(W)

    global_med = {c: (np.nanmedian(W[c].values) if c in W else np.nan) for c in CATS}
    for c in CATS:
        if np.isnan(global_med[c]): global_med[c] = DEFAULT_WEIGHTS[c]
    for c in ["공휴일_대체","명절_설날","명절_추석"]:
        global_med[c] = min(global_med[c], cap_holiday)

    W_filled = W.fillna(pd.Series(global_med))
    global_w = {c: float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 일수/가중유효일수 합계 + 비고(명절/대체공휴일 일수)."""
    counts = df.pivot_table(index=["연","월"], columns="카테고리", values="날짜",
                            aggfunc="count").reindex(columns=CATS, fill_value=0).astype(int)

    # 월별 가중치 적용
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values

    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)

    # 비고(명절/대체공휴일 요약)
    def memo(row):
        notes = []
        if row.get("일수_명절_설날",0)>0: notes.append(f"설연휴 {int(row['일수_명절_설날'])}일")
        if row.get("일수_명절_추석",0)>0: notes.append(f"추석연휴 {int(row['일수_명절_추석'])}일")
        if row.get("일수_공휴일_대체",0)>0: notes.append(f"대체공휴일 {int(row['일수_공휴일_대체'])}일")
        return " · ".join(notes) if notes else ""
    out["비고"] = out.apply(memo, axis=1)

    out = out.reset_index()
    return out

# ─────────────────────────────────────────────────────────────
# 시각화/표 렌더링
def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    months = range(1,13); days = range(1,32)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 31)
    ax.set_xticks([i+0.5 for i in range(12)]); ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i+0.5 for i in range(31)]); ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis(); ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

    # 그리드
    for x in range(13): ax.plot([x,x],[0,31], color="#D0D5DB", lw=0.8)
    for y in range(32): ax.plot([0,12],[y,y], color="#D0D5DB", lw=0.8)

    for j, m in enumerate(months):
        for i, d in enumerate(days):
            try:
                row = df_year[(df_year["월"]==m) & (df_year["일"]==d)].iloc[0]
            except Exception:
                continue
            cat = row["카테고리"]
            color = PALETTE.get(cat, "#EEEEEE")
            rect = mpl.patches.Rectangle((j, i), 1, 1, color=color, alpha=0.95)
            ax.add_patch(rect)
            label = CAT_SHORT.get(cat, "")
            ax.text(j+0.5, i+0.5, label, ha="center", va="center",
                    fontsize=9,
                    color="white" if cat in ["일요일","공휴일_대체","명절_설날","명절_추석"] else "black",
                    fontweight="bold")
    # 범례(가중치)
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{c} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

def center_table(df: pd.DataFrame, width_px: int = 1100):
    """표를 가운데 정렬로 예쁘게 렌더링"""
    style = df.style.set_properties(**{
        "text-align":"center"
    }).set_table_styles([{
        "selector":"th", "props":"text-align:center;"
    }])
    html = style.to_html()
    wrapped = f"""
    <style>
      .tbl-wrap {{max-width:{width_px}px; margin: 0 auto;}}
    </style>
    <div class='tbl-wrap'>{html}</div>
    """
    st.markdown(wrapped, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 사이드바(예측 기간)
with st.sidebar:
    st.header("예측 기간")
    yr_range = list(range(2026, 2031))
    months = list(range(1,13))

    c1, c2 = st.columns(2)
    with c1:
        start_y = st.selectbox("예측 시작(연)", yr_range, index=0, key="sy")
    with c2:
        start_m = st.selectbox("예측 시작(월)", months, index=0, key="sm")

    c3, c4 = st.columns(2)
    with c3:
        end_y = st.selectbox("예측 종료(연)", yr_range, index=1, key="ey")
    with c4:
        end_m = st.selectbox("예측 종료(월)", months, index=11, key="em")

    matrix_y = st.selectbox("매트릭스 표시 연도", yr_range, index=0, key="my")

    run_btn = st.button("분석 시작", type="primary")

st.title("📅 Effective Days — 유효일수 분석")
st.caption("월별 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 **같은 월의 ‘평일_1(화·수·목)’ 중앙값 대비**로 산정합니다. (명절/공휴일 가중치 상한 0.95 적용)")

if not run_btn:
    st.stop()

# ─ 데이터 불러오기
default_path = Path("data") / "effective_days_calendar.xlsx"
if default_path.exists():
    file = open(default_path, "rb")
else:
    st.error("레포에 data/effective_days_calendar.xlsx 가 필요합니다. 또는 업로드 기능을 잠시 켜서 사용하세요.")
    st.stop()

try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀 읽기 오류: {e}")
    st.stop()

# 전처리/가중치
base_df, supply_col = normalize_calendar(raw)
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 예측기간 필터
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts   = pd.Timestamp(int(end_y),   int(end_m),   1) + pd.offsets.MonthEnd(0)
if end_ts < start_ts:
    st.error("예측 종료가 시작보다 빠릅니다."); st.stop()

pred_df = base_df[(base_df["날짜"]>=start_ts) & (base_df["날짜"]<=end_ts)].copy()
if pred_df.empty:
    st.error("선택한 구간에 해당하는 날짜가 없습니다. 2026~2030 범위를 확인하세요.")
    st.stop()

# 매트릭스(상단)
st.subheader("연간 매트릭스")
yr_df = base_df[base_df["연"]==int(matrix_y)].copy()
fig = draw_calendar_matrix(int(matrix_y), yr_df, W_global)
st.pyplot(fig, clear_figure=True)

# 가중치 요약(가운데 정렬)
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({
    "카테고리":["평일_1(화·수·목)","평일_2(월·금)","토요일","일요일","공휴일·대체","명절(설)","명절(추석)"],
    "전역 가중치(중앙값)":[round(W_global["평일_1"],4),round(W_global["평일_2"],4),
                      round(W_global["토요일"],4),round(W_global["일요일"],4),
                      round(W_global["공휴일_대체"],4),round(W_global["명절_설날"],4),
                      round(W_global["명절_추석"],4)]
})
center_table(w_show, width_px=650)

# 월별 유효일수 표
eff_tbl = effective_days_by_month(pred_df, W_monthly)
# 표시 컬럼
show_cols = (
    ["연","월","월일수"]
    + [f"일수_{c}" for c in CATS]
    + ["유효일수합","적용_비율(유효/월일수)","비고"]
)
eff_show = eff_tbl[show_cols].sort_values(["연","월"]).reset_index(drop=True)

st.subheader("월별 유효일수 요약")
center_table(eff_show, width_px=1200)

# 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl.sort_values(["연","월"]).to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv"
)

# 참고 설명
with st.expander("계산 방식 간단 설명", expanded=False):
    st.markdown("""
- **가중치 산정**: 같은 월에서 `평일_1(화·수·목)` 공급량 **중앙값**을 기준(=1.0)으로, 카테고리별 중앙값 비율을 가중치로 사용합니다.  
- **명절 처리**: `구분`에 '설/추'가 있거나 `명절여부=TRUE`인 경우 **명절(설/추석)으로 우선 분류**하고, 그 다음 공휴일/대체 여부를 봅니다.  
  - 설/추석이 일반 공휴일과 겹쳐도 명절로 계산됩니다.  
- **보수적 처리 예시**: 2026-02(설 2/14~2/18), 2026-09(추석 9/24~9/27)처럼 연휴가 길면 해당 **일수를 모두 명절로 집계**하여 유효일수가 감소합니다.
""")

