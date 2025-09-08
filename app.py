# app.py  —  Effective Days (유효일수 분석)
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
# 페이지/폰트 세팅
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    cands = [
        here / "data" / "fonts" / "NanumGothic.ttf",
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]
    for p in cands:
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
    "명절_추석":"명절(추석)"
}
CAT_SHORT = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}
PALETTE = {
    "평일_1":"#7DC3C1","평일_2":"#3DA4AB",
    "토요일":"#5D6D7E","일요일":"#34495E",
    "공휴일_대체":"#E57373",
    "명절_설날":"#F5C04A","명절_추석":"#F39C12",
}
DEFAULT_WEIGHTS = {
    "평일_1":1.0,"평일_2":0.952,"토요일":0.85,"일요일":0.60,
    "공휴일_대체":0.799,"명절_설날":0.842,"명절_추석":0.799
}

# ─────────────────────────────────────────────────────────────
# 유틸
def to_date(x):
    s = str(x).strip()
    if len(s)==8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file, engine="openpyxl")

def normalize_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
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
    if date_col is None: raise ValueError("날짜 열을 찾지 못했습니다. (예: 날짜/일자/date/yyyymmdd)")

    d["날짜"] = d[date_col].map(to_date)
    d = d.dropna(subset=["날짜"]).copy()
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    d["일"] = d["날짜"].dt.day.astype(int)

    # 요일
    if "요일" not in d.columns:
        yo_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
        d["요일"] = d["날짜"].dt.dayofweek.map(yo_map)

    # 불리언 표준화
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns: d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else: d[col] = np.nan

    # 공급량 열 추정
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 명절 분류
    def infer_festival(row):
        g = str(row.get("구분",""))
        mon = int(row["월"])
        txt = g.replace(" ","")
        if "설" in txt: return "명절_설날"
        if "추석" in txt or "추" in txt: return "명절_추석"
        if "명절" in txt or str(row.get("명절여부","")).upper()=="TRUE":
            if mon <= 3: return "명절_설날"
            if mon >= 9: return "명절_추석"
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

def compute_weights_monthly(df: pd.DataFrame, supply_col: Optional[str],
                            base_cat="평일_1", cap_holiday=0.95) -> Tuple[pd.DataFrame, Dict[str,float]]:
    W = []
    for m in range(1,13):
        sub = df[df["월"]==m]
        if sub.empty:
            W.append(pd.Series({c: np.nan for c in CATS}, name=m)); continue
        if (supply_col is None) or sub[sub["카테고리"]==base_cat].empty:
            row = {c:(1.0 if c==base_cat else np.nan) for c in CATS}
            W.append(pd.Series(row, name=m)); continue
        base_med = sub.loc[sub["카테고리"]==base_cat, supply_col].median()
        row = {}
        for c in CATS:
            if c==base_cat: row[c]=1.0
            else:
                s = sub.loc[sub["카테고리"]==c, supply_col]
                row[c] = float(s.median()/base_med) if (len(s)>0 and base_med>0) else np.nan
        W.append(pd.Series(row, name=m))
    W = pd.DataFrame(W)
    global_med = {c:(np.nanmedian(W[c].values) if c in W else np.nan) for c in CATS}
    for c in CATS:
        if np.isnan(global_med[c]): global_med[c]=DEFAULT_WEIGHTS[c]
    for c in ["공휴일_대체","명절_설날","명절_추석"]:
        global_med[c] = min(global_med[c], cap_holiday)
    W_filled = W.fillna(pd.Series(global_med))
    global_w = {c:float(np.nanmedian(W_filled[c].values)) for c in CATS}
    return W_filled, global_w

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    counts = df.pivot_table(index=["연","월"], columns="카테고리", values="날짜",
                            aggfunc="count").reindex(columns=CATS, fill_value=0).astype(int)
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values
    eff_sum = eff.sum(axis=1).rename("유효일수합")
    days_in_month = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([days_in_month, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)

    def _note(r):
        note=[]
        s  = int(r.get("일수_명절_설날",0))
        ch = int(r.get("일수_명절_추석",0))
        c  = int(r.get("일수_공휴일_대체",0))
        if s>0:  note.append(f"설연휴 {s}일 반영")
        if ch>0: note.append(f"추석연휴 {ch}일 반영")
        if c>0:  note.append(f"대체공휴일 {c}일")
        return " · ".join(note)
    out["비고"] = out.apply(_note, axis=1)
    return out.reset_index()

def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    months, days = range(1,13), range(1,32)
    fig, ax = plt.subplots(figsize=(13,7))
    ax.set_xlim(0,12); ax.set_ylim(0,31)
    ax.set_xticks([i+0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i+0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=12)
    for x in range(13): ax.plot([x,x],[0,31], color="#D0D5DB", lw=0.8)
    for y in range(32): ax.plot([0,12],[y,y], color="#D0D5DB", lw=0.8)
    for j,m in enumerate(months):
        for i,d in enumerate(days):
            try: row = df_year[(df_year["월"]==m) & (df_year["일"]==d)].iloc[0]
            except Exception: continue
            cat = row["카테고리"]; color = PALETTE.get(cat, "#EEEEEE")
            rect = mpl.patches.Rectangle((j,i),1,1,color=color,alpha=0.95); ax.add_patch(rect)
            ax.text(j+0.5,i+0.5, CAT_SHORT.get(cat,""), ha="center", va="center",
                    fontsize=9, color="white" if cat in ["일요일","공휴일_대체","명절_설날","명절_추석"] else "black",
                    fontweight="bold")
    handles=[mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_LABEL[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

def center_frame(df: pd.DataFrame, width_ratio=(1,3,1), caption: str|None=None):
    c1,c2,c3 = st.columns(width_ratio)
    with c2:
        sty = (df.style
               .format(precision=4)
               .set_properties(**{"text-align":"center"})
               .set_table_styles([dict(selector="th", props=[("text-align","center")])]))
        st.dataframe(sty, use_container_width=True, hide_index=True)
        if caption: st.caption(caption)

# ─────────────────────────────────────────────────────────────
# 상단 타이틀/설명
st.title("Effective Days — 유효일수 분석")
st.caption("월별 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 **같은 달의 ‘평일_1(화·수·목)’ 공급량 중앙값** 대비 "
           "각 카테고리 중앙값 비율로 산정합니다. (명절/공휴일 가중치는 상한 0.95)")

# 데이터 소스(Repo 기본 + 업로드 옵션)
with st.expander("데이터 불러오기(필요시 열기)", expanded=False):
    src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, horizontal=True)
    default_path = Path("data") / "effective_days_calendar.xlsx"
    if src == "Repo 내 파일 사용":
        if default_path.exists():
            st.success(f"레포 파일 사용: {default_path.name}")
            file = default_path.open("rb")
        else:
            st.warning("레포에 data/effective_days_calendar.xlsx 가 없습니다. 업로드를 이용하세요.")
            file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
    else:
        file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

# ─────────────────────────────────────────────────────────────
# 예측 기간 컨트롤 — 본문 중앙, 가로 배치(한 줄 4개)
years = list(range(2026, 2031))  # 2026~2030

cc1, cc2, cc3 = st.columns([1, 5, 1])  # 중앙 정렬
with cc2:
    st.subheader("예측 기간")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        start_y = st.selectbox("예측 시작(연)", years, index=0, key="start_y")
    with r1c2:
        start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="start_m")
    with r1c3:
        end_y   = st.selectbox("예측 종료(연)", years, index=min(1,len(years)-1), key="end_y")
    with r1c4:
        end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="end_m")

    r2c1, r2c2, r2c3 = st.columns([1,2,1])
    with r2c2:
        matrix_year = st.selectbox("매트릭스 표시 연도", years, index=0, key="matrix_year")

    r3c1, r3c2, r3c3 = st.columns([1,2,1])
    with r3c2:
        run_btn = st.button("분석 시작", type="primary")

# 분석 시작 누르기 전 스톱
if not run_btn:
    st.stop()

# 입력/데이터 체크
if file is None:
    st.error("엑셀을 선택하세요.")
    st.stop()

try:
    raw = load_excel(file)
except Exception:
    st.error("엑셀을 읽는 중 문제가 발생했습니다.")
    st.stop()

try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts   = pd.Timestamp(int(end_y), int(end_m), 1)
if end_ts < start_ts:
    st.error("종료가 시작보다 빠릅니다.")
    st.stop()

mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 구간의 날짜가 엑셀에 없습니다. (미래 연도 2026~ 포함인지 확인)")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 1) 매트릭스 (맨 위 중앙)
years_in_range = sorted(base_df["연"].unique().tolist())
view_year = matrix_year if matrix_year in years_in_range else years_in_range[0]
fig = draw_calendar_matrix(view_year, base_df[base_df["연"]==view_year], W_global)
mc1, mc2, mc3 = st.columns([1,6,1])
with mc2:
    st.pyplot(fig, clear_figure=True)

# 2) 카테고리 가중치 요약 (중앙)
w_df = pd.DataFrame({
    "카테고리":[CAT_LABEL[c] for c in CATS],
    "전역 가중치(중앙값)":[round(W_global[c],4) for c in CATS]
})
center_frame(
    w_df,
    caption="가중치는 동일 달의 ‘평일_1(화·수·목)’ 공급량 중앙값을 1로 두고, "
            "각 카테고리 중앙값 대비 비율로 산정됩니다. 데이터가 부족한 달은 전체 중앙값/기본값으로 보강되며, "
            "명절·공휴일 가중치는 상한 0.95를 적용합니다."
)

# 3) 월별 유효일수 요약 (중앙)
eff_tbl = effective_days_by_month(pred_df, W_monthly)
show_cols = (["연","월","월일수"] + [f"일수_{c}" for c in CATS] + ["유효일수합","적용_비율(유효/월일수)","비고"])
eff_show = eff_tbl[show_cols].copy()
eff_show.columns = [c.replace("평일_1","평일_1(화·수·목)").replace("평일_2","평일_2(월·금)") for c in eff_show.columns]

st.subheader("월별 유효일수 요약")
center_frame(
    eff_show,
    caption="비고 예시) ‘설연휴 5일 반영’, ‘추석연휴 4일 반영’ 등. "
            "연휴가 주말과 겹치더라도 본 도구에서는 **명절 기간 전체**를 보수적으로 명절 가중치로 계산합니다."
)

csv = eff_tbl.sort_values(["연","월"]).to_csv(index=False).encode("utf-8-sig")
dl1, dl2, dl3 = st.columns([1,2,1])
with dl2:
    st.download_button("월별 유효일수 결과 CSV 다운로드", data=csv,
                       file_name="effective_days_by_month.csv", mime="text/csv")
