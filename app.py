# app.py — Effective Days (아이콘 헤더 + 분석 시작 버튼 유지 + 매트릭스 즉시 갱신 + 좌측하단 CSV + 설명을 표 오른쪽에 더 가깝게)
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
    "(표본 부족 시 전역 중앙값/기본값 보강. 휴일/명절 가중치는 상한 적용)"
)

CATS: List[str] = ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]
CAT_SHORT: Dict[str, str] = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}
PALETTE = {
    "평일_1":"#7DC3C1","평일_2":"#3DA4AB","토요일":"#5D6D7E","일요일":"#34495E",
    "공휴일_대체":"#E57373","명절_설날":"#F5C04A","명절_추석":"#F39C12",
}
DEFAULT_WEIGHTS = {"평일_1":1.0,"평일_2":0.952,"토요일":0.85,"일요일":0.60,"공휴일_대체":0.799,"명절_설날":0.842,"명절_추석":0.799}
CAP_HOLIDAY = 0.90  # 휴일·명절 가중치 상한

# (NEW) ─────────────────────── 아이콘 헤더용 CSS/함수 ───────────────────────
st.markdown(
    """
    <style>
      .icon-h1{display:flex;align-items:center;gap:.6rem;font-size:2.0rem;font-weight:800;margin:.2rem 0 .6rem 0;}
      .icon-h2{display:flex;align-items:center;gap:.5rem;font-size:1.3rem;font-weight:700;margin:1.0rem 0 .6rem 0;}
      .icon-h3{display:flex;align-items:center;gap:.45rem;font-size:1.1rem;font-weight:700;margin:.6rem 0 .4rem 0;}
      .icon-emoji{font-size:1.25em;line-height:1;filter:drop-shadow(0 1px 0 rgba(0,0,0,.05))}
    </style>
    """,
    unsafe_allow_html=True,
)

def icon_title(text: str, icon: str = "🧩"):
    st.markdown(f"<div class='icon-h1'><span class='icon-emoji'>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)

def icon_section(text: str, icon: str = "🗺️"):
    st.markdown(f"<div class='icon-h2'><span class='icon-emoji'>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)

def icon_small(text: str, icon: str = "🗂️"):
    st.markdown(f"<div class='icon-h3'><span class='icon-emoji'>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)

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

# 키워드 파서
HOL_KW = {"seol": ["설","설날","seol"], "chu": ["추","추석","chuseok","chu"], "sub": ["대체","대체공휴","substitute"]}
def contains_any(s: str, keys: List[str]) -> bool:
    s = (s or "").lower()
    return any(k.lower() in s for k in keys)

def normalize_calendar(df: pd.DataFrame):
    """엑셀 표준화 + 카테고리(원본/카운트용/ED용/표시용) 생성."""
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
    d["요일"] = d["날짜"].dt.dayofweek.map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"})

    # 불리언 통일
    def to_bool(x):
        s = str(x).strip().upper()
        return True if s == "TRUE" else False
    for col in ["공휴일여부","명절여부"]:
        if col in d.columns: d[col] = d[col].apply(to_bool)
        else: d[col] = False

    # 공급량 열(있으면 사용)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 1) 1차 분류
    def base_category(row) -> str:
        g = str(row.get("구분","")); y = row["요일"]
        if contains_any(g, HOL_KW["seol"]) or (row.get("명절여부", False) and row["월"] in (1,2)): return "명절_설날"
        if contains_any(g, HOL_KW["chu"])  or (row.get("명절여부", False) and row["월"] in (9,10)): return "명절_추석"
        if ("공휴" in g) or contains_any(g, HOL_KW["sub"]) or row.get("공휴일여부", False):        return "공휴일_대체"
        if y=="토": return "토요일"
        if y=="일": return "일요일"
        if y in ["화","수","목"]: return "평일_1"
        if y in ["월","금"]:     return "평일_2"
        return "평일_1"
    d["카테고리_SRC"] = d.apply(base_category, axis=1)

    # 2) 대체휴일 사유(설/추)
    def sub_reason(row) -> Optional[str]:
        if row["카테고리_SRC"] != "공휴일_대체": return None
        g = str(row.get("구분",""))
        if contains_any(g, HOL_KW["seol"]): return "설"
        if contains_any(g, HOL_KW["chu"]):  return "추"
        return None
    d["대체_사유"] = d.apply(sub_reason, axis=1)

    # 3) 카운트/ED용 카테고리(명절 대체는 명절로 귀속)
    def cat_for_count(row):
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "설": return "명절_설날"
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "추": return "명절_추석"
        return row["카테고리_SRC"]
    d["카테고리_CNT"] = d.apply(cat_for_count, axis=1)
    d["카테고리_ED"]  = d["카테고리_CNT"]

    # 매트릭스 라벨/색
    def label_for_matrix(row):
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "설": return "설*"
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "추": return "추*"
        return CAT_SHORT.get(row["카테고리_CNT"], "")
    d["카테고리_표시"] = d.apply(label_for_matrix, axis=1)
    d["카테고리_색"] = d["카테고리_CNT"].map(lambda k: PALETTE.get(k, "#EEEEEE"))

    for col in ["카테고리_SRC","카테고리_CNT","카테고리_ED"]:
        d[col] = pd.Categorical(d[col], categories=CATS)

    return d, supply_col

def compute_weights_monthly(df: pd.DataFrame, supply_col: Optional[str], cat_col="카테고리_ED",
                            base_cat="평일_1", cap_holiday=CAP_HOLIDAY) -> Tuple[pd.DataFrame, Dict[str,float]]:
    W = []
    for m in range(1,13):
        sub = df[df["월"]==m]
        if sub.empty:
            W.append(pd.Series({c: np.nan for c in CATS}, name=m)); continue
        if (supply_col is None) or sub[sub[cat_col]==base_cat].empty:
            W.append(pd.Series({**{c: np.nan for c in CATS}, base_cat: 1.0}, name=m)); continue
        base_med = sub.loc[sub[cat_col]==base_cat, supply_col].median()
        row = {}
        for c in CATS:
            if c==base_cat: row[c]=1.0; continue
            s = sub.loc[sub[cat_col]==c, supply_col]
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

def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame, count_col="카테고리_CNT") -> pd.DataFrame:
    counts = (df.pivot_table(index=["연","월"], columns=count_col, values="날짜", aggfunc="count")
                .reindex(columns=CATS, fill_value=0).astype(int))
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c]*month_idx.map(weights_monthly[c]).values
    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")
    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)

    aux = df.assign(_cnt=1)
    sub_s = aux[(aux["카테고리_SRC"]=="공휴일_대체") & (aux["대체_사유"]=="설")]\
            .groupby(["연","월"])["_cnt"].sum().rename("대체_설").astype(int)
    sub_c = aux[(aux["카테고리_SRC"]=="공휴일_대체") & (aux["대체_사유"]=="추")]\
            .groupby(["연","월"])["_cnt"].sum().rename("대체_추").astype(int)
    out = out.join(sub_s, how="left").join(sub_c, how="left").fillna({"대체_설":0,"대체_추":0})

    def remark_row(r):
        notes=[]
        if r.get("일수_명절_설날",0)>0:
            add=f"(대체 {int(r['대체_설'])} 포함)" if r.get("대체_설",0)>0 else ""
            notes.append(f"설연휴 {int(r['일수_명절_설날'])}일 {add}".strip())
        if r.get("일수_명절_추석",0)>0:
            add=f"(대체 {int(r['대체_추'])} 포함)" if r.get("대체_추",0)>0 else ""
            notes.append(f"추석연휴 {int(r['일수_명절_추석'])}일 {add}".strip())
        only_sub=int(r.get("일수_공휴일_대체",0))-int(r.get("대체_설",0))-int(r.get("대체_추",0))
        if only_sub>0: notes.append(f"대체공휴일 {only_sub}일")
        return " · ".join([n for n in notes if n])
    out["비고"]=out.apply(remark_row,axis=1)
    return out.reset_index()

def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    months = range(1,13); days = range(1,32)
    fig, ax = plt.subplots(figsize=(13,7))
    ax.set_xlim(0,12); ax.set_ylim(0,31)
    ax.set_xticks([i+0.5 for i in range(12)]); ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i+0.5 for i in range(31)]); ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis(); ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)
    for x in range(13): ax.plot([x,x],[0,31], color="#D0D5DB", lw=0.8)
    for y in range(32): ax.plot([0,12],[y,y], color="#D0D5DB", lw=0.8)
    for j,m in enumerate(months):
        for i,d in enumerate(days):
            row = df_year[(df_year["월"]==m) & (df_year["일"]==d)]
            if row.empty: continue
            label=row.iloc[0]["카테고리_표시"]; color=row.iloc[0]["카테고리_색"]
            rect=mpl.patches.Rectangle((j,i),1,1,color=color,alpha=0.95); ax.add_patch(rect)
            ax.text(j+0.5,i+0.5,label,ha="center",va="center",fontsize=9,
                    color="white" if label in ["설","추","설*","추*","휴"] else "black", fontweight="bold")
    handles=[mpl.patches.Patch(color=PALETTE[c], label=f"{c} ({weights.get(c,1):.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

def center_html(df: pd.DataFrame, width_px: int = 1100, float4: Optional[List[str]] = None, int_cols: Optional[List[str]] = None) -> str:
    float4 = float4 or []; int_cols = int_cols or []
    sty = df.style.set_table_styles([
        {"selector":"th","props":"text-align:center; font-weight:600;"},
        {"selector":"td","props":"text-align:center;"},
        {"selector":"table","props":f"margin-left:auto; margin-right:auto; width:{width_px}px; border-collapse:collapse;"},
    ])
    sty = sty.hide(axis="index")
    for c in float4:
        if c in df.columns: sty = sty.format({c:"{:.4f}"})
    for c in int_cols:
        if c in df.columns: sty = sty.format({c:"{:.0f}"})
    return sty.to_html()

# ───────────────────────── UI ─────────────────────────
# (NEW) 아이콘 타이틀
icon_title(TITLE, "🧩")
st.caption(DESC)

# 분석 시작 버튼 상태
if "ran" not in st.session_state: st.session_state.ran = False

with st.sidebar:
    # (NEW) 사이드바 섹션 아이콘 헤더
    icon_small("데이터 소스", "🗂️")
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

    st.markdown("---")
    icon_small("예측 기간", "⏱️")
    years = list(range(2026, 2031))
    colA, colB = st.columns(2)
    with colA: y_start = st.selectbox("예측 시작(연)", years, index=0, key="ys")
    with colB: m_start = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="ms")
    colC, colD = st.columns(2)
    with colC: y_end = st.selectbox("예측 종료(연)", years, index=1, key="ye")
    with colD: m_end = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="me")

    if st.button("분석 시작", type="primary"): st.session_state.ran = True

if not st.session_state.ran: st.stop()

# ───────────────────────── 데이터 로드 & 전처리 ─────────────────────────
default_path = Path("data") / "effective_days_calendar.xlsx"
raw = pd.read_excel(file if 'file' in locals() and file is not None else default_path, engine="openpyxl")
base_df, supply_col = normalize_calendar(raw)

W_monthly, W_global = compute_weights_monthly(base_df, supply_col, cat_col="카테고리_ED", base_cat="평일_1", cap_holiday=CAP_HOLIDAY)

start_ts = pd.Timestamp(int(y_start), int(m_start), 1)
end_ts   = pd.Timestamp(int(y_end),   int(m_end),   1)
mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택한 예측 구간에 해당하는 날짜가 엑셀에 없어.")
    st.stop()

# ───────────────────────── 매트릭스 ─────────────────────────
icon_section("유효일수 카테고리 매트릭스", "🗺️")
years_in_range = sorted(pred_df["연"].unique().tolist())
c_sel, _ = st.columns([1, 9])
with c_sel:
    show_year = st.selectbox("매트릭스 표시 연도", years_in_range, index=0, key="matrix_year")
fig = draw_calendar_matrix(show_year, pred_df[pred_df["연"]==show_year], W_global)
st.pyplot(fig, clear_figure=True)

# ───────────────────────── 가중치 요약 (표+설명 더 가까이) ─────────────────────────
icon_section("카테고리 가중치 요약", "⚖️")
col_table, col_desc = st.columns([0.5, 1.05], gap="small")

with col_table:
    w_show = pd.DataFrame({"카테고리": CATS, "전역 가중치(중앙값)": [round(W_global[c],4) for c in CATS]})
    html = center_html(w_show, width_px=540, float4=["전역 가중치(중앙값)"])
    st.markdown(html, unsafe_allow_html=True)

with col_desc:
    st.markdown(
        f"""
**유효일수 산정(간단 설명)**  
- 월별 기준카테고리(평일_1) 중앙값 \(Med_{{m,평1}}\), 카테고리 \(c\) 중앙값 \(Med_{{m,c}}\) ⇒ **월별 가중치** \(w_{{m,c}}=Med_{{m,c}}/Med_{{m,평1}}\)  
- 표본 부족 시 전역 중앙값/기본값 보강, **휴일·명절 상한 \(\\le {CAP_HOLIDAY:.2f}\)** 적용  
- **설/추석 유래 대체휴일**은 해당 명절로 귀속(매트릭스: `설*`, `추*`)  
- **월별 유효일수** \(ED_m=\sum_c (\text{{해당월 일수}}_c \times w_{{m,c}})\)
"""
    )

# ───────────────────────── 월별 유효일수 표 + 좌측하단 CSV ─────────────────────────
icon_section("월별 유효일수 요약", "📊")
eff_tbl = effective_days_by_month(pred_df, W_monthly, count_col="카테고리_CNT")

show_cols = (["연","월","월일수"] + [f"일수_{c}" for c in CATS] + ["유효일수합","적용_비율(유효/월일수)","비고"])
eff_show = eff_tbl[show_cols].sort_values(["연","월"]).reset_index(drop=True)

float4_cols = ["유효일수합","적용_비율(유효/월일수)"]
int_cols = [c for c in eff_show.columns if c not in float4_cols+["비고"]]
html2 = center_html(eff_show, width_px=1180, float4=float4_cols, int_cols=int_cols)
st.markdown(html2, unsafe_allow_html=True)

left_dl, _ = st.columns([1, 9])
with left_dl:
    csv_bytes = eff_show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="월별 유효일수 CSV 다운로드",
        data=csv_bytes,
        file_name="effective_days_summary.csv",
        mime="text/csv",
        use_container_width=False,
    )
