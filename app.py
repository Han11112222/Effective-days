# app.py — Effective Days (공휴일 표시 복원 · 옵션 반영 · 매트릭스 해치 표기 · 표 소수2자리 고정)
# 2025-09-15 업데이트:
#  - 예측 기간 UI를 데이터 범위 기반(최소 2015년)으로 구성
#  - 임시공휴일을 일반 공휴일(‘휴’)로 표시
#  - 월별 요약 CSV + 일자별(매트릭스 동일) CSV 다운로드
#  - 매트릭스(가중치 숫자) 엑셀 다운로드: xlsxwriter 없을 때 openpyxl/임시파일로 폴백

import os
from io import BytesIO
from tempfile import NamedTemporaryFile
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
    "표본 부족 시 전역 중앙값/기본값을 사용하며, 휴일/명절 가중치에는 상한을 적용합니다."
)

CATS: List[str] = ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]
CAT_SHORT: Dict[str, str] = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}
# 범례용 표시명(공휴일_대체 → 공휴일)
CATS_DISPLAY: Dict[str, str] = {c: ("공휴일" if c=="공휴일_대체" else c) for c in CATS}
PALETTE = {
    "평일_1":"#7DC3C1","평일_2":"#3DA4AB","토요일":"#5D6D7E","일요일":"#34495E",
    "공휴일_대체":"#E57373","명절_설날":"#F5C04A","명절_추석":"#F39C12",
}
# 기본 가중치(표본 부족 보강용)
DEFAULT_WEIGHTS = {"평일_1":1.0,"평일_2":0.9713,"토요일":0.8566,"일요일":0.7651,"공휴일_대체":0.8410,"명절_설날":0.8381,"명절_추석":0.7990}
CAP_HOLIDAY = 0.90  # 휴일·명절 상한

MIN_YEAR_UI = 2015  # UI 선택 최소 연도

# ─────────────────────── 아이콘 헤더 CSS/함수 ───────────────────────
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
def icon_title(text: str, icon: str = "🧩"):  st.markdown(f"<div class='icon-h1'><span class='icon-emoji'>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)
def icon_section(text: str, icon: str = "🗺️"): st.markdown(f"<div class='icon-h2'><span class='icon-emoji'>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)
def icon_small(text: str, icon: str = "🗂️"):   st.markdown(f"<div class='icon-h3'><span class='icon-emoji'>{icon}</span><span>{text}</span></div>", unsafe_allow_html=True)

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
    if len(s) == 8 and s.isdigit(): return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def to_bool(x) -> bool:
    s = str(x).strip().upper()
    return s in {"TRUE","T","Y","YES","1"}

HOL_KW = {"seol": ["설","설날","seol"], "chu": ["추","추석","chuseok","chu"], "sub": ["대체","대체공휴","substitute"]}
TEMP_KW = ["임시","임시공휴","임시공휴일","temporary"]  # 임시공휴일 키워드

def contains_any(s: str, keys: List[str]) -> bool:
    s = (s or "").lower()
    return any(k.lower() in s for k in keys)

def in_lny_window(month: int, day: int) -> bool:
    return (month == 1 and day >= 20) or (month == 2 and day <= 20)

# ───────────── 캘린더 정규화 ─────────────
def normalize_calendar(df: pd.DataFrame):
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 열 추출
    date_col = None
    for c in d.columns:
        if str(c).lower() in ["날짜","일자","date"]: date_col = c; break
    if date_col is None:
        for c in d.columns:
            try:
                if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.9: date_col = c; break
            except Exception:
                pass
    if date_col is None: raise ValueError("날짜 열을 찾지 못했습니다. (예: 날짜/일자/date/yyyymmdd)")

    d["날짜"] = d[date_col].map(to_date)
    d = d.dropna(subset=["날짜"]).copy()
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    d["일"] = d["날짜"].dt.day.astype(int)
    d["요일"] = d["날짜"].dt.dayofweek.map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"})

    # 불리언 통일
    d["공휴일여부"] = d["공휴일여부"].apply(to_bool) if "공휴일여부" in d.columns else False
    d["명절여부"]   = d["명절여부"].apply(to_bool)   if "명절여부"   in d.columns else False

    # 공급량 열(있으면 사용)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]): supply_col = c; break

    # 1) 1차 분류
    def base_category(row) -> str:
        g = str(row.get("구분",""))
        y = row["요일"]; m = int(row["월"]); day = int(row["일"])
        has_seol = contains_any(g, HOL_KW["seol"])
        has_chu  = contains_any(g, HOL_KW["chu"])
        is_pub   = bool(row.get("공휴일여부", False)) or contains_any(g, TEMP_KW)  # 임시공휴일 포함

        if has_seol:
            return "명절_설날" if in_lny_window(m, day) and not (m==1 and day==1) else "공휴일_대체"
        if has_chu:
            return "명절_추석" if m == 9 else "공휴일_대체"

        if row.get("명절여부", False):
            if in_lny_window(m, day) and not (m==1 and day==1): return "명절_설날"
            if m == 9: return "명절_추석"

        if is_pub: return "공휴일_대체"

        if y=="토": return "토요일"
        if y=="일": return "일요일"
        if y in ["화","수","목"]: return "평일_1"
        if y in ["월","금"]:     return "평일_2"
        return "평일_1"
    d["카테고리_SRC"] = d.apply(base_category, axis=1)

    # 2) 대체휴일 사유(설/추) — 표기용
    def sub_reason(row) -> Optional[str]:
        if row["카테고리_SRC"] != "공휴일_대체": return None
        g = str(row.get("구분","")); m = int(row["월"]); day = int(row["일"])
        if contains_any(g, HOL_KW["seol"]) and in_lny_window(m, day) and not (m==1 and day==1): return "설"
        if contains_any(g, HOL_KW["chu"]) and m == 9: return "추"
        return None
    d["대체_사유"] = d.apply(sub_reason, axis=1)

    # 3) 강제 오버라이드
    jan1 = (d["월"]==1) & (d["일"]==1)
    d.loc[jan1, ["카테고리_SRC","대체_사유"]] = ["공휴일_대체", None]
    mask_oct_2627 = (d["월"]==10) & (d["연"].isin([2026, 2027]))
    d.loc[mask_oct_2627 & (d["카테고리_SRC"]=="명절_추석"), "카테고리_SRC"] = "공휴일_대체"
    d.loc[mask_oct_2627 & (d["대체_사유"]=="추"), "대체_사유"] = None

    # 4) 카운트/ED용 카테고리(명절 대체는 명절로 귀속)
    def cat_for_count(row):
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "설": return "명절_설날"
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "추": return "명절_추석"
        return row["카테고리_SRC"]
    d["카테고리_CNT"] = d.apply(cat_for_count, axis=1)
    d["카테고리_ED"]  = d["카테고리_CNT"]

    # 5) 매트릭스 라벨/색
    def label_for_matrix(row):
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "설": return "설*"
        if row["카테고리_SRC"] == "공휴일_대체" and row["대체_사유"] == "추": return "추*"
        return CAT_SHORT.get(row["카테고리_CNT"], "")
    d["카테고리_표시"] = d.apply(label_for_matrix, axis=1)
    d["카테고리_색"] = d["카테고리_CNT"].map(lambda k: PALETTE.get(k, "#EEEEEE"))

    for col in ["카테고리_SRC","카테고리_CNT","카테고리_ED"]:
        d[col] = pd.Categorical(d[col], categories=CATS)

    return d, supply_col

# ───────────── 가중치 계산 ─────────────
def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    cat_col="카테고리_ED",
    base_cat="평일_1",
    cap_holiday=CAP_HOLIDAY,
    ignore_substitute_in_weights: bool = True,
) -> Tuple[pd.DataFrame, Dict[str,float]]:

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
            if c == base_cat:
                row[c] = 1.0; continue
            s_sub = sub[sub[cat_col]==c]
            if ignore_substitute_in_weights and c in ("명절_설날","명절_추석"):
                s_sub = s_sub[s_sub["카테고리_SRC"] != "공휴일_대체"]  # 설*/추* 표본 제외
            s = s_sub[supply_col] if (supply_col and not s_sub.empty) else pd.Series(dtype=float)
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

# ───────────── 월별 유효일수 ─────────────
def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame, count_col="카테고리_CNT") -> pd.DataFrame:
    counts = (df.pivot_table(index=["연","월"], columns=count_col, values="날짜", aggfunc="count")
                .reindex(columns=CATS, fill_value=0).astype(int))
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values
    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")

    # 카테고리별 효과(일수 × (w-1)) 및 총합
    effect = counts.copy().astype(float)
    for c in CATS:
        w = month_idx.map(weights_monthly[c]).values
        effect[c] = counts[c] * (w - 1.0)
    effect_sum = effect.sum(axis=1).rename("총효과(Σ일수×(w-1))")

    out = pd.concat([
        month_days,
        counts.add_prefix("일수_"),
        eff.add_prefix("적용_"),
        effect.add_prefix("효과_"),
        eff_sum,
        effect_sum,
    ], axis=1)
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"])

    # 대체휴일 메모
    aux = df.assign(_cnt=1)
    sub_s = aux[(aux["카테고리_SRC"]=="공휴일_대체") & (aux["대체_사유"]=="설")].groupby(["연","월"])['_cnt'].sum().rename("대체_설").astype(int)
    sub_c = aux[(aux["카테고리_SRC"]=="공휴일_대체") & (aux["대체_사유"]=="추")].groupby(["연","월"])['_cnt'].sum().rename("대체_추").astype(int)
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

# ───────────── 캘린더 그림 ─────────────
def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float], highlight_sub_samples: bool=False):
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
            r = row.iloc[0]
            label = r["카테고리_표시"]; color = r["카테고리_색"]
            hatch = None; edgecolor = None; lw = 0.0
            if highlight_sub_samples and (r["카테고리_SRC"]=="공휴일_대체") and (r["대체_사유"] in ("설","추")):
                hatch = "////"; edgecolor = "black"; lw = 1.2
            rect = mpl.patches.Rectangle((j,i),1,1, facecolor=color, edgecolor=edgecolor, linewidth=lw, hatch=hatch, alpha=0.95)
            ax.add_patch(rect)
            ax.text(j+0.5,i+0.5,label,ha="center",va="center",fontsize=9,
                    color="white" if label in ["설","추","설*","추*","휴"] else "black", fontweight="bold")

    handles=[mpl.patches.Patch(color=PALETTE[c], label=f"{CATS_DISPLAY[c]} ({weights.get(c,1):.3f})") for c in CATS]
    if highlight_sub_samples:
        handles.append(mpl.patches.Patch(facecolor="white", edgecolor="black", hatch="////", label="가중치 제외 표본(설*/추*)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

# ───────────── 표 렌더링 ─────────────
def center_html(df: pd.DataFrame, width_px: int = 1100, formats: Optional[Dict[str,str]] = None, int_cols: Optional[List[str]] = None) -> str:
    int_cols = int_cols or []
    sty = df.style.set_table_styles([
        {"selector":"th","props":"text-align:center; font-weight:600;"},
        {"selector":"td","props":"text-align:center;"},
        {"selector":"table","props":f"margin-left:auto; margin-right:auto; width:{width_px}px; border-collapse:collapse;"},
    ])
    sty = sty.hide(axis="index")
    if formats: sty = sty.format(formats)
    for c in int_cols:
        if c in df.columns: sty = sty.format({c:"{:.0f}"})
    return sty.to_html()

# ───────────────────────── UI ─────────────────────────
icon_title(TITLE, "🧩")
st.caption(DESC)

if "ran" not in st.session_state: st.session_state.ran = False

with st.sidebar:
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
    icon_small("옵션", "⚙️")
    opt_ignore_sub = st.checkbox("명절 가중치 계산에서 설/추 대체공휴일 제외", value=True)
    st.caption("✓ 체크 시 설·추 가중치 계산에서 ‘설*/추*’ 표본은 제외됩니다(일수 집계엔 포함).")

    st.markdown("---")
    icon_small("예측 기간", "⏱️")

    # 파일의 실제 연도 범위를 읽어 UI 범위로 사용(최소 2015년)
    def compute_year_options(_file) -> List[int]:
        try:
            if hasattr(_file, "seek"): _file.seek(0)
            raw_preview = pd.read_excel(_file if _file is not None else default_path, engine="openpyxl")
            base_preview, _ = normalize_calendar(raw_preview)
            years_all = sorted(set(base_preview["연"].tolist()))
            if not years_all:
                return list(range(MIN_YEAR_UI, MIN_YEAR_UI + 16))
            min_y, max_y = min(years_all), max(years_all)
            min_y = min(min_y, MIN_YEAR_UI)
            return list(range(min_y, max(max_y, MIN_YEAR_UI) + 5))  # +4년 버퍼
        except Exception:
            return list(range(MIN_YEAR_UI, MIN_YEAR_UI + 16))

    years = compute_year_options(file)

    def safe_index(lst, val, fallback=0):
        try: return lst.index(val)
        except ValueError: return fallback

    colA, colB = st.columns(2)
    with colA: y_start = st.selectbox("예측 시작(연)", years, index=safe_index(years, MIN_YEAR_UI), key="ys")
    with colB: m_start = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="ms")  # 1월
    colC, colD = st.columns(2)
    with colC: y_end = st.selectbox("예측 종료(연)", years, index=len(years)-1 if len(years)>1 else 0, key="ye")
    with colD: m_end = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="me")  # 12월

    if st.button("분석 시작", type="primary"): st.session_state.ran = True

if not st.session_state.ran: st.stop()

# ───────────────────────── 데이터 로드 & 전처리 ─────────────────────────
try:
    if 'file' in locals() and hasattr(file, 'seek'): file.seek(0)
except Exception:
    pass

default_path = Path("data") / "effective_days_calendar.xlsx"
raw = pd.read_excel(file if 'file' in locals() and file is not None else default_path, engine="openpyxl")
base_df, supply_col = normalize_calendar(raw)

W_monthly, W_global = compute_weights_monthly(
    base_df, supply_col,
    cat_col="카테고리_ED",
    base_cat="평일_1",
    cap_holiday=CAP_HOLIDAY,
    ignore_substitute_in_weights=opt_ignore_sub
)

# 표시 구간
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
fig = draw_calendar_matrix(show_year, pred_df[pred_df["연"]==show_year], W_global, highlight_sub_samples=opt_ignore_sub)
st.pyplot(fig, clear_figure=True)

# ───────────────────────── 가중치 요약 ─────────────────────────
icon_section("카테고리 가중치 요약", "⚖️")
col_table, col_desc = st.columns([0.5, 1.05], gap="small")

with col_table:
    w_show = pd.DataFrame({"카테고리": [CATS_DISPLAY[c] for c in CATS], "전역 가중치(중앙값)": [W_global[c] for c in CATS]})
    html = center_html(w_show, width_px=540, formats={"전역 가중치(중앙값)":"{:.4f}"})
    st.markdown(html, unsafe_allow_html=True)

with col_desc:
    st.markdown(
        f"""
**유효일수 산정 요약**  
- 월별 기준카테고리(**평일_1: 화·수·목**) 중앙값 \(Med_{{m,평1}}\), 카테고리 \(c\) 중앙값 \(Med_{{m,c}}\) ⇒ **월별 가중치** \(w_{{m,c}}=Med_{{m,c}}/Med_{{m,평1}}\)  
- 표본 부족 시 전역 중앙값/기본값 보강, **휴일·명절 상한 \(\\le {CAP_HOLIDAY:.2f}\)** 적용  
- **설/추 대체공휴일(설*/추*)**: **일수 집계 포함**, **가중치 계산은 옵션에 따라 제외(기본)**  
- **월별 유효일수** \(ED_m=\sum_c (\text{{해당월 일수}}_c \times w_{{m,c}})\)
"""
    )

# ───────────────────────── 월별 유효일수 표 + 다운로드 ─────────────────────────
icon_section("월별 유효일수 요약", "📊")
eff_tbl = effective_days_by_month(pred_df, W_monthly, count_col="카테고리_CNT")

show_cols = (["연","월","월일수"] + [f"일수_{c}" for c in CATS] + ["유효일수합","적용_비율(유효/월일수)","비고"])
eff_show = eff_tbl[show_cols].sort_values(["연","월"]).reset_index(drop=True)

# 화면 표시는 두 열만 소수 2자리 문자열로 고정
eff_disp = eff_show.copy()
eff_disp["유효일수합"] = eff_disp["유효일수합"].map(lambda x: f"{x:.2f}")
eff_disp["적용_비율(유효/월일수)"] = eff_disp["적용_비율(유효/월일수)"].map(lambda x: f"{x:.2f}")

formats = {"유효일수합":"{}", "적용_비율(유효/월일수)":"{}"}
int_cols = [c for c in eff_disp.columns if c not in ["유효일수합","적용_비율(유효/월일수)","비고"]]
html2 = center_html(eff_disp, width_px=1180, formats=formats, int_cols=int_cols)
st.markdown(html2, unsafe_allow_html=True)

left_dl, right_dl = st.columns([1, 1])
with left_dl:
    # 월별 요약 CSV (효과 열 포함)
    csv_bytes = eff_tbl.sort_values(["연","월"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="월별 유효일수 CSV 다운로드(효과 포함)",
        data=csv_bytes,
        file_name="effective_days_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right_dl:
    # 일자별 카테고리/가중치 CSV (매트릭스 동일)
    daily = pred_df.copy()
    def weight_row(r):
        m = int(r["월"]); c = r["카테고리_ED"]
        try:
            return float(W_monthly.loc[m, c])
        except Exception:
            return float(W_global.get(c, 1.0))
    daily["적용_가중치"] = daily.apply(weight_row, axis=1)
    daily["Δ(가중치-1)"] = daily["적용_가중치"] - 1.0
    daily["공휴일표현"] = np.where(
        (daily["카테고리_SRC"]=="공휴일_대체") & daily["대체_사유"].isna(), "공휴일",
        np.where((daily["카테고리_SRC"]=="공휴일_대체") & daily["대체_사유"].notna(),
                 "대체공휴일(" + daily["대체_사유"].astype(str) + ")", "")
    )
    daily_export = daily[["날짜","연","월","일","요일","카테고리_CNT","카테고리_표시","공휴일표현","적용_가중치","Δ(가중치-1)"]].copy()
    daily_bytes = daily_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="일자별 카테고리/가중치 CSV 다운로드(매트릭스 동일)",
        data=daily_bytes,
        file_name="effective_days_calendar_detail.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ───────────────────────── 매트릭스(가중치 숫자) 엑셀 내보내기 ─────────────────────────
def build_year_matrix_numeric(df: pd.DataFrame, weights_monthly: pd.DataFrame, year: int) -> pd.DataFrame:
    df_y = df[df["연"]==year][["월","일","카테고리_CNT"]].copy()
    df_y["가중치"] = df_y.apply(lambda r: float(weights_monthly.loc[int(r["월"]), r["카테고리_CNT"]]), axis=1)
    mat = df_y.pivot(index="일", columns="월", values="가중치").reindex(index=range(1,32), columns=range(1,13))
    return mat

def _write_excel_content(writer):
    # 전역 가중치
    gdf = pd.DataFrame({"카테고리": [CATS_DISPLAY[c] for c in CATS], "전역 가중치(중앙값)": [W_global[c] for c in CATS]}).round(4)
    gdf.to_excel(writer, sheet_name="가중치요약", index=False)
    # 월별 가중치
    Wm_out = W_monthly.copy()
    Wm_out.index = [f"{m}월" for m in Wm_out.index]
    Wm_out = Wm_out[[c for c in CATS if c in Wm_out.columns]].round(4)
    Wm_out.columns = [CATS_DISPLAY[c] for c in Wm_out.columns]
    Wm_out.to_excel(writer, sheet_name="월별가중치")
    # 연도별 숫자 매트릭스
    for yy in years_in_range:
        mat = build_year_matrix_numeric(pred_df, W_monthly, yy).round(4)
        mat.columns = [f"{m}월" for m in mat.columns]
        mat.index.name = "일"
        mat.to_excel(writer, sheet_name=str(yy))

def build_excel_bytes() -> bytes:
    # 1차: xlsxwriter 사용 시도
    try:
        import xlsxwriter  # 존재 여부 확인
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            _write_excel_content(writer)
        return buf.getvalue()
    except Exception:
        # 2차: openpyxl로 BytesIO 쓰기 시도
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                _write_excel_content(writer)
            return buf.getvalue()
        except Exception:
            # 3차: 임시파일 폴백
            with NamedTemporaryFile(suffix=".xlsx", delete=True) as tmp:
                with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
                    _write_excel_content(writer)
                tmp.seek(0)
                return tmp.read()

excel_bytes = build_excel_bytes()
st.download_button(
    label="매트릭스(가중치 숫자) 엑셀 다운로드",
    data=excel_bytes,
    file_name=f"effective_days_matrix_{y_start}-{int(m_start):02d}_{y_end}-{int(m_end):02d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=False,
)
