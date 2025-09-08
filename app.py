# app.py — Effective Days (유효일수 분석 전용, v2)
import sys
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
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 상단 환경 버전 표시(디버그 겸)
st.sidebar.info(f"Py {sys.version.split()[0]} · streamlit {st.__version__} · pandas {pd.__version__}")

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
# 카테고리/색/기본가중치
CATS = ["평일_1","평일_2","토요일","일요일","공휴일_대체","명절_설날","명절_추석"]
CAT_SHORT = {"평일_1":"평1","평일_2":"평2","토요일":"토","일요일":"일","공휴일_대체":"휴","명절_설날":"설","명절_추석":"추"}
CAT_DESC = {"평일_1":"평일_1(화·수·목)","평일_2":"평일_2(월·금)","토요일":"토요일","일요일":"일요일","공휴일_대체":"공휴일·대체","명절_설날":"명절(설)","명절_추석":"명절(추석)"}

PALETTE = {
    "평일_1":"#7DC3C1",   # teal light
    "평일_2":"#3DA4AB",
    "토요일":"#5D6D7E",
    "일요일":"#34495E",
    "공휴일_대체":"#E57373", # soft red
    "명절_설날":"#F5C04A",   # warm gold
    "명절_추석":"#F39C12",
}
DEFAULT_WEIGHTS = {  # 데이터 부족 시 초기값
    "평일_1":1.0, "평일_2":0.952, "토요일":0.85, "일요일":0.60,
    "공휴일_대체":0.799, "명절_설날":0.842, "명절_추석":0.799
}

# ❶ 특수 명절 강제 인식 윈도우(연도별)
#   예) (시작, 끝, 카테고리, 비고)
SPECIAL_HOLIDAYS: Dict[int, List[Tuple[str,str,str,str]]] = {
    2026: [
        ("2026-02-14","2026-02-18","명절_설날","설연휴 5일 반영"),
        ("2026-09-24","2026-09-27","명절_추석","추석연휴 4일 반영"),
    ]
}

# ─────────────────────────────────────────────────────────────
# 표 렌더링(가운데 정렬/가운데 배치/폭 제한)
def show_centered_table(df: pd.DataFrame, width_px: int = 980, index: bool=False):
    html = df.to_html(index=index, border=0)
    css = f"""
    <style>
      .tbl-wrap {{ display:flex; justify-content:center; }}
      .tbl-wrap table {{
          width: {width_px}px; margin: 0 auto; table-layout: fixed;
          border-collapse: collapse; font-size: 14px;
      }}
      .tbl-wrap th, .tbl-wrap td {{
          text-align: center; padding: 8px 6px; border-bottom: 1px solid #eaeaea;
          white-space: nowrap;
      }}
      .tbl-wrap thead th {{ background:#f7f7f9; font-weight:600; }}
    </style>
    <div class="tbl-wrap">{html}</div>
    """
    st.markdown(css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 로드/정규화
def to_date(x):
    s = str(x).strip()
    if len(s)==8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame):
    """엑셀 원본을 표준 스키마로 정규화하고 (DataFrame, 공급량컬럼명 or None) 반환"""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜
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

    # 불리언 힌트
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else:
            d[col] = np.nan

    # 공급량 컬럼 추정(없으면 None)
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
        if "추석" in g: return "명절_추석"
        # 명절여부만 있을 때 월로 분기
        if str(row.get("명절여부","")).upper() == "TRUE":
            if mon in (1,2): return "명절_설날"
            if mon in (9,10): return "명절_추석"
        # '명절(설·추석)' 같은 문자열 대응
        if "명절" in g:
            if mon in (1,2): return "명절_설날"
            if mon in (9,10): return "명절_추석"
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

    # ❷ 특수 윈도우 강제 덮어쓰기 + 비고 저장
    d["비고"] = ""
    for yy, windows in SPECIAL_HOLIDAYS.items():
        for s, e, cat, note in windows:
            sdt = pd.to_datetime(s); edt = pd.to_datetime(e)
            mask = (d["날짜"]>=sdt) & (d["날짜"]<=edt)
            d.loc[mask, "카테고리"] = cat
            d.loc[mask, "비고"] = d.loc[mask, "비고"].where(d["비고"]=="", other=d["비고"]+", "+note)
            d.loc[mask & (d["비고"]==""), "비고"] = note

    d["카테고리"] = pd.Categorical(d["카테고리"], categories=CATS, ordered=False)
    return d, supply_col

# ─────────────────────────────────────────────────────────────
# 가중치 계산
def compute_weights_monthly(
    df: pd.DataFrame,
    supply_col: Optional[str],
    base_cat: str = "평일_1",
    cap_holiday: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, float]]:
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

# ─────────────────────────────────────────────────────────────
# 월별 유효일수
def effective_days_by_month(df: pd.DataFrame, weights_monthly: pd.DataFrame) -> pd.DataFrame:
    counts = df.pivot_table(index=["연","월"], columns="카테고리", values="날짜",
                            aggfunc="count").reindex(columns=CATS, fill_value=0).astype(int)
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values
    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")

    # 월별 비고(특수 윈도우 설명 집계)
    notes = df.groupby(["연","월"])["비고"].apply(
        lambda s: ", ".join(sorted({x for x in s if isinstance(x,str) and x.strip()!=""}))
    ).rename("비고")

    out = pd.concat([month_days, counts.add_prefix("일수_"), eff_sum, notes], axis=1).reset_index()
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)
    # 컬럼 순서 정리 + 설명용 한글 라벨
    nice_cols = (["연","월","월일수"] +
                 [f"일수_{c}" for c in CATS] +
                 ["유효일수합","적용_비율(유효/월일수)","비고"])
    return out[nice_cols]

# ─────────────────────────────────────────────────────────────
# 매트릭스
def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    months = range(1,13)
    days = range(1,32)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 31)
    ax.set_xticks([i+0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i+0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

    # 그리드
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
            txt_color = "white" if cat in ["일요일","공휴일_대체","명절_설날","명절_추석"] else "black"
            ax.text(j+0.5, i+0.5, label, ha="center", va="center", fontsize=9, color=txt_color, fontweight="bold")

    # 범례(가중치 병기)
    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_DESC[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="카테고리 (가중치)")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# 사이드바: 파일 업로드만
with st.sidebar:
    st.header("데이터 불러오기")
    src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)
    default_path = Path("data") / "effective_days_calendar.xlsx"
    if src == "Repo 내 파일 사용":
        if default_path.exists():
            st.success(f"레포 파일 사용: {default_path.name}")
            file = open(default_path, "rb")
        else:
            file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
    else:
        file = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])

st.title("📅 Effective Days — 유효일수 분석")
st.caption("월별 가중 유효일수 = Σ(해당일 카테고리 가중치). 가중치는 **같은 달의 ‘평일_1(화·수·목)’ 중앙값 대비** 각 카테고리 중앙값 비율로 산정합니다. (명절/공휴일 상한 0.95)")

if file is None:
    st.warning("엑셀을 업로드하거나 data/effective_days_calendar.xlsx 를 레포에 넣어주세요.")
    st.stop()

# 데이터 로드/정규화
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

# ─ 상단: 예측 구간(기본 ‘향후 3년’ 자동)
years_avail = sorted(base_df["연"].unique().tolist())
min_year, max_year = min(years_avail), max(years_avail)

# 기본 3년 구간: 데이터의 최솟값이 2026 이상이면 2026~2028, 아니면 "최소연도~+2년"
default_start_year = 2026 if 2026 in years_avail else min_year
default_end_year = min(default_start_year + 2, max_year)

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    s_y = st.selectbox("시작 연", options=years_avail, index=years_avail.index(default_start_year))
with c2:
    s_m = st.selectbox("시작 월", options=list(range(1,13)), index=0)
with c3:
    e_y = st.selectbox("종료 연", options=years_avail, index=years_avail.index(default_end_year))
with c4:
    e_m = st.selectbox("종료 월", options=list(range(1,13)), index=11)

start_ts = pd.Timestamp(int(s_y), int(s_m), 1)
end_ts = pd.Timestamp(int(e_y), int(e_m), 1)
if end_ts < start_ts:
    st.error("종료가 시작보다 빠릅니다.")
    st.stop()

mask = (base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts + pd.offsets.MonthEnd(0))
pred_df = base_df.loc[mask].copy()
if pred_df.empty:
    st.error("선택 구간에 해당하는 날짜가 엑셀에 없습니다.")
    st.stop()

# ─ 상단: 매트릭스 연도 선택(작게)
years_in_range = sorted(pred_df["연"].unique().tolist())
mat_year = st.selectbox("매트릭스 표시 연도", years_in_range, index=0, label_visibility="visible")

# 1) 가중치 표(설명 라벨 포함, 가운데 정렬)
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({
    "카테고리": [CAT_DESC[c] for c in CATS],
    "전역 가중치(중앙값)": [round(W_global[c], 4) for c in CATS]
})
show_centered_table(w_show, width_px=460, index=False)

st.caption("※ 가중치는 달별 ‘평일_1’ 대비 각 카테고리 중앙값 비율을 다시 중앙값으로 취한 값입니다. 데이터가 보수적이면 명절 가중치가 낮아질 수 있습니다.")

# 2) 매트릭스(선택 연도)
fig = draw_calendar_matrix(mat_year, pred_df[pred_df["연"]==mat_year], W_global)
st.pyplot(fig, clear_figure=True)

# 3) 월별 유효일수 요약(비고 포함, 가운데 정렬)
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly).sort_values(["연","월"])
show_centered_table(eff_tbl, width_px=1100, index=False)

# CSV 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl.to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv"
)
