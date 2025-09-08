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
st.set_page_config(page_title="Effective Days · 유효일수 분석", page_icon="📅", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 전체 테이블 가독성(중앙정렬/적당한 폭) CSS
st.markdown("""
<style>
/* st.dataframe 텍스트 중앙정렬 */
[data-testid="stDataFrame"] div[role="gridcell"] { justify-content: center !important; }
/* 표가 좌우로 너무 넓어지지 않도록 컨테이너 폭 제한 */
.block-container { max-width: 1250px; }
</style>
""", unsafe_allow_html=True)

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
# 분류/색상
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
DEFAULT_WEIGHTS = {  # 데이터 부족 시 초기값
    "평일_1":1.0, "평일_2":0.952, "토요일":0.85, "일요일":0.60,
    "공휴일_대체":0.799, "명절_설날":0.842, "명절_추석":0.799
}

# ─────────────────────────────────────────────────────────────
# 유틸 함수
def to_date(x):
    s = str(x).strip()
    if len(s)==8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def normalize_calendar(df: pd.DataFrame):
    """
    엑셀을 표준 스키마로 정규화.
    반환: (정규화 DataFrame, 공급량 컬럼명 or None)
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # 날짜 열 추정
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

    # 불리언 힌트 표준화
    for col in ["주중여부","주말여부","공휴일여부","명절여부"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.upper().map({"TRUE":True,"FALSE":False})
        else:
            d[col] = np.nan

    # 공급량 컬럼 추정(없어도 동작)
    supply_col = None
    for c in d.columns:
        if ("공급" in str(c)) and pd.api.types.is_numeric_dtype(d[c]):
            supply_col = c; break

    # 카테고리 매핑
    def infer_festival(row):
        g = str(row.get("구분",""))
        mon = int(row["월"])
        if "설" in g: return "명절_설날"
        if "추" in g: return "명절_추석"
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
    월별 가중치 계산:
    - 같은 '월'에서 base_cat(평일_1)의 공급량 중앙값을 기준으로
      각 카테고리 중앙값/베이스 중앙값 = 가중치
    - 부족 데이터는 전월/전역 중앙값 및 DEFAULT로 보강
    - 명절/공휴일 가중치 상한 cap_holiday 적용(보수적)
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
                row[c]=1.0
            else:
                s = sub.loc[sub["카테고리"]==c, supply_col]
                row[c] = float(s.median()/base_med) if (len(s)>0 and base_med>0) else np.nan
        W.append(pd.Series(row, name=m))
    W = pd.DataFrame(W)  # index=월

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
    """
    월별 카테고리 일수, 가중 유효일수 합계, 비고(설/추석/대체 연속 구간 요약)를 생성
    """
    def _format_ranges(days: list[int]) -> str:
        if not days: return ""
        days = sorted(days)
        ranges, start, prev = [], days[0], days[0]
        for d in days[1:]:
            if d == prev + 1:
                prev = d
            else:
                ranges.append((start, prev))
                start = prev = d
        ranges.append((start, prev))
        parts = [f"{a}–{b}" if a!=b else f"{a}" for a,b in ranges]
        return ", ".join(parts)

    # 개수/유효일수
    counts = df.pivot_table(index=["연","월"], columns="카테고리", values="날짜",
                            aggfunc="count").reindex(columns=CATS, fill_value=0).astype(int)
    eff = counts.copy().astype(float)
    month_idx = counts.index.get_level_values("월")
    for c in CATS:
        eff[c] = eff[c] * month_idx.map(weights_monthly[c]).values

    eff_sum = eff.sum(axis=1).rename("유효일수합")
    month_days = df.groupby(["연","월"])["날짜"].nunique().rename("월일수")

    out = pd.concat([month_days, counts.add_prefix("일수_"), eff.add_prefix("적용_"), eff_sum], axis=1).reset_index()
    out["적용_비율(유효/월일수)"] = (out["유효일수합"]/out["월일수"]).round(4)

    # 비고(설/추석/대체 날짜 요약)
    notes = []
    for _, r in out.iterrows():
        y, m = int(r["연"]), int(r["월"])
        mon_df = df[(df["연"]==y) & (df["월"]==m)]
        msg = []
        s_days = mon_df.loc[mon_df["카테고리"]=="명절_설날","날짜"].dt.day.tolist()
        if s_days: msg.append(f"설({len(s_days)}일: {m}/{_format_ranges(s_days)})")
        c_days = mon_df.loc[mon_df["카테고리"]=="명절_추석","날짜"].dt.day.tolist()
        if c_days: msg.append(f"추석({len(c_days)}일: {m}/{_format_ranges(c_days)})")
        h_days = mon_df.loc[mon_df["카테고리"]=="공휴일_대체","날짜"].dt.day.tolist()
        if h_days: msg.append(f"대체({len(h_days)}일: {m}/{_format_ranges(h_days)})")
        notes.append(" · ".join(msg))
    out["비고"] = notes
    return out

def draw_calendar_matrix(year: int, df_year: pd.DataFrame, weights: Dict[str,float]):
    """
    12x31 매트릭스 캘린더(월=열, 일=행)
    """
    months = range(1,13)
    days = range(1,32)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 31)
    ax.set_xticks([i+0.5 for i in range(12)])
    ax.set_xticklabels([f"{m}월" for m in months], fontsize=11)
    ax.set_yticks([i+0.5 for i in range(31)])
    ax.set_yticklabels([f"{d}" for d in days], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{year} 유효일수 카테고리 매트릭스", fontsize=16, pad=10)

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
                    fontsize=9, color="white" if cat in ["일요일","공휴일_대체","명절_설날","명절_추석"] else "black",
                    fontweight="bold")

    handles = [mpl.patches.Patch(color=PALETTE[c], label=f"{CAT_LABEL[c]} ({weights[c]:.3f})") for c in CATS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="카테고리 (가중치)")

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# 사이드바 — 데이터/기간 & 버튼
with st.sidebar:
    st.title("📅 Effective Days")
    st.caption("월별 유효일수 = Σ(해당일 카테고리 가중치). "
               "가중치는 같은 달의 ‘평일_1(화·수·목)’ 대비 중앙값 비율로 산정합니다. (명절/공휴일 상한 0.95)")

    st.subheader("데이터 불러오기")
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

    st.subheader("예측 기간")
    years = list(range(2026, 2031))  # 2026~2030
    start_y = st.selectbox("시작 연", years, index=0)
    start_m = st.selectbox("시작 월", list(range(1,13)), index=0)
    end_y   = st.selectbox("종료 연", years, index=1 if len(years)>1 else 0)
    end_m   = st.selectbox("종료 월", list(range(1,13)), index=11)

    # 캘린더 매트릭스 표시 연도
    view_year = st.selectbox("매트릭스 표시 연도", years, index=0)

    run_btn = st.button("분석 시작", type="primary")

st.title("Effective Days — 유효일수 분석")

# 버튼을 눌러야 실행
if not run_btn:
    st.info("좌측에서 **기간을 선택**하고 **분석 시작** 버튼을 눌러주세요.")
    st.stop()

# 데이터 로드
if file is None:
    st.warning("엑셀을 업로드하거나 data/effective_days_calendar.xlsx 를 레포에 넣어주세요.")
    st.stop()

try:
    raw = pd.read_excel(file, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀을 읽는 중 문제가 발생했습니다: {e}")
    st.stop()

# 정규화
try:
    base_df, supply_col = normalize_calendar(raw)
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

# 가중치 계산(학습 데이터 전체에서 산정)
W_monthly, W_global = compute_weights_monthly(base_df, supply_col, base_cat="평일_1", cap_holiday=0.95)

# 기간 필터
start_ts = pd.Timestamp(int(start_y), int(start_m), 1)
end_ts = pd.Timestamp(int(end_y), int(end_m), 1) + pd.offsets.MonthEnd(0)
if end_ts < start_ts:
    st.error("종료가 시작보다 빠릅니다. 기간을 다시 선택하세요.")
    st.stop()
pred_df = base_df[(base_df["날짜"] >= start_ts) & (base_df["날짜"] <= end_ts)].copy()
if pred_df.empty:
    st.error("선택한 구간에 해당하는 날짜가 엑셀에 없습니다. 데이터 기간을 확인하세요.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# ① 캘린더 매트릭스 (맨 위)
st.subheader("유효일수 카테고리 매트릭스")
df_year = base_df[base_df["연"]==int(view_year)].copy()
if df_year.empty:
    st.warning(f"{view_year}년 데이터가 원본에 없습니다. 다른 연도를 선택하세요.")
else:
    fig = draw_calendar_matrix(int(view_year), df_year, W_global)
    st.pyplot(fig, clear_figure=True)

# ② 카테고리 가중치 요약 (중앙정렬 & 라벨 친절히)
st.subheader("카테고리 가중치 요약")
w_show = pd.DataFrame({
    "카테고리": [CAT_LABEL[c] for c in CATS],
    "전역 가중치(중앙값)": [round(W_global[c], 4) for c in CATS]
})
st.dataframe(w_show, hide_index=True, use_container_width=True)

st.caption("※ 가중치는 **동일 월**의 ‘평일_1(화·수·목)’ 공급량 **중앙값** 대비 "
           "각 카테고리 공급량 **중앙값의 비율**입니다. "
           "데이터가 부족하면 전역 중앙값/기본값으로 보강하며, "
           "명절·공휴일은 보수적으로 **0.95 상한**을 둡니다. "
           "예: 추석 가중치 0.48이라면, 해당 월의 평일_1 대비 추석기간 실적이 평균 **48% 수준**이었음을 뜻합니다.")

# ③ 월별 유효일수 표 (비고 포함)
st.subheader("월별 유효일수 요약")
eff_tbl = effective_days_by_month(pred_df, W_monthly)
# 보기 좋은 칼럼 순서
show_cols = (["연","월","월일수"] +
             [f"일수_{c}" for c in CATS] +
             ["유효일수합","적용_비율(유효/월일수)","비고"])
st.dataframe(eff_tbl[show_cols].sort_values(["연","월"]),
             hide_index=True, use_container_width=True)

st.caption("비고 예시: ‘설(5일: 2/14–18)’, ‘추석(4일: 9/24–27)’, ‘대체(2일: 2/9–10)’ 등. "
           "연휴가 주말과 겹치더라도 본 도구는 **명절/대체 구간 전체**를 명절 가중치로 계산합니다.")

# ④ CSV 다운로드
st.download_button(
    "월별 유효일수 결과 CSV 다운로드",
    data=eff_tbl.sort_values(["연","월"])[show_cols].to_csv(index=False).encode("utf-8-sig"),
    file_name="effective_days_by_month.csv",
    mime="text/csv"
)
