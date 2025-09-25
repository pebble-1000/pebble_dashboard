# app.py
# AURC 계산기 (월 ↔ 주 단위 전환 지원)
# - CSV 안전 로더(한글 인코딩)
# - 단위 토글: 월/주
# - 생존곡선 S(t) 구성, AURC 계산
# - 회사 표준 구간(G01~G04) 개선율 적용(단위 자동 매핑)
# - Data QA: 분포/센서링, G02 구간 누적 이탈률 + 95% CI
# - 벤치마크 비교(선택)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from typing import Tuple, List, Optional

st.set_page_config(page_title="AURC 계산기 (월↔주)", page_icon="📈", layout="wide")
sns.set_theme(style="whitegrid")

# -------------------- 폰트(가능 시) --------------------
def setup_korean_font():
    # 자주 쓰이는 한글 폰트 후보
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",    # Linux
        "C:/Windows/Fonts/malgun.ttf",                        # Windows
    ]
    found = False
    for path in candidates:
        try:
            fm.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            found = True
            break
        except Exception:
            continue
    if not found:
        # fallback: 영어만 나와도 깨지지 않게
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_korean_font()

# -------------------- CSV 안전 로더 --------------------
TRY_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
def read_csv_kr(file_like, **kwargs):
    last_err = None
    for enc in TRY_ENCODINGS:
        try:
            return pd.read_csv(file_like, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

# -------------------- 스키마 감지 --------------------
def autodetect_format(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    cols = {c.lower() for c in df.columns}
    if {"month", "survival"}.issubset(cols):
        out = df.rename(columns=str.lower)[["month", "survival"]].copy()
        out["month"] = out["month"].astype(int)
        return "agg_survival_month", out.sort_values("month")
    if {"month", "hazard"}.issubset(cols):
        out = df.rename(columns=str.lower)[["month", "hazard"]].copy()
        out["month"] = out["month"].astype(int)
        return "agg_hazard_month", out.sort_values("month")
    if {"month", "churn"}.issubset(cols):
        out = df.rename(columns=str.lower)[["month", "churn"]].copy()
        out["month"] = out["month"].astype(int)
        return "agg_churn_month", out.sort_values("month")
    # raw (개인 단위)
    low = {c.lower(): c for c in df.columns}
    if "done_month" in low:
        return "raw_done_month", df.rename(columns=str.lower).copy()
    if "done_week" in low:
        return "raw_done_week", df.rename(columns=str.lower).copy()
    raise ValueError("형식 인식 실패: (month+survival/hazard/churn) 또는 raw의 done_month/done_week 필요")

# -------------------- 변환/계산 --------------------
def hazards_to_survival(h: np.ndarray) -> np.ndarray:
    S, surv = [], 1.0
    for hm in h:
        surv *= max(0.0, 1.0 - float(hm))
        S.append(surv)
    return np.array(S, dtype=float)

def survival_to_hazards(S: np.ndarray) -> np.ndarray:
    h, prev = [], 1.0
    for s in S:
        s = float(s)
        h.append(0.0 if prev <= 0 else max(0.0, 1.0 - s/prev))
        prev = s
    return np.array(h, dtype=float)

def churn_to_hazards(churn: np.ndarray) -> np.ndarray:
    return np.clip(np.array(churn, dtype=float), 0.0, 1.0)

def aurc_from_survival(S: np.ndarray, horizon: Optional[int] = None) -> float:
    if horizon is not None:
        S = S[:horizon]
    return float(np.sum(S))

def build_survival_from_done(x: pd.Series, max_time: int, treat_nan_as_censored: bool=True) -> np.ndarray:
    """done_time(정수형 기간)로부터 이산형 생존곡선 S(t) 구성"""
    arr = x.to_numpy()
    N = len(arr)
    S = []
    for t in range(1, max_time+1):
        if treat_nan_as_censored:
            survived = np.sum((arr >= t) | pd.isna(arr))
        else:
            survived = np.sum(arr >= t)
        S.append(survived / max(N,1))
    return np.array(S, dtype=float)

def apply_improvement(h: np.ndarray, seg_ranges: List[Tuple[int,int]], seg_improves: List[float]) -> np.ndarray:
    h2 = h.copy().astype(float)
    for (a,b), pct in zip(seg_ranges, seg_improves):
        g = max(0.0, min(1.0, pct/100.0))
        for t in range(a, b+1):
            idx = t-1
            if 0 <= idx < len(h2):
                h2[idx] *= (1.0 - g)
    return np.clip(h2, 0.0, 1.0)

def rate_with_ci(d, n, alpha=0.05):
    if n <= 0 or d < 0 or d > n:
        return np.nan, (np.nan, np.nan)
    p = d / n
    lo, hi = proportion_confint(d, n, alpha=alpha, method="wilson")
    return p, (lo, hi)

def confidence_badge_by_width(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return "Low"
    width = hi - lo
    if width < 0.03: return "High"   # <3pp
    if width < 0.05: return "Med"    # 3~5pp
    return "Low"

# -------------------- 사이드바 --------------------
st.title("📈 AURC 계산기 — 월↔주 단위 전환")
with st.sidebar:
    st.markdown("### 1) 데이터 업로드")
    up = st.file_uploader("CSV 파일", type=["csv"])

    st.markdown("### 2) 단위 선택")
    unit = st.radio("분석 단위", ["월 단위", "주 단위"], index=0)

    st.markdown("### 3) 기간 설정")
    default_h = 24 if unit=="월 단위" else 24*4  # 24개월/96주 기본값
    max_time = st.number_input("최대 기간(단위에 따름)", min_value=8, max_value=240, value=default_h, step=1)
    h_default = min(int(default_h), int(max_time))
    horizon = st.number_input("AURC 분석 구간(상한)", min_value=6, max_value=int(max_time), value=h_default, step=1)

    st.markdown("### 4) 옵션")
    treat_nan_as_censored = st.checkbox("NaN(진행중)을 검열(생존) 처리", value=True)

    st.markdown("### 5) 구간 컨벤션(G01~G04) 개선율")
    st.caption("회사 표준 구간 (월 기준)을 자동으로 선택한 단위로 변환합니다.")
    base_segments_month = {
        "G01 결제→매칭": (1, 1),
        "G02 매칭→첫수업": (1, 2),
        "G03 첫수업→2회차": (2, 3),
        "G04 2회차 후 1개월": (4, 5),
    }

    def month_range_to_unit(rng: Tuple[int,int], unit: str) -> Tuple[int,int]:
        if unit == "월 단위":
            return rng
        # 주 단위 변환: 1개월≈4주 가정 → [m_start*4-3, m_end*4]
        a, b = rng
        return ((a-1)*4+1, b*4)

    seg_ranges = {}
    seg_improves = {}
    for name, rng_m in base_segments_month.items():
        a_u, b_u = month_range_to_unit(rng_m, unit)
        c1, c2, c3 = st.columns([1,1,2])
        with c1: sa = st.number_input(f"{name} 시작", 1, 240, a_u, key=f"{name}_s")
        with c2: sb = st.number_input(f"{name} 종료", 1, 240, b_u, key=f"{name}_e")
        seg_ranges[name] = (int(sa), int(sb))
        with c3: seg_improves[name] = st.slider(f"{name} 개선율(%)", 0.0, 50.0, 0.0 if "G02" not in name else 3.0, 0.5, key=f"{name}_p")

    st.markdown("### 6) 벤치마크(선택)")
    bm = st.file_uploader("벤치마크 CSV", type=["csv"])

if up is None:
    st.info("좌측에서 CSV를 업로드하면 분석이 시작됩니다.")
    st.stop()

# -------------------- 데이터 적재/해석 --------------------
df = read_csv_kr(up)
kind, df2 = autodetect_format(df)

# 기준 생존곡선/해저드(선택 단위로 변환)
if kind == "agg_survival_month":
    S_month = df2["survival"].astype(float).to_numpy()
    if unit == "월 단위":
        S_base = S_month[:max_time]
        h_base = survival_to_hazards(S_base)
    else:
        S_week = np.repeat(S_month, 4)  # 월→주 계단 보간
        S_base = S_week[:max_time]
        h_base = survival_to_hazards(S_base)

elif kind == "agg_hazard_month":
    h_month = np.clip(df2["hazard"].astype(float).to_numpy(), 0, 1)
    if unit == "월 단위":
        h_base = h_month[:max_time]
        S_base = hazards_to_survival(h_base)
    else:
        h_week = np.repeat(h_month, 4)  # 월→주 반복
        h_base = h_week[:max_time]
        S_base = hazards_to_survival(h_base)

elif kind == "agg_churn_month":
    c_month = np.clip(df2["churn"].astype(float).to_numpy(), 0, 1)
    if unit == "월 단위":
        h_base = c_month[:max_time]
        S_base = hazards_to_survival(h_base)
    else:
        h_week = np.repeat(c_month, 4)
        h_base = h_week[:max_time]
        S_base = hazards_to_survival(h_base)

elif kind == "raw_done_month":
    if unit == "월 단위":
        done = df2["done_month"]
        S_base = build_survival_from_done(done, max_time=int(max_time), treat_nan_as_censored=treat_nan_as_censored)
        h_base = survival_to_hazards(S_base)
    else:
        done = df2["done_month"] * 4.0
        S_base = build_survival_from_done(done, max_time=int(max_time), treat_nan_as_censored=treat_nan_as_censored)
        h_base = survival_to_hazards(S_base)

else:  # raw_done_week
    if unit == "주 단위":
        done = df2["done_week"]
        S_base = build_survival_from_done(done, max_time=int(max_time), treat_nan_as_censored=treat_nan_as_censored)
        h_base = survival_to_hazards(S_base)
    else:
        done = np.ceil(df2["done_week"] / 4.0)
        S_base = build_survival_from_done(pd.Series(done), max_time=int(max_time), treat_nan_as_censored=treat_nan_as_censored)
        h_base = survival_to_hazards(S_base)

# -------------------- 기준 지표 & 시나리오 --------------------
base_aurc = aurc_from_survival(S_base, horizon=int(horizon))

# 개선 시나리오 적용
seg_list = list(seg_ranges.items())
seg_idx = [v for _, v in seg_list]
seg_pct = [seg_improves[k] for k, _ in seg_list]
h_scn = apply_improvement(h_base, seg_idx, seg_pct)
S_scn = hazards_to_survival(h_scn)
new_aurc = aurc_from_survival(S_scn, horizon=int(horizon))
delta = new_aurc - base_aurc
rel = (delta/base_aurc*100.0) if base_aurc>0 else 0.0

# -------------------- 상단 KPI --------------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("AURC(기대)", f"{base_aurc:.2f}")
c2.metric("개선 후 AURC", f"{new_aurc:.2f}")
c3.metric("ΔAURC", f"{delta:+.2f}")
c4.metric("단위", "월" if unit=="월 단위" else "주")

# -------------------- 생존/해저드 시각화 --------------------
st.markdown("### 생존곡선 S(t)")
t_axis = np.arange(1, len(S_base)+1)
fig1, ax1 = plt.subplots(figsize=(9,4))
sns.lineplot(x=t_axis, y=S_base, marker="o", ax=ax1, label="현재")
sns.lineplot(x=t_axis, y=S_scn[:len(S_base)], marker="x", ax=ax1, label="개선 후")
ax1.set_xlabel("Month" if unit=="월 단위" else "Week")
ax1.set_ylabel("Survival S(t)")
st.pyplot(fig1)

st.markdown("### 이탈위험 Hazard(t)")
tmp = pd.DataFrame({"t": t_axis, "hazard_base": h_base[:len(t_axis)], "hazard_scn": h_scn[:len(t_axis)]})
tmpm = tmp.melt(id_vars="t", value_vars=["hazard_base","hazard_scn"], var_name="series", value_name="hazard")
tmpm["series"] = tmpm["series"].replace({
    "hazard_base": "현재",
    "hazard_scn": "개선 후"
})
fig2, ax2 = plt.subplots(figsize=(9,4))
sns.barplot(data=tmpm, x="t", y="hazard", hue="series", ax=ax2)
ax2.set_xlabel("Month" if unit=="월 단위" else "Week")
ax2.set_ylabel("Hazard")
st.pyplot(fig2)

# -------------------- Data QA --------------------
st.markdown("### Data QA")
if kind.startswith("raw_done"):
    x = df2["done_week"] if kind=="raw_done_week" else df2["done_month"]
    if unit == "주 단위" and kind=="raw_done_month":
        x = x * 4.0
    if unit == "월 단위" and kind=="raw_done_week":
        x = np.ceil(x / 4.0)
    total_n = len(x)
    ended = int(np.sum(~pd.isna(x)))
    active = int(total_n - ended)
    c1,c2,c3 = st.columns(3)
    c1.metric("총 샘플 수", total_n)
    c2.metric("종료 수", ended)
    c3.metric("검열 수", active)

    # 분포
    figd, axd = plt.subplots(figsize=(9,3))
    bins = int(max(10, min(60, max_time)))
    sns.histplot(x, bins=bins, ax=axd)
    axd.set_xlabel("Month" if unit=="월 단위" else "Week")
    st.pyplot(figd)

    # G02 구간 누적 이탈률 근사 + CI
    g02_a, g02_b = seg_ranges["G02 매칭→첫수업"]
    s = slice(g02_a-1, g02_b)
    g02_haz = h_base[s]
    g02_churn = 1 - np.prod(1 - g02_haz)  # 누적 이탈확률 근사

    arr = x.to_numpy() if isinstance(x, pd.Series) else np.array(x)
    n = int(np.sum((arr >= g02_a) | pd.isna(arr)))
    d = int(np.sum((arr >= g02_a) & (arr < g02_b)))
    p, (lo, hi) = rate_with_ci(d, n, alpha=0.05)
    conf = confidence_badge_by_width(lo, hi)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("G02 누적 이탈(근사)", f"{g02_churn*100:.1f}%")
    c2.metric("표본 n", f"{n}")
    c3.metric("이탈 d", f"{d}")
    c4.metric("95% CI / 신뢰도", f"{(lo*100 if not np.isnan(lo) else float('nan')):.1f}%~{(hi*100 if not np.isnan(hi) else float('nan')):.1f}% / {conf}")
else:
    st.caption("raw 데이터가 아니므로 QA 상세(분모/분자)는 제한됩니다.")

# -------------------- 벤치마크 비교(선택) --------------------
st.markdown("### 벤치마크 비교 (선택)")
if bm is not None:
    dfb = read_csv_kr(bm)
    kind_b, dfb2 = autodetect_format(dfb)
    if kind_b == "agg_survival_month":
        Sb_m = dfb2["survival"].astype(float).to_numpy()
        Sb = Sb_m if unit=="월 단위" else np.repeat(Sb_m, 4)
    elif kind_b == "agg_hazard_month":
        Hb_m = np.clip(dfb2["hazard"].astype(float).to_numpy(), 0, 1)
        Hb = Hb_m if unit=="월 단위" else np.repeat(Hb_m, 4)
        Sb = hazards_to_survival(Hb)
    elif kind_b == "agg_churn_month":
        Cb_m = np.clip(dfb2["churn"].astype(float).to_numpy(), 0, 1)
        Hb = Cb_m if unit=="월 단위" else np.repeat(Cb_m, 4)
        Sb = hazards_to_survival(Hb)
    elif kind_b == "raw_done_month":
        Db = dfb2["done_month"] if unit=="월 단위" else dfb2["done_month"]*4.0
        Sb = build_survival_from_done(Db, max_time=int(max_time), treat_nan_as_censored=treat_nan_as_censored)
    else:  # raw_done_week
        Db = dfb2["done_week"] if unit=="주 단위" else np.ceil(dfb2["done_week"]/4.0)
        Sb = build_survival_from_done(pd.Series(Db), max_time=int(max_time), treat_nan_as_censored=treat_nan_as_censored)

    Sb = Sb[:len(S_base)]
    auc_b = aurc_from_survival(Sb, horizon=int(horizon))
    c1,c2 = st.columns(2)
    c1.metric("AURC(현재)", f"{base_aurc:.2f}")
    c2.metric("AURC(벤치마크)", f"{auc_b:.2f}")

    figb, axb = plt.subplots(figsize=(9,4))
    sns.lineplot(x=np.arange(1, len(S_base)+1), y=S_base, ax=axb, label="현재")
    sns.lineplot(x=np.arange(1, len(Sb)+1), y=Sb, ax=axb, label="벤치마크")
    axb.set_xlabel("Month" if unit=="월 단위" else "Week")
    axb.set_ylabel("S(t)")
    st.pyplot(figb)
else:
    st.caption("벤치마크 CSV를 올리면 비교 차트가 표시됩니다.")
