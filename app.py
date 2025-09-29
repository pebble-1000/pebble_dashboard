# app.py
# AURC 계산기 — One-Page PDF Style + Cohorts (FULL v4)
# - 한 페이지 PDF 레이아웃
# - 월/주 단위 토글, 동적 Horizon
# - 구간 컨벤션(G01~G04) 개선율: 기본값 G02=70%, G03=60%, G04=50% (G01=0%)
# - 생존/ΔS/Hazard 시각화 (한글 축)
# - 코호트: fst_months 전용(1/3/6/12 ON/OFF)
# - AURC 표, Data QA, 벤치마크 비교, 의사결정 카드(Go/검토/보류)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from typing import Tuple, List, Optional

st.set_page_config(page_title="AURC 계산기 — PDF Style + Cohorts v4", page_icon="📈", layout="wide")
sns.set_theme(style="whitegrid")

# -------------------- 폰트 --------------------
def setup_korean_font():
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",     # Linux
        "C:/Windows/Fonts/malgun.ttf",                         # Windows
        "NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"
    ]
    for path in candidates:
        try:
            if "/" in path:
                import matplotlib.font_manager as fm
                fm.fontManager.addfont(path)
                matplotlib.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            else:
                matplotlib.rcParams["font.family"] = path
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
setup_korean_font()

# -------------------- 스타일 --------------------
STYLE = """
<style>
.section-card {border:1px solid #e6e9ef; padding:1rem 1.2rem; border-radius:12px; background: #ffffff;}
.section-title {font-weight:700; margin-bottom:0.2rem;}
.section-caption {color:#6b7280; font-size:0.9rem; margin-top:0.1rem;}
hr.soft {border:none; border-top:1px solid #efeff5; margin:0.6rem 0 0.8rem 0;}
.kpi {border:1px solid #e6e9ef; padding:0.8rem 1rem; border-radius:12px; background:#fafafa;}
.kpi h3 {margin:0; font-size:0.9rem; color:#6b7280; font-weight:600;}
.kpi .val {font-size:1.4rem; font-weight:800; margin-top:0.3rem;}
.small-note {color:#6b7280; font-size:0.85rem;}

/* Decision Card */
.decision {border-radius:14px; padding:1rem 1.2rem; color:#0b0f19;}
.go {background:#ecfdf5; border:1px solid #a7f3d0;}
.review {background:#fffbeb; border:1px solid #fde68a;}
.hold {background:#fef2f2; border:1px solid #fecaca;}
.decision h3 {margin:0 0 .4rem 0; font-size:1.05rem;}
.decision .badge {display:inline-block; padding:.2rem .5rem; border-radius:9999px; font-weight:700; font-size:.85rem;}
.badge-go {background:#10b981; color:white;}
.badge-review {background:#f59e0b; color:white;}
.badge-hold {background:#ef4444; color:white;}
ul.tight {margin:.2rem 0 0 .9rem; padding:0;}
ul.tight li {margin:.12rem 0;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# -------------------- CSV 로더 --------------------
TRY_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
def read_csv_kr(file_like, **kwargs):
    last_err = None
    for enc in TRY_ENCODINGS:
        try:
            return pd.read_csv(file_like, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

# -------------------- 스키마 --------------------
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
    low = {c.lower(): c for c in df.columns}
    if "done_month" in low:
        return "raw_done_month", df.rename(columns=str.lower).copy()
    if "done_week" in low:
        return "raw_done_week", df.rename(columns=str.lower).copy()
    raise ValueError("형식 인식 실패: (month+survival/hazard/churn) 또는 raw의 done_month/done_week 필요")

# -------------------- 계산 --------------------
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

def aurc_from_survival(S: np.ndarray, horizon: Optional[int] = None) -> float:
    if horizon is not None:
        S = S[:horizon]
    return float(np.sum(S))

def build_survival_from_done(x: pd.Series, max_time: int, treat_nan_as_censored: bool=True) -> np.ndarray:
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

def median_survival_time(S: np.ndarray) -> Optional[int]:
    for i, v in enumerate(S, start=1):
        if v <= 0.5:
            return i
    return None

# -------------------- 헤더 --------------------
st.markdown("<div class='section-title' style='font-size:1.5rem;'>📈 고객 이탈 예측 · AURC 계산기</div>", unsafe_allow_html=True)
st.markdown("<div class='small-note'>PDF 스타일 · 한 페이지 · 코호트 생존 + 의사결정 카드</div>", unsafe_allow_html=True)
st.markdown("<hr class='soft'>", unsafe_allow_html=True)

# -------------------- 업로드 Row --------------------
col_u1, col_u2 = st.columns([1,1])
with col_u1:
    st.markdown("<div class='section-card'><div class='section-title'>데이터 업로드</div><div class='section-caption'>CSV를 선택해 주세요.</div>", unsafe_allow_html=True)
    up = st.file_uploader("CSV 파일", type=["csv"], key="csv_main")
    st.markdown("</div>", unsafe_allow_html=True)

with col_u2:
    st.markdown("<div class='section-card'><div class='section-title'>(선택) 벤치마크 업로드</div><div class='section-caption'>비교용 CSV.</div>", unsafe_allow_html=True)
    bm = st.file_uploader("벤치마크 CSV", type=["csv"], key="csv_bm")
    st.markdown("</div>", unsafe_allow_html=True)

if up is None:
    st.info("CSV를 업로드하면 아래 섹션들이 활성화됩니다.")
    st.stop()

# -------------------- 데이터 적재/해석 --------------------
df = read_csv_kr(up)
kind, df2 = autodetect_format(df)

unit_col, horizon_col, opts_col = st.columns([1,1,2])
with unit_col:
    st.markdown("<div class='section-card'><div class='section-title'>단위 선택</div>", unsafe_allow_html=True)
    unit = st.radio("분석 단위", ["월 단위", "주 단위"], index=0, horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

# S_full, h_full 만들기
if kind == "agg_survival_month":
    S_month = df2["survival"].astype(float).to_numpy()
    if unit == "월 단위":
        S_full = S_month
        h_full = survival_to_hazards(S_full)
    else:
        h_m = survival_to_hazards(S_month)
        h_w = 1.0 - (1.0 - h_m)**(1/4)
        h_w = np.repeat(h_w, 4)
        h_full = np.clip(h_w, 0, 1)
        S_full = hazards_to_survival(h_full)

elif kind == "agg_hazard_month":
    h_m = np.clip(df2["hazard"].astype(float).to_numpy(), 0, 1)
    if unit == "월 단위":
        h_full = h_m
        S_full = hazards_to_survival(h_full)
    else:
        h_w = 1.0 - (1.0 - h_m)**(1/4)
        h_full = np.repeat(h_w, 4)
        S_full = hazards_to_survival(h_full)

elif kind == "agg_churn_month":
    c_m = np.clip(df2["churn"].astype(float).to_numpy(), 0, 1)
    if unit == "월 단위":
        h_full = c_m
        S_full = hazards_to_survival(h_full)
    else:
        h_w = 1.0 - (1.0 - c_m)**(1/4)
        h_full = np.repeat(h_w, 4)
        S_full = hazards_to_survival(h_full)

elif kind == "raw_done_month":
    if unit == "월 단위":
        done = df2["done_month"]
    else:
        done = df2["done_month"] * 4.0
    Tguess = 240 if unit=="월 단위" else 240*4
    S_full = build_survival_from_done(done, max_time=int(Tguess), treat_nan_as_censored=True)
    h_full = survival_to_hazards(S_full)

else:  # raw_done_week
    if unit == "주 단위":
        done = df2["done_week"]
    else:
        done = np.ceil(df2["done_week"] / 4.0)
    Tguess = 240 if unit=="월 단위" else 240*4
    S_full = build_survival_from_done(pd.Series(done), max_time=int(Tguess), treat_nan_as_censored=True)
    h_full = survival_to_hazards(S_full)

T = len(S_full)

with horizon_col:
    st.markdown("<div class='section-card'><div class='section-title'>분석 기간(Horizon)</div>", unsafe_allow_html=True)
    default_h = 36 if unit=="월 단위" else 36*4
    h_default = min(int(default_h), int(T))
    horizon = st.number_input("AURC 분석 구간(상한)", min_value=6, max_value=int(T), value=int(h_default), step=1)
    st.markdown("<div class='section-caption small-note'>데이터 길이 T = {}</div>".format(T), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with opts_col:
    st.markdown("<div class='section-card'><div class='section-title'>옵션</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        treat_nan_as_censored = st.checkbox("NaN 검열 처리", value=True)
    with c2:
        show_diff = st.checkbox("ΔS(t) 표시", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 개선 입력 & KPI --------------------
S_base = S_full[:int(horizon)]
h_base = h_full[:int(horizon)]

st.markdown("<div class='section-title' style='margin-top:0.6rem;'>구간 컨벤션(G01~G04) 개선율</div>", unsafe_allow_html=True)
gcols = st.columns(4)
base_segments_month = {
    "G01 결제→매칭": (1, 1),
    "G02 매칭→첫수업": (2, 2),
    "G03 첫수업→2회차": (3, 3),
    "G04 2회차 후 1개월": (4, 5),
}
def month_to_unit(rng, unit):
    if unit == "월 단위": return rng
    a,b = rng; return ((a-1)*4+1, b*4)

# 기본 개선율(요청 반영)
default_pct_map = {
    "G01 결제→매칭": 0.0,
    "G02 매칭→첫수업": 70.0,
    "G03 첫수업→2회차": 60.0,
    "G04 2회차 후 1개월": 50.0,
}

seg_ranges, seg_improves = {}, {}
for (name, rng_m), gc in zip(base_segments_month.items(), gcols):
    a_u, b_u = month_to_unit(rng_m, unit)
    with gc:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>{name}</div>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            sa = st.number_input("시작", min_value=1, max_value=240, value=int(a_u), step=1, key=f"{name}_s")
        with s2:
            sb = st.number_input("종료", min_value=1, max_value=240, value=int(b_u), step=1, key=f"{name}_e")
        default_pct = float(default_pct_map.get(name, 0.0))
        pct = st.number_input("개선율(%)", min_value=0.0, max_value=100.0, value=default_pct, step=0.5, key=f"{name}_p")
        seg_ranges[name] = (int(sa), int(sb))
        seg_improves[name] = pct
        st.markdown("</div>", unsafe_allow_html=True)

# 개선 적용
def apply_improvement(h: np.ndarray, seg_ranges: List[Tuple[int,int]], seg_improves: List[float]) -> np.ndarray:
    h2 = h.copy().astype(float)
    for (a,b), pct in zip(seg_ranges, seg_improves):
        g = max(0.0, min(1.0, pct/100.0))
        for t in range(a, b+1):
            idx = t-1
            if 0 <= idx < len(h2):
                h2[idx] *= (1.0 - g)
    return np.clip(h2, 0.0, 1.0)

seg_list = list(seg_ranges.items())
seg_idx = [v for _, v in seg_list]
seg_pct = [seg_improves[k] for k, _ in seg_list]
h_scn = apply_improvement(h_base, seg_idx, seg_pct)
S_scn = hazards_to_survival(h_scn)

base_aurc = aurc_from_survival(S_base)
new_aurc  = aurc_from_survival(S_scn)
delta     = new_aurc - base_aurc
rel       = (delta/base_aurc*100.0) if base_aurc>0 else np.nan

k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown("<div class='kpi'><h3>AURC(현재)</h3><div class='val'>{:.2f}</div></div>".format(base_aurc), unsafe_allow_html=True)
with k2:
    if unit=="주 단위":
        st.markdown("<div class='kpi'><h3>AURC(개월 환산)</h3><div class='val'>{:.2f}</div></div>".format(base_aurc/4.0), unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi'><h3>Horizon</h3><div class='val'>{}</div></div>".format(int(horizon)), unsafe_allow_html=True)
with k3:
    st.markdown("<div class='kpi'><h3>ΔAURC</h3><div class='val'>{:+.2f}</div></div>".format(delta), unsafe_allow_html=True)
with k4:
    st.markdown("<div class='kpi'><h3>단위</h3><div class='val'>{}</div></div>".format("월" if unit=="월 단위" else "주"), unsafe_allow_html=True)

# -------------------- 의사결정 카드 --------------------
st.markdown("<hr class='soft'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>의사결정 카드</div>", unsafe_allow_html=True)

cA, cB, cC, cD = st.columns([1,1,1,1.4])
with cA:
    thr_go = st.number_input("Go 임계값 (ΔAURC ≥)", min_value=0.0, max_value=1e6, value=30.0, step=1.0)
with cB:
    thr_review = st.number_input("검토 임계값 (ΔAURC ≥)", min_value=0.0, max_value=1e6, value=10.0, step=1.0)
with cC:
    min_rel = st.number_input("최소 상대개선(%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
with cD:
    effort = st.number_input("예상 공수(인일)", min_value=0.0, max_value=1e6, value=5.0, step=0.5)

impact_per_pd = (delta / effort) if effort>0 else np.nan
meets_rel = (rel >= min_rel) if not np.isnan(rel) else False

def decide(delta, rel, thr_go, thr_review, min_rel):
    if np.isnan(rel):  # base_aurc=0 등
        rel_ok = True  # 상대조건 무시
    else:
        rel_ok = rel >= min_rel
    if (delta >= thr_go) and rel_ok:
        return "Go", "badge-go", "go"
    if (delta >= thr_review) and rel_ok:
        return "검토", "badge-review", "review"
    return "보류", "badge-hold", "hold"

label, badge, cls = decide(delta, rel, thr_go, thr_review, min_rel)

card_html = f"""
<div class="decision {cls}">
  <h3>권고안: <span class="badge {badge}">{label}</span></h3>
  <ul class="tight">
    <li>실측 ΔAURC: <b>{delta:+.2f}</b> (기준: Go ≥ {thr_go:.2f}, 검토 ≥ {thr_review:.2f})</li>
    <li>상대 개선율: <b>{(0 if np.isnan(rel) else rel):.2f}%</b> (최소 요구치 ≥ {min_rel:.2f}%)</li>
    <li>예상 공수: <b>{effort:.1f} 인일</b> → Impact/인일: <b>{(0 if np.isnan(impact_per_pd) else impact_per_pd):.2f}</b></li>
  </ul>
  <div class="small-note">※ 임계값은 조직 컨벤션에 맞게 조정하세요. (예: ΔAURC 30=Go, 10~30=검토, &lt;10=보류)</div>
</div>
"""
st.markdown(card_html, unsafe_allow_html=True)

st.markdown("<hr class='soft'>", unsafe_allow_html=True)

# -------------------- 그래프 Row --------------------
t_axis = np.arange(1, len(S_base)+1)
gc1, gc2 = st.columns([2,1])
with gc1:
    st.markdown("<div class='section-card'><div class='section-title'>생존곡선 S(t)</div>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(9,4))
    sns.lineplot(x=t_axis, y=S_base, marker="o", ax=ax1, label="현재", color="#111827")
    sns.lineplot(x=t_axis, y=S_scn[:len(S_base)], marker="x", ax=ax1, label="개선 후", color="#2563eb")
    ax1.set_xlabel("월" if unit=="월 단위" else "주")
    ax1.set_ylabel("생존확률 S(t)")
    for name, (a,b) in seg_ranges.items():
        ax1.axvspan(a-0.5, b+0.5, alpha=0.08)
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

with gc2:
    if len(S_base)>0 and show_diff:
        st.markdown("<div class='section-card'><div class='section-title'>차이 곡선 ΔS(t)</div>", unsafe_allow_html=True)
        diff = (S_scn[:len(S_base)] - S_base)
        figd, axd = plt.subplots(figsize=(6,4))
        axd.plot(np.arange(1, len(diff)+1), diff, linewidth=1.8, color="#ef4444")
        axd.axhline(0, linestyle="--", linewidth=1, color="#9ca3af")
        axd.set_xlabel("월" if unit=="월 단위" else "주")
        axd.set_ylabel("차이 ΔS(t)")
        st.pyplot(figd)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Hazard (막대 전용) --------------------
st.markdown("<div class='section-card'><div class='section-title'>이탈위험 Hazard(t)</div>", unsafe_allow_html=True)
if len(S_base) > 0:
    tmp = pd.DataFrame({"t": t_axis, "현재": h_base[:len(t_axis)], "개선 후": h_scn[:len(t_axis)]})
    tmpm = tmp.melt(id_vars="t", value_vars=["현재","개선 후"], var_name="series", value_name="hazard")
    fig2, ax2 = plt.subplots(figsize=(9,4))
    sns.barplot(data=tmpm, x="t", y="hazard", hue="series", ax=ax2, palette=["#94a3b8","#3b82f6"])
    ax2.set_xlabel("월" if unit=="월 단위" else "주")
    ax2.set_ylabel("이탈위험 Hazard(t)")
    st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 구매 개월수별 생존 (fst_months 전용) --------------------
st.markdown("<div class='section-card'><div class='section-title'>구매 개월수별 생존 (fst_months)</div><div class='section-caption'>1/3/6/12개월 코호트를 fst_months로 구분, 개별 ON/OFF.</div>", unsafe_allow_html=True)

if "fst_months" not in df2.columns:
    st.warning("'fst_months' 컬럼을 찾지 못했습니다. CSV에 'fst_months'를 포함해 주세요.")
else:
    COHORT_LEVELS = [1,3,6,12]
    cbox_cols = st.columns(4)
    chosen = []
    for m, cbc in zip(COHORT_LEVELS, cbox_cols):
        with cbc:
            if st.checkbox(f"{m}개월 표시", value=True, key=f"cohort_{m}"):
                chosen.append(m)

    if len(chosen) == 0:
        st.info("표시할 코호트를 하나 이상 선택해 주세요.")
    else:
        palette = {1:"#22c55e", 3:"#14b8a6", 6:"#a855f7", 12:"#ef4444", "전체":"#111827"}
        figc, axc = plt.subplots(figsize=(9,4))
        axc.plot(t_axis, S_base, label="전체", linewidth=2.0, color=palette["전체"])

        cohort_rows = []
        for m in chosen:
            sub = df2[df2["fst_months"]==m]
            if sub.empty:
                continue
            if kind == "raw_done_week":
                done_vec = sub["done_week"] if unit=="주 단위" else np.ceil(sub["done_week"]/4.0)
            elif kind == "raw_done_month":
                done_vec = sub["done_month"] if unit=="월 단위" else sub["done_month"]*4.0
            else:
                continue
            S_c = build_survival_from_done(done_vec, max_time=int(horizon), treat_nan_as_censored=True)
            axc.plot(np.arange(1, len(S_c)+1), S_c, label=f"{m}개월", linewidth=1.8, color=palette.get(m, None))
            n_total = len(done_vec)
            n_end = int(np.sum(~pd.isna(done_vec)))
            churn_rate = n_end / n_total if n_total>0 else np.nan
            A = aurc_from_survival(S_c, horizon=int(horizon))
            med = median_survival_time(S_c)
            cohort_rows.append([f"{m}개월", n_total, churn_rate*100 if pd.notna(churn_rate) else np.nan, A, med])

        axc.set_xlabel("월" if unit=="월 단위" else "주")
        axc.set_ylabel("생존확률 S(t)")
        axc.legend()
        st.pyplot(figc)

        if cohort_rows:
            dfco = pd.DataFrame(cohort_rows, columns=["코호트","샘플 수","중단율(%)","AURC","중위 생존(단위)"])
            st.dataframe(dfco, use_container_width=True)
        else:
            st.info("선택한 코호트를 찾지 못했습니다.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 결과 표(전체) --------------------
def results_df(unit: str, base_aurc: float, new_aurc: float, horizon: int):
    rows = []
    if unit == "주 단위":
        rows.append(["현재", base_aurc, base_aurc/4.0, horizon, "주"])
        rows.append(["개선 후", new_aurc, new_aurc/4.0, horizon, "주"])
    else:
        rows.append(["현재", base_aurc, base_aurc, horizon, "개월"])
        rows.append(["개선 후", new_aurc, new_aurc, horizon, "개월"])
    dfres = pd.DataFrame(rows, columns=["구분", "AURC(원단위)", "AURC(개월 환산)", "Horizon", "단위"])
    dfres["ΔAURC"] = [np.nan, new_aurc - base_aurc]
    dfres["개선율(%)"] = [np.nan, (new_aurc-base_aurc)/base_aurc*100.0 if base_aurc>0 else np.nan]
    return dfres

st.markdown("<div class='section-card'><div class='section-title'>AURC 분석 결과 (표)</div>", unsafe_allow_html=True)
st.dataframe(results_df(unit, base_aurc, new_aurc, int(horizon)), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- QA --------------------
st.markdown("<div class='section-card'><div class='section-title'>Data QA</div>", unsafe_allow_html=True)
if kind.startswith("raw_done"):
    x = df2["done_week"] if kind=="raw_done_week" else df2["done_month"]
    if unit == "주 단위" and kind=="raw_done_month":
        x = x * 4.0
    if unit == "월 단위" and kind=="raw_done_week":
        x = np.ceil(x / 4.0)
    total_n = len(x)
    ended = int(np.sum(~pd.isna(x)))
    active = int(total_n - ended)
    qc1,qc2,qc3 = st.columns(3)
    qc1.metric("총 샘플 수", total_n)
    qc2.metric("종료 수", ended)
    qc3.metric("검열 수", active)

    figd2, axd2 = plt.subplots(figsize=(9,3))
    bins = int(max(10, min(60, len(S_full))))
    sns.histplot(x, bins=bins, ax=axd2, color="#64748b")
    axd2.set_xlabel("월" if unit=="월 단위" else "주")
    st.pyplot(figd2)

    st.caption("※ G02 이탈률은 done_* 기반 근사치이며, 검열 보정이 완전하지 않을 수 있습니다.")
else:
    st.caption("집계형 데이터이므로 QA 상세(분모/분자)는 제한됩니다.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 벤치마크 비교 --------------------
if bm is not None:
    st.markdown("<div class='section-card'><div class='section-title'>벤치마크 비교</div>", unsafe_allow_html=True)
    dfb = read_csv_kr(bm)
    kind_b, dfb2 = autodetect_format(dfb)

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

    if kind_b == "agg_survival_month":
        Sb_m = dfb2["survival"].astype(float).to_numpy()
        if unit == "월 단위":
            Sb = Sb_m
        else:
            hb_m = survival_to_hazards(Sb_m)
            hb_w = 1.0 - (1.0 - hb_m)**(1/4)
            hb_w = np.repeat(hb_w, 4)
            Sb = hazards_to_survival(hb_w)
    elif kind_b == "agg_hazard_month":
        Hb_m = np.clip(dfb2["hazard"].astype(float).to_numpy(), 0, 1)
        if unit == "월 단위":
            Sb = hazards_to_survival(Hb_m)
        else:
            Hb_w = 1.0 - (1.0 - Hb_m)**(1/4)
            Hb_w = np.repeat(Hb_w, 4)
            Sb = hazards_to_survival(Hb_w)
    elif kind_b == "agg_churn_month":
        Cb_m = np.clip(dfb2["churn"].astype(float).to_numpy(), 0, 1)
        if unit == "월 단위":
            Sb = hazards_to_survival(Cb_m)
        else:
            Hb_w = 1.0 - (1.0 - Cb_m)**(1/4)
            Hb_w = np.repeat(Hb_w, 4)
            Sb = hazards_to_survival(Hb_w)
    elif kind_b == "raw_done_month":
        Db = dfb2["done_month"] if unit=="월 단위" else dfb2["done_month"]*4.0
        Sb = build_survival_from_done(Db, max_time=int(len(S_full)), treat_nan_as_censored=True)
    else:  # raw_done_week
        Db = dfb2["done_week"] if unit=="주 단위" else np.ceil(dfb2["done_week"]/4.0)
        Sb = build_survival_from_done(pd.Series(Db), max_time=int(len(S_full)), treat_nan_as_censored=True)

    Sb = Sb[:len(S_base)]
    auc_b = aurc_from_survival(Sb)
    cc1,cc2 = st.columns(2)
    if unit == "주 단위":
        cc1.metric("AURC(현재, 주)", f"{aurc_from_survival(S_base):.2f}")
        cc2.metric("AURC(벤치마크, 주)", f"{auc_b:.2f}")
        st.caption(f"개월 환산: 현재 {aurc_from_survival(S_base)/4:.2f} vs 벤치마크 {auc_b/4:.2f}")
    else:
        cc1.metric("AURC(현재, 개월)", f"{aurc_from_survival(S_base):.2f}")
        cc2.metric("AURC(벤치마크, 개월)", f"{auc_b:.2f}")

    figb, axb = plt.subplots(figsize=(9,4))
    sns.lineplot(x=np.arange(1, len(S_base)+1), y=S_base, ax=axb, label="현재", color="#111827")
    sns.lineplot(x=np.arange(1, len(Sb)+1), y=Sb, ax=axb, label="벤치마크", color="#0ea5e9")
    axb.set_xlabel("월" if unit=="월 단위" else "주")
    axb.set_ylabel("생존확률 S(t)")
    st.pyplot(figb)
    st.markdown("</div>", unsafe_allow_html=True)
