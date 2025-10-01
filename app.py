
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as _fm
import calendar as _cal
import re

# ==============================
# Page & Fonts
# ==============================
st.set_page_config(page_title="수업 잔존기간 분석 (월/주 격자 + 호환모드)", layout="centered")

_kor_candidates = ["AppleGothic","Malgun Gothic","NanumGothic","NanumBarunGothic","Noto Sans CJK KR","Noto Sans KR","Pretendard"]
_avail = set(f.name for f in _fm.fontManager.ttflist)
for _nm in _kor_candidates:
    if _nm in _avail:
        matplotlib.rcParams["font.family"] = _nm
        break
matplotlib.rcParams["axes.unicode_minus"] = False

st.markdown(r"""
<style>
.topbar{position:sticky; top:0; z-index:5; background:#fff; padding:12px 6px; border-bottom:1px solid #eee;}
.title{font-size:1.25rem; font-weight:800;}
.subtitle{font-size:.95rem; color:#6b7280;}
.card{border:1px solid #e5e7eb; border-radius:14px; padding:10px 12px; background:#fff;}
.label{color:#6b7280; font-size:.85rem;}
.value{font-size:1.05rem; font-weight:700;}
hr{margin: 10px 0;}
</style>
<div class="topbar">
  <div class="title">수업 잔존기간 통합 분석 도구</div>
  <div class="subtitle">업로드 → <b>날짜 필터</b> → 생존분석(KM) → 코호트/이탈률 → 구간 개선 → 개선 효과</div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Utils
# ==============================
STOP = {"finish","auto_finish","done","nocard","nopay"}

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def km_bins_timegrid(durations_month, events, H, unit="month"):
    """
    KM survival sampled at integer bins 1..H for the chosen unit.
    - durations_month: durations in "done_month" units (1 month = 4 weeks by data definition)
    - unit: "month" or "week"
    """
    if unit not in {"month","week"}:
        raise ValueError("unit must be 'month' or 'week'")
    d_m = np.asarray(durations_month, float)
    if unit == "month":
        d = d_m
    else:  # week grid: 1 month == 4 weeks (data says: done_month is 4-week based)
        d = d_m * 4.0

    e = np.asarray(events, bool)
    H = int(H)

    uniq_evt = np.unique(d[e])
    S = 1.0
    step = {}
    for t in uniq_evt:
        n = np.sum(d >= t)        # at risk just before t
        di = np.sum((d == t) & e) # events at t
        step[t] = 1.0 if n <= 0 else max(0.0, min(1.0, 1.0 - di / n))

    times = np.array(sorted(step.keys()))
    S_bins = np.ones(H, float)
    idx = 0
    for m in range(1, H+1):
        while idx < len(times) and times[idx] <= m:
            S *= step[times[idx]]
            idx += 1
        S_bins[m-1] = S
    return S_bins

def aurc_sum_left(S, H):
    """AURC (표준 이산합): sum of S over 0..H-1 with S(0)=1."""
    H = min(int(H), len(S))
    auc, prev = 0.0, 1.0
    for i in range(H):
        auc += prev
        prev = S[i]
    return float(auc)

def aurc_half_trapezoid(S, H):
    """AURC (호환): 0.5개월 격자 + 사다리꼴 적분.
    S is post-step monthly values: S[0]=month1 ... S[H-1]=monthH.
    Build y at 0,0.5,1.0,...,H with step-constant segments.
    """
    H = int(H)
    if H <= 0:
        return 0.0
    y = np.empty(2*H + 1, float)
    y[0] = 1.0
    y[1] = 1.0
    for m in range(1, H):
        y[2*m]   = float(S[m-1])
        y[2*m+1] = float(S[m-1])
    y[2*H] = float(S[H-1])
    return float(np.trapz(y, dx=0.5))

def hazards_to_survival(h):
    S=[]; s=1.0
    for v in h:
        s *= (1.0 - float(v))
        S.append(s)
    return np.array(S, float)

def survival_to_hazards(S):
    h=[]; prev=1.0
    for cur in S:
        cur = float(cur)
        val = 0.0 if prev <= 0 else (1.0 - cur/prev)
        h.append(max(0.0, min(1.0, val)))
        prev = cur
    return np.array(h, float)

def median_survival_index(S):
    for i,v in enumerate(S, start=1):
        if v <= 0.5: return i
    return float("inf")

def segment_dropout_rate(S, a, b):
    a = max(1, int(a)); b = max(a, int(b))
    S_start = 1.0 if a == 1 else float(S[a-2])
    S_end   = float(S[b-1])
    return max(0.0, S_start - S_end)

def _find_date_candidates(cols):
    out = []
    keys = ["date","datetime","at","time","crda","pay","lst_","reactive"]
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keys):
            out.append(c)
    return out

# ==============================
# 1) 업로드
# ==============================
st.header("1) 데이터 업로드")
up = st.file_uploader("CSV 업로드 (필수: done_month, tutoring_state; 코호트: fst_months 또는 fst_fst_months)", type=["csv"])
if up is None:
    st.info("CSV 업로드 시 아래 단계가 활성화됩니다.")
    st.stop()

# try encodings
df = None
for enc in ["utf-8","cp949","euc-kr"]:
    try:
        df = pd.read_csv(up, encoding=enc)
        break
    except Exception:
        pass
if df is None:
    st.error("CSV 인코딩을 인식하지 못했습니다. UTF-8로 저장 후 다시 업로드해주세요.")
    st.stop()

orig_cols = df.columns.tolist()
lowmap = {c.lower(): c for c in orig_cols}
def col(name): return lowmap.get(name.lower())

need = ["done_month","tutoring_state"]
miss = [n for n in need if col(n) is None]
if miss:
    st.error(f"필수 컬럼 누락: {miss}")
    st.stop()

# ==============================
# 2) 날짜 필터
# ==============================
st.header("2) 날짜 필터 (선택/직접 입력)")
date_candidates = _find_date_candidates(orig_cols)
if len(date_candidates) == 0:
    st.caption("날짜 후보 컬럼을 찾지 못했습니다. 이 섹션을 건너뜁니다.")
else:
    date_col = st.selectbox("필터할 날짜 컬럼", date_candidates, index=0)
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["_filter_dt"] = dt
    if df["_filter_dt"].notna().any():
        min_dt = pd.to_datetime(df["_filter_dt"].min()).date()
        max_dt = pd.to_datetime(df["_filter_dt"].max()).date()

        mode = st.radio("입력 방식", ["선택박스", "직접 입력(YYYY-MM-DD)"], horizontal=True, index=0)

        def _parse_date_str(s: str):
            s = (s or "").strip()
            if not s: return None
            s = re.sub(r"[./]", "-", s)
            s = re.sub(r"\s+", "", s)
            ts = pd.to_datetime(s, errors="coerce")
            return None if pd.isna(ts) else pd.Timestamp(ts)

        if mode == "선택박스":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**시작일**")
                years = list(range(min_dt.year, max_dt.year+1))
                y_start = st.selectbox("년", years, index=0, key="y_start")
                m_start = st.selectbox("월", list(range(1,13)), index=max(0, min_dt.month-1), key="m_start")
                max_day_s = _cal.monthrange(y_start, m_start)[1]
                d_start = st.selectbox("일", list(range(1, max_day_s+1)),
                                       index=max(0, min(min_dt.day, max_day_s)-1), key="d_start")
                start = pd.Timestamp(year=y_start, month=m_start, day=d_start)
            with c2:
                st.markdown("**종료일**")
                y_end = st.selectbox("년 ", years, index=len(years)-1, key="y_end")
                m_end = st.selectbox("월 ", list(range(1,13)), index=max(0, max_dt.month-1), key="m_end")
                max_day_e = _cal.monthrange(y_end, m_end)[1]
                d_end = st.selectbox("일 ", list(range(1, max_day_e+1)),
                                     index=max(0, min(max_dt.day, max_day_e)-1), key="d_end")
                end = pd.Timestamp(year=y_end, month=m_end, day=d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            c1, c2 = st.columns(2)
            with c1:
                start_txt = st.text_input("시작일 (YYYY-MM-DD)", value=min_dt.strftime("%Y-%m-%d"))
            with c2:
                end_txt = st.text_input("종료일 (YYYY-MM-DD)", value=max_dt.strftime("%Y-%m-%d"))
            range_txt = st.text_input("또는 한 줄로 (예: 2023-05-01 ~ 2024-04-30)", value="")
            if range_txt.strip():
                parts = re.split(r"\s*[~〜–—]\s*", range_txt.strip())
                if len(parts) == 2:
                    start_txt, end_txt = parts[0].strip(), parts[1].strip()
            start = _parse_date_str(start_txt)
            end_raw = _parse_date_str(end_txt)
            end = None if end_raw is None else (end_raw + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

        if (start is not None) and (end is not None):
            if start > end:
                st.error("시작일이 종료일보다 클 수 없습니다.")
            else:
                mask = (df["_filter_dt"] >= start) & (df["_filter_dt"] <= end)
                before = len(df)
                df = df.loc[mask].drop(columns=["_filter_dt"]).copy()
                st.caption(f"필터 적용: {date_col} ∈ [{start.date()} ~ {end.normalize().date()}], 행수 {before} → {len(df)}")
        else:
            df = df.drop(columns=["_filter_dt"], errors="ignore")
    else:
        st.warning("선택한 컬럼에서 유효한 날짜를 찾지 못해 날짜 필터를 건너뜁니다.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ==============================
# 3) 생존분석(KM) — 시간 단위/격자 + 호환 모드
# ==============================
st.header("3) 현재 생존분석 (KM)")

unit = st.radio("시간 격자(단위)", ["월(표준)", "주(표준)", "0.5개월+사다리꼴(호환)"], horizontal=True, index=0)
is_week = unit.startswith("주")
is_half = unit.startswith("0.5")

with st.expander("🧩 레거시 호환 모드 (옵션)", expanded=False):
    use_active_correction = st.checkbox("진행중 보정 적용 (0.8×, 28일=1개월)", value=False)
    cutoff_date = st.date_input("컷오프 날짜", value=pd.to_datetime("2025-07-13").date())
    st.caption("설명: 활성인데 마지막 수업일이 컷오프 이전이면 중단으로 간주하고 done_month를 실개월×0.8로 보정합니다.")

# Horizon 입력 (단위에 맞춰)
if is_week:
    H = st.number_input("Horizon (주)", min_value=8, max_value=240, value=52, step=1)
elif is_half:
    H = st.number_input("Horizon (개월, 호환)", min_value=6, max_value=60, value=36, step=1)
else:
    H = st.number_input("Horizon (개월)", min_value=6, max_value=60, value=36, step=1)

# 이벤트/기간 원본
state = df[col("tutoring_state")].astype(str).str.lower()
events = state.isin(STOP).values
dur_m = to_num(df[col("done_month")]).fillna(0.0).values  # months(4wk)

# 진행중 보정
if use_active_correction:
    crda_col = col("crda") or col("fst_pay_date")
    lst_col  = col("lst_tutoring_datetime")
    if crda_col is not None and lst_col is not None:
        crda = pd.to_datetime(df[crda_col], errors="coerce")
        lst  = pd.to_datetime(df[lst_col], errors="coerce")
        cutoff_ts = pd.to_datetime(cutoff_date)

        active_mask = ~state.isin(STOP)
        to_finish = active_mask & lst.notna() & (lst < cutoff_ts)

        # 28일=1개월 환산
        actual_m = (lst - crda).dt.total_seconds()/(60*60*24*28.0)
        dm = to_num(df[col("done_month")])
        need_corr = to_finish & dm.notna() & actual_m.notna() & (dm > actual_m)
        dm_corr = dm.copy()
        dm_corr.loc[need_corr] = actual_m[need_corr] * 0.8

        events = (state.isin(STOP) | to_finish).values
        dur_m  = dm_corr.fillna(0.0).values
    else:
        st.warning("진행중 보정을 위해 crda/fst_pay_date & lst_tutoring_datetime 컬럼이 필요합니다.")

# KM 계산
if is_half:
    # half-month grid uses monthly S, then special trapezoid AURC
    S_all = km_bins_timegrid(dur_m, events, int(H), unit="month")
else:
    grid_unit = "week" if is_week else "month"
    S_all = km_bins_timegrid(dur_m, events, int(H), unit=grid_unit)

# AURC 계산
def calc_aurc(S, H):
    if is_half:
        return aurc_half_trapezoid(S, H)
    else:
        return aurc_sum_left(S, H)

A_all = calc_aurc(S_all, H)
med_idx = median_survival_index(S_all)

n_total = len(df)
n_stop = int(events.sum())
n_active = n_total - n_stop

label_unit = "주" if is_week else "개월"
aurc_label = "0.5개월 사다리꼴" if is_half else f"{label_unit} 합"

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f"<div class='card'><div class='label'>분석 대상</div><div class='value'>{n_total:,}</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><div class='label'>중단 수업</div><div class='value'>{n_stop:,}</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><div class='label'>활성 수업</div><div class='value'>{n_active:,}</div></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='card'><div class='label'>AURC ({aurc_label}; 0~{int(H)})</div><div class='value'>{A_all:.2f}</div></div>", unsafe_allow_html=True)

# 곡선
x = np.arange(0, int(H)+1)
fig, ax = plt.subplots(figsize=(8,4))
ax.step(x, np.concatenate([[1.0], S_all]), where="post", label="전체(KM)")
ax.set_ylim(0,1.02); ax.set_xlabel(label_unit); ax.set_ylabel("생존확률 S(t)"); ax.grid(alpha=.3); ax.legend()
st.pyplot(fig)

# ---- 구매 개월수별 생존 곡선 (KM) ----
st.subheader("구매 개월수별 생존 곡선 (KM)")
cohort_col_plot = col("fst_months") or col("fst_fst_months")
if cohort_col_plot is None:
    st.caption("코호트 컬럼(fst_months / fst_fst_months)을 찾지 못해 곡선을 그릴 수 없습니다.")
else:
    cm = pd.to_numeric(df[cohort_col_plot], errors="coerce")
    cohorts = [("전체", None)]
    for m in [1,3,6,12]:
        if (cm == m).any():
            cohorts.append((f"{m}개월 구매", m))
    figc, axc = plt.subplots(figsize=(8,4))
    axc.step(x, np.concatenate([[1.0], S_all]), where="post", label="전체")
    for label, m in cohorts[1:]:
        g = df.loc[cm == m]
        if len(g) == 0: continue
        e = g[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
        d = to_num(g[col("done_month")]).fillna(0.0).values
        if is_half:
            Sg = km_bins_timegrid(d, e, int(H), unit="month")
        else:
            Sg = km_bins_timegrid(d, e, int(H), unit=("week" if is_week else "month"))
        axc.step(x, np.concatenate([[1.0], Sg]), where="post", label=label)
    axc.set_ylim(0,1.02); axc.set_xlabel(label_unit); axc.set_ylabel("생존확률 S(t)"); axc.grid(alpha=.3); axc.legend()
    st.pyplot(figc)

# ---- 이탈률 표(KM) ----
st.subheader("월별/주별 이탈률 (KM 정의)")
S_post = np.concatenate([[1.0], S_all])
haz_km = 1.0 - (S_post[1:] / np.clip(S_post[:-1], 1e-12, None))
churn_cum = 1.0 - S_all
churn_df = pd.DataFrame({
    label_unit: np.arange(1, int(H)+1),
    "이탈률(KM, %)": (haz_km*100).round(2),
    "누적 이탈률(%, 1-S(t))": (churn_cum*100).round(2)
})
st.dataframe(churn_df, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ==============================
# 4) 구간 개선 목표 설정 (단위 인지)
# ==============================
st.header("4) 구간 개선 목표 설정")

mode = st.radio("세그먼트 정의 방식", ["간단 근사(격자 단위)", "이벤트-앵커 근사(회차 스케줄 유도)(비활성화 상태)"], index=0, horizontal=True)

h_base = survival_to_hazards(S_all).copy()

def render_cards(cards, unit_name):
    for name, cur, new, pct in cards:
        st.write(f"**{name}** — 현재 이탈률: {cur:.2f}% → 개선율 {pct}% → 개선 후 {new:.2f}% ({unit_name} 기준)")

if mode.startswith("간단"):
    st.caption(f"선택한 격자({label_unit}) 기준으로 초기 {label_unit}별 세그먼트 3개를 제공합니다.")
    cols = st.columns(3)
    defaults = [70, 60, 50]
    if is_week:
        segments = [("첫 주 (W1)",1,1), ("둘째 주 (W2)",2,2), ("셋째 주 (W3)",3,3)]
    elif is_half:
        # half-month grid still applies improvements on monthly hazards
        segments = [("첫 달 (M1)",1,1), ("둘째 달 (M2)",2,2), ("셋째 달 (M3)",3,3)]
    else:
        segments = [("첫 달 (M1)",1,1), ("둘째 달 (M2)",2,2), ("셋째 달 (M3)",3,3)]
    user_pcts = []
    for i,(nm,a,b) in enumerate(segments):
        with cols[i]:
            user_pcts.append(st.number_input(f"{nm} 개선율(%)", min_value=0, max_value=100, value=defaults[i], step=1))
    cards=[]; h_tmp = h_base.copy()
    for (nm,a,b), pct in zip(segments, user_pcts):
        cur = segment_dropout_rate(S_all, a, b) * 100.0
        h_tmp[a-1:b] = h_tmp[a-1:b] * (1.0 - pct/100.0)
        S_tmp = hazards_to_survival(h_tmp)
        new = segment_dropout_rate(S_tmp, a, b) * 100.0
        cards.append((nm, cur, new, pct))
    render_cards(cards, label_unit)
    S_scn = hazards_to_survival(h_tmp)

else:
    st.caption("앵커=결제일(crda/fst_pay_date). 주당 회차/예정일을 활용한 커버리지 가중은 다음 릴리스에서 활성화됩니다.")
    S_scn = S_all

st.markdown("<hr/>", unsafe_allow_html=True)

# ==============================
# 5) 개선 효과 결과 (단위 인지)
# ==============================
st.header("5) 개선 효과 결과")
def calc_A(S,H):
    return aurc_half_trapezoid(S,H) if is_half else aurc_sum_left(S,H)

A0 = calc_A(S_all, H)
A1 = calc_A(S_scn, H) if 'S_scn' in locals() else A0
dA = A1 - A0
ratio = (A1/A0 - 1.0)*100.0 if A0>0 else np.nan

k1,k2,k3 = st.columns(3)
with k1: st.markdown(f"<div class='card'><div class='label'>Baseline AURC ({aurc_label})</div><div class='value'>{A0:.2f}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='card'><div class='label'>Scenario AURC ({aurc_label})</div><div class='value'>{A1:.2f}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='card'><div class='label'>ΔAURC / 개선율</div><div class='value'>{dA:+.2f} / {ratio:+.1f}%</div></div>", unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.step(np.arange(0,int(H)+1), np.concatenate([[1.0], S_all]), where="post", label="현재")
ax2.step(np.arange(0,int(H)+1), np.concatenate([[1.0], S_scn if 'S_scn' in locals() else S_all]), where="post", label="개선 후")
ax2.set_ylim(0,1.02); ax2.set_xlabel(label_unit); ax2.set_ylabel("생존확률 S(t)"); ax2.grid(alpha=.3); ax2.legend()
st.pyplot(fig2)

st.markdown("<hr/>", unsafe_allow_html=True)

# ==============================
# 6) 코호트별 개선 효과 요약
# ==============================
st.header("6) 코호트별 개선 효과 요약")

def apply_simple_segments_to_df(df_sub, H, segments_with_pct, is_half, is_week):
    e = df_sub[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
    d_m = to_num(df_sub[col("done_month")]).fillna(0.0).values
    if is_half:
        S = km_bins_timegrid(d_m, e, int(H), unit="month")
    else:
        S = km_bins_timegrid(d_m, e, int(H), unit=("week" if is_week else "month"))
    A0 = calc_A(S, H)
    h = survival_to_hazards(S).copy()
    for (a,b,pct) in segments_with_pct:
        h[a-1:b] = h[a-1:b] * (1.0 - pct/100.0)
    S2 = hazards_to_survival(h); A1 = calc_A(S2, H)
    return A0, A1

if 'user_pcts' in locals():
    if is_week:
        segs = [(1,1,user_pcts[0]), (2,2,user_pcts[1]), (3,3,user_pcts[2])]
    else:
        segs = [(1,1,user_pcts[0]), (2,2,user_pcts[1]), (3,3,user_pcts[2])]
else:
    segs = [(1,1,70), (2,2,60), (3,3,50)]

A0_all, A1_all = apply_simple_segments_to_df(df, H, segs, is_half, is_week)
cohort_rows = [["전체", len(df), round(A0_all,2), round(A1_all,2), round(A1_all-A0_all,2), round((A1_all/A0_all-1.0)*100.0,1) if A0_all>0 else np.nan]]

cohort_col2 = col("fst_months") or col("fst_fst_months")
if cohort_col2 is not None:
    cm2 = pd.to_numeric(df[cohort_col2], errors="coerce")
    for m in [1,3,6,12]:
        g = df.loc[cm2==m]
        if len(g)==0: continue
        A0_g, A1_g = apply_simple_segments_to_df(g, H, segs, is_half, is_week)
        cohort_rows.append([f"{m}개월 구매", len(g), round(A0_g,2), round(A1_g,2), round(A1_g-A0_g,2), round((A1_g/A0_g-1.0)*100.0,1) if A0_g>0 else np.nan])

out = pd.DataFrame(cohort_rows, columns=["구분","N","현재 AUC","개선 후 AUC","증가량","개선율(%)"])
order = ["전체","1개월 구매","3개월 구매","6개월 구매","12개월 구매"]
out["__ord"] = pd.Categorical(out["구분"], categories=order, ordered=True)
st.dataframe(out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True), use_container_width=True)

st.caption("설명: 격자를 월/주로 바꿔 AURC와 이탈률을 확인할 수 있습니다. 호환 모드는 레거시 재현용입니다.")
