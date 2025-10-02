import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as _fm
import calendar as _cal
import re
import os, glob
import plotly.graph_objects as go

st.set_page_config(page_title="AURC 잔존 분석 (월/주 격자 + 호환모드)", layout="centered")

# ---- Global CSS (app-wide) ----
def inject_global_css():
    st.markdown("""
    <style>
    /* layout */
    .block-container { padding-top: 1.6rem; padding-bottom: 3.2rem; max-width: 800px; }

    /* brand palette */
    :root{
      --brand1:#6a8dff; --brand2:#7f4bff; --brand3:#a64fff;
      --card-bg:#ffffff; --border:rgba(0,0,0,.08); --text:#222;
    }

    /* section pill title (optional helper) */
    .section-pill{
      background:linear-gradient(135deg,var(--brand1),var(--brand2) 60%,var(--brand3));
      color:#fff; font-weight:700; border-radius:12px; padding:10px 14px; display:inline-block;
    }

    /* metric cards  */
    .card{ background:var(--card-bg); border:1px solid var(--border); border-radius:16px;
           padding:12px 14px; box-shadow:0 3px 14px rgba(0,0,0,.05); text-align:center; }
    .card .label{ font-size:12px; color:#666; }
    .card .value{ font-size:22px; font-weight:700; margin-top:4px; }

    /* delta badge */
    .badge{display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px;}
    .badge.pos{background:#e9f8ef; color:#1e7b3a; border:1px solid #bfe6ce;}
    .badge.neg{background:#fdecec; color:#b03a37; border:1px solid #f5c1bf;}

    /* segment cards (구간 개선 목표) */
    .seg-card{ background:#fff; border:1px solid var(--border); border-radius:14px;
               padding:16px; box-shadow:0 3px 16px rgba(0,0,0,.05); }
    .seg-card--active{ box-shadow:0 6px 18px rgba(104,120,255,.18) !important;
                       border-color: rgba(104,120,255,.35) !important; }
    .seg-head{ font-weight:700; margin-bottom:8px;
               display:flex; justify-content:center; align-items:center; gap:8px; text-align:center; }
    .seg-meta{ font-size:13px; color:#666; margin-top:4px; text-align:center; }
    .seg-kpi{ font-size:13px; color:#444; margin:8px 0 2px; text-align:center; }
    .seg-kpi .v{ font-weight:700; font-size:18px; margin-left:6px; }
    .seg-row { margin-top: 10px; }
    .seg-btn-wrap{ display:flex; justify-content:center; margin-top:12px; }

    /* AUC 요약 카드/테이블 */
    .result-card{ border:1px solid var(--border); border-radius:16px; padding:10px 12px; background:#fff; }
    .gradbar{ height:10px; border-radius:8px;
              background:linear-gradient(90deg,var(--brand1),var(--brand2) 50%,var(--brand3)); }
    .summary-table{ width:100%; border-collapse:separate; border-spacing:0; margin-top:12px; }
    .summary-table thead th{
      position:sticky; top:0; z-index:2;
      color:#fff; font-weight:700; padding:12px 10px; text-align:left;
      background:linear-gradient(90deg,var(--brand1),var(--brand2) 60%,var(--brand3));
    }
    .summary-table thead th:first-child{ border-top-left-radius:12px; }
    .summary-table thead th:last-child{ border-top-right-radius:12px; }
    .summary-table tbody td{ padding:12px 10px; border-bottom:1px solid var(--border); }
    .summary-table tbody tr:nth-child(even) td{ background:#fafbff; }
    .summary-table tbody tr:hover td{ background:#f2f5ff; transition:background .15s ease; }
    .summary-table tbody tr:last-child td{ border-bottom:none; }
    .summary-table .col-right{ text-align:right; }
    .summary-table .row-title{ font-weight:700; }

    /* buttons */
    .stButton>button{
      background:linear-gradient(90deg,var(--brand1),var(--brand2));
      color:#fff; border:0; border-radius:10px; padding:8px 14px; font-weight:600;
      box-shadow:0 2px 8px rgba(0,0,0,.08);
    }
    .stButton>button:hover{ filter:brightness(1.04); }

    /* radios / selects compact */
    div[role="radiogroup"] label { font-weight:500; }
    div[data-baseweb="select"]>div{ min-height: 34px; }
    label{ line-height:1.2; }
    </style>
    """, unsafe_allow_html=True)

# 호출 한 번만!
inject_global_css()

# ---- 폰트 설정 (한글) ----
_kor_candidates = ["AppleGothic","Malgun Gothic","NanumGothic","NanumBarunGothic","Noto Sans CJK KR","Noto Sans KR","Pretendard"]
_avail = set(f.name for f in _fm.fontManager.ttflist)
for _nm in _kor_candidates:
    if _nm in _avail:
        matplotlib.rcParams["font.family"] = _nm
        break
matplotlib.rcParams["axes.unicode_minus"] = False

STOP = {"finish","auto_finish","done","nocard","nopay"}
def to_num(x): return pd.to_numeric(x, errors="coerce")

# ---------------- Core KM / AURC helpers ----------------
def km_bins_timegrid(durations_month, events, H, unit="month"):
    if unit not in {"month","week"}: raise ValueError("unit must be 'month' or 'week'")
    d_m = np.asarray(durations_month, float)
    d = d_m if unit=="month" else d_m * 4.0
    e = np.asarray(events, bool); H = int(H)
    uniq_evt = np.unique(d[e]); S = 1.0; step = {}
    for t in uniq_evt:
        n = np.sum(d >= t); di = np.sum((d == t) & e)
        step[t] = 1.0 if n <= 0 else max(0.0, min(1.0, 1.0 - di / n))
    times = np.array(sorted(step.keys())); S_bins = np.ones(H, float); idx = 0
    for m in range(1, H+1):
        while idx < len(times) and times[idx] <= m:
            S *= step[times[idx]]; idx += 1
        S_bins[m-1] = S
    return S_bins

def aurc_sum_left(S, H):
    H = min(int(H), len(S)); auc, prev = 0.0, 1.0
    for i in range(H): auc += prev; prev = S[i]
    return float(auc)

def aurc_half_trapezoid(S, H):
    H = int(H)
    if H <= 0: return 0.0
    y = np.empty(2*H + 1, float); y[0] = 1.0; y[1] = 1.0
    for m in range(1, H): y[2*m] = float(S[m-1]); y[2*m+1] = float(S[m-1])
    y[2*H] = float(S[H-1]); return float(np.trapz(y, dx=0.5))

def hazards_to_survival(h):
    S=[]; s=1.0
    for v in h: s *= (1.0 - float(v)); S.append(s)
    return np.array(S, float)

def survival_to_hazards(S):
    h=[]; prev=1.0
    for cur in S:
        cur = float(cur); val = 0.0 if prev <= 0 else (1.0 - cur/prev)
        h.append(max(0.0, min(1.0, val))); prev = cur
    return np.array(h, float)

def median_survival_index(S):
    for i,v in enumerate(S, start=1):
        if v <= 0.5: return i
    return float("inf")

def segment_dropout_rate(S, a, b):
    a = max(1, int(a)); b = max(a, int(b))
    S_start = 1.0 if a == 1 else float(S[a-2]); S_end = float(S[b-1])
    return max(0.0, S_start - S_end)

# ---------------- 1) 데이터 업로드 / 자동 로드 ----------------
st.header("1) 데이터 업로드 / 자동 로드")
st.caption("업로드가 없으면 ./data 또는 현재 폴더의 최신 CSV를 자동으로 불러옵니다.")
up = st.file_uploader("CSV 업로드 (필수: done_month, tutoring_state; 코호트: fst_months 또는 fst_fst_months)", type=["csv"])

df = None; loaded_from = None
encodings = ["utf-8","cp949","euc-kr"]
def _try_read(path):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception: pass
    return None, None

if up is not None:
    for enc in encodings:
        try: df = pd.read_csv(up, encoding=enc); loaded_from = f"업로더({enc})"; break
        except Exception: pass
else:
    cand = []
    for root in ["./data", "."]:
        if os.path.isdir(root): cand += sorted(glob.glob(os.path.join(root, "*.csv")), key=os.path.getmtime, reverse=True)
    if len(cand):
        df, enc = _try_read(cand[0])
        if df is not None: loaded_from = f"자동로드: {cand[0]} ({enc})"

if df is None:
    st.error("CSV를 찾지 못했습니다. 업로드하거나 ./data 또는 현재 폴더에 CSV를 두세요."); st.stop()
else:
    st.success(f"불러옴: {loaded_from} — 행수 {len(df):,}")

orig_cols = df.columns.tolist()
lowmap = {c.lower(): c for c in orig_cols}
def col(name): return lowmap.get(name.lower())

need = ["done_month","tutoring_state"]
miss = [n for n in need if col(n) is None]
if miss: st.error(f"필수 컬럼 누락: {miss}"); st.stop()

# ---------------- 2) 날짜 필터 ----------------
st.header("2) 날짜 필터 ")
date_col = col("fst_pay_date") or col("crda")
if date_col is None:
    st.warning("fst_pay_date 또는 crda 컬럼을 찾지 못해 날짜 필터를 건너뜁니다.")
else:
    st.caption(f"날짜 기준 컬럼: **{date_col}** 고정, 없으면 crda로 대체")
    dt = pd.to_datetime(df[date_col], errors="coerce"); df["_filter_dt"] = dt
    if df["_filter_dt"].notna().any():
        min_dt = pd.to_datetime(df["_filter_dt"].min()).date(); max_dt = pd.to_datetime(df["_filter_dt"].max()).date()
        mode = st.radio("입력 방식", ["직접 입력(YYYY-MM-DD)", "선택박스"], horizontal=True, index=0)
        def _parse_date_str(s: str):
            s = (s or "").strip()
            if not s: return None
            s = re.sub(r"[./]", "-", s); s = re.sub(r"\s+", "", s)
            ts = pd.to_datetime(s, errors="coerce"); return None if pd.isna(ts) else pd.Timestamp(ts)
        if mode == "직접 입력(YYYY-MM-DD)":
            c1, c2 = st.columns(2)
            with c1: start_txt = st.text_input("시작일 (YYYY-MM-DD)", value=min_dt.strftime("%Y-%m-%d"))
            with c2: end_txt = st.text_input("종료일 (YYYY-MM-DD)", value=max_dt.strftime("%Y-%m-%d"))
            range_txt = st.text_input("또는 한 줄로 (예: 2023-05-01 ~ 2024-04-30)", value="")
            if range_txt.strip():
                parts = re.split(r"\s*[~〜–—]\s*", range_txt.strip())
                if len(parts) == 2: start_txt, end_txt = parts[0].strip(), parts[1].strip()
            start = _parse_date_str(start_txt); end_raw = _parse_date_str(end_txt)
            end = None if end_raw is None else (end_raw + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**시작일**")
                years = list(range(min_dt.year, max_dt.year+1))
                y_start = st.selectbox("년", years, index=0, key="y_start")
                m_start = st.selectbox("월", list(range(1,13)), index=max(0, min_dt.month-1), key="m_start")
                max_day_s = _cal.monthrange(y_start, m_start)[1]
                d_start = st.selectbox("일", list(range(1, max_day_s+1)), index=max(0, min(min_dt.day, max_day_s)-1), key="d_start")
                start = pd.Timestamp(year=y_start, month=m_start, day=d_start)
            with c2:
                st.markdown("**종료일**")
                y_end = st.selectbox("년 ", years, index=len(years)-1, key="y_end")
                m_end = st.selectbox("월 ", list(range(1,13)), index=max(0, max_dt.month-1), key="m_end")
                max_day_e = _cal.monthrange(y_end, m_end)[1]
                d_end = st.selectbox("일 ", list(range(1, max_day_e+1)), index=max(0, min(max_dt.day, max_day_e)-1), key="d_end")
                end = pd.Timestamp(year=y_end, month=m_end, day=d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if (start is not None) and (end is not None):
            if start > end: st.error("시작일이 종료일보다 클 수 없습니다.")
            else:
                mask = (df["_filter_dt"] >= start) & (df["_filter_dt"] <= end)
                before = len(df); df = df.loc[mask].drop(columns=["_filter_dt"]).copy()
                st.caption(f"필터 적용: {date_col} ∈ [{start.date()} ~ {end.normalize().date()}], 행수 {before} → {len(df)}")
        else:
            df = df.drop(columns=["_filter_dt"], errors="ignore")

st.markdown("---")

# ---------------- 3) 현재 생존분석 (KM) ----------------
st.header("3) 현재 생존분석 (KM)")
unit = st.radio("시간 격자(단위)", ["월(표준)", "주(표준)", "0.5개월+사다리꼴(호환)"], horizontal=True, index=0)
is_week = unit.startswith("주"); is_half = unit.startswith("0.5")
with st.expander("🧩 레거시 호환 모드 (옵션)", expanded=False):
    use_active_correction = st.checkbox("진행중 보정 적용 (0.8×, 28일=1개월)", value=False)
    cutoff_date = st.date_input("컷오프 날짜", value=pd.to_datetime("2025-07-13").date())
if is_week: H = st.number_input("Horizon (주)", min_value=8, max_value=240, value=52, step=1)
elif is_half: H = st.number_input("Horizon (개월, 호환)", min_value=6, max_value=60, value=36, step=1)
else: H = st.number_input("Horizon (개월)", min_value=6, max_value=60, value=36, step=1)

state = df[col("tutoring_state")].astype(str).str.lower()
events = state.isin(STOP).values
dur_m = to_num(df[col("done_month")]).fillna(0.0).values

if use_active_correction:
    crda_col = col("crda") or col("fst_pay_date"); lst_col  = col("lst_tutoring_datetime")
    if crda_col is not None and lst_col is not None:
        crda = pd.to_datetime(df[crda_col], errors="coerce"); lst  = pd.to_datetime(df[lst_col], errors="coerce")
        cutoff_ts = pd.to_datetime(cutoff_date)
        active_mask = ~state.isin(STOP); to_finish = active_mask & lst.notna() & (lst < cutoff_ts)
        actual_m = (lst - crda).dt.total_seconds()/(60*60*24*28.0)
        dm = to_num(df[col("done_month")])
        need_corr = to_finish & dm.notna() & actual_m.notna() & (dm > actual_m)
        dm_corr = dm.copy(); dm_corr.loc[need_corr] = actual_m[need_corr] * 0.8
        events = (state.isin(STOP) | to_finish).values; dur_m  = dm_corr.fillna(0.0).values
    else: st.warning("진행중 보정을 위해 crda/fst_pay_date & lst_tutoring_datetime 컬럼이 필요합니다.")

if is_half: S_all = km_bins_timegrid(dur_m, events, int(H), unit="month")
else: S_all = km_bins_timegrid(dur_m, events, int(H), unit=("week" if is_week else "month"))

def calc_aurc(S, H): return aurc_half_trapezoid(S, H) if is_half else aurc_sum_left(S, H)
A_all = calc_aurc(S_all, H)
n_total = len(df); n_stop = int(events.sum()); n_active = n_total - n_stop
label_unit = "주" if is_week else "개월"; aurc_label = "0.5개월 사다리꼴" if is_half else f"{label_unit} 합"

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f"<div class='card'><div class='label'>분석 대상</div><div class='value'>{n_total:,}</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><div class='label'>중단 수업</div><div class='value'>{n_stop:,}</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><div class='label'>활성 수업</div><div class='value'>{n_active:,}</div></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='card'><div class='label'>AURC ({aurc_label}; 0~{int(H)})</div><div class='value'>{A_all:.2f}</div></div>", unsafe_allow_html=True)

x = np.arange(0, int(H)+1)
fig, ax = plt.subplots(figsize=(8,4))
ax.step(x, np.concatenate([[1.0], S_all]), where="post", label="전체(KM)")
ax.set_ylim(0,1.02); ax.set_xlabel(label_unit); ax.set_ylabel("생존확률 S(t)"); ax.grid(alpha=.3); ax.legend()
st.pyplot(fig)

st.subheader("구매 개월수별 생존 곡선 (KM)")
cohort_col_plot = (col("fst_months") or col("fst_fst_months"))
if cohort_col_plot is None:
    st.caption("코호트 컬럼(fst_months / fst_fst_months)을 찾지 못해 곡선을 그릴 수 없습니다.")
else:
    cm = pd.to_numeric(df[cohort_col_plot], errors="coerce")
    cohorts = [("전체", None)]
    for m in [1,3,6,12]:
        if (cm == m).any(): cohorts.append((f"{m}개월 구매", m))
    figc, axc = plt.subplots(figsize=(8,4))
    axc.step(x, np.concatenate([[1.0], S_all]), where="post", label="전체")
    for label, m in cohorts[1:]:
        g = df.loc[cm == m]
        if len(g) == 0: continue
        e = g[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
        d = to_num(g[col("done_month")]).fillna(0.0).values
        Sg = km_bins_timegrid(d, e, int(H), unit=("week" if is_week else "month") if not is_half else "month")
        axc.step(x, np.concatenate([[1.0], Sg]), where="post", label=label)
    axc.set_ylim(0,1.02); axc.set_xlabel(label_unit); axc.set_ylabel("생존확률 S(t)"); axc.grid(alpha=.3); axc.legend()
    st.pyplot(figc)

# ---------------- 현재 AUC 분석 결과 (요약 표) ----------------
st.subheader("현재 AURC 분석 결과")
#st.markdown("<div class='result-card'><div class='gradbar'></div></div>", unsafe_allow_html=True)

def _make_S(df_sub):
    e = df_sub[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
    d_m = to_num(df_sub[col("done_month")]).fillna(0.0).values
    unit_here = ("week" if is_week else "month") if not is_half else "month"
    return km_bins_timegrid(d_m, e, int(H), unit=unit_here), int(e.sum()), len(df_sub)

def _median_in_months(S):
    idx = median_survival_index(S)
    if np.isinf(idx):  # 중위 생존기간이 관측된 범위 밖
        return float("inf")
    return round(idx/4.0, 1) if is_week and not is_half else round(float(idx), 1)

def _row(label, df_sub):
    S, n_stop, n = _make_S(df_sub)
    auc = calc_aurc(S, H)
    med = _median_in_months(S)
    stop_rate = (n_stop/n * 100.0) if n>0 else np.nan
    return {
        "구분": label,
        "샘플 수": f"{n:,}",
        "중단율": f"{stop_rate:.1f}%",
        f"AUC ({int(H)}개월)": f"{auc:.2f}",
        "중위 생존기간": ("∞" if np.isinf(med) else f"{med:.1f}")
    }

rows = []
rows.append(_row("전체", df))

cohort_col_for_table = (col("fst_months") or col("fst_fst_months"))
if cohort_col_for_table is not None:
    cm = pd.to_numeric(df[cohort_col_for_table], errors="coerce")
    for m in [1,3,6,12]:
        g = df.loc[cm==m]
        if len(g)==0: continue
        rows.append(_row(f"{m}개월 구매", g))

def _render_table(rows):
    headers = ["구분", "샘플 수", "중단율", f"AUC ({int(H)}개월)", "중위 생존기간"]
    html = ["<table class='summary-table'>"]
    html.append("<thead><tr>")
    for h in headers: html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")
    for r in rows:
        html.append("<tr>")
        html.append(f"<td class='row-title'>{r['구분']}</td>")
        html.append(f"<td class='col-right'>{r['샘플 수']}</td>")
        html.append(f"<td class='col-right'>{r['중단율']}</td>")
        html.append(f"<td class='col-right'>{r[f'AUC ({int(H)}개월)']}</td>")
        html.append(f"<td class='col-right'>{r['중위 생존기간']}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)

st.markdown("<div class='result-card'>"+_render_table(rows)+"</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- 4) 구간 개선 목표 설정 (카드형 UI) ----------------
st.header("4) 구간 개선 목표 설정")
st.caption("dm = done_month.<br>월 격자: M1은 dm ≤ 1, M2는 1 < dm ≤ 2, M3는 2 < dm ≤ 3.<br>주 격자: 첫/둘째/셋째 주로 해석.", unsafe_allow_html=True)

mode = st.radio(
    "세그먼트 정의 방식",
    ["간단 근사(격자 단위)", "이벤트-앵커 근사(회차 스케줄 유도)(비활성화 상태)"],
    index=0, horizontal=True
)

h_base = survival_to_hazards(S_all).copy()

def _seg_defs():
    if is_week:
        segs = [("첫 주 (W1)", 1, 1, "week_index ≤ 1", "🚀"),
                ("둘째 주 (W2)", 2, 2, "1 < week_index ≤ 2", "📘"),
                ("셋째 주 (W3)", 3, 3, "2 < week_index ≤ 3", "⏱")]
        keys = ["pct_w1","pct_w2","pct_w3"]
        defaults = [70, 60, 50]
        unit_name = "주"
    else:
        segs = [("첫 달 (M1)", 1, 1, "dm ≤ 1", "🚀"),
                ("둘째 달 (M2)", 2, 2, "1 < dm ≤ 2", "📘"),
                ("셋째 달 (M3)", 3, 3, "2 < dm ≤ 3", "⏱")]
        keys = ["pct_m1","pct_m2","pct_m3"]
        defaults = [70, 60, 50]
        unit_name = "개월"
    return segs, keys, defaults, unit_name

segs, keys, defaults, unit_name = _seg_defs()

if mode.startswith("간단"):
    # 현재 구간 이탈률 계산
    current_rates = []
    for (_, a, b, _, _) in segs:
        cur = segment_dropout_rate(S_all, a, b) * 100.0
        current_rates.append(cur)

    cols = st.columns(3)
    user_pcts = []
    h_tmp = h_base.copy()
    new_rates = []

    for i, ((title, a, b, hint, emoji), key, default) in enumerate(zip(segs, keys, defaults)):
        # 현재(세션) 값으로 active 스타일 적용
        cur_pct = int(st.session_state.get(key, default))
        active_cls = " seg-card--active" if cur_pct > 0 else ""

        with cols[i]:
            st.markdown(
                f"""
                <div class='seg-card{active_cls}'>
                    <div class='seg-head'>{emoji} {title}</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<div class='seg-meta'>[ {hint} ]</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='seg-kpi'>현재 이탈률: <span class='v'>{current_rates[i]:.2f}%</span></div>",
                unsafe_allow_html=True
            )

            pct = st.number_input(
                "개선율(%)",
                min_value=0, max_value=100,
                value=cur_pct, step=1,
                key=key, help=hint
            )
            user_pcts.append(pct)

            # 개선율 적용 후 새로운 이탈률
            h_tmp[a-1:b] = h_tmp[a-1:b] * (1.0 - pct/100.0)
            S_tmp = hazards_to_survival(h_tmp)
            new_rate = segment_dropout_rate(S_tmp, a, b) * 100.0
            new_rates.append(new_rate)

            st.markdown(
                f"<div class='seg-kpi'>개선 후 이탈률: <span class='v'>{new_rate:.2f}%</span></div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='seg-row seg-btn-wrap'>", unsafe_allow_html=True)
    st.button("개선 효과 계산하기")
    st.markdown("</div>", unsafe_allow_html=True)

    S_scn = hazards_to_survival(h_tmp)
else:
    st.caption("앵커=결제일(crda/fst_pay_date). 주당 회차/예정일 기반 가중은 다음 릴리스에서 활성화됩니다.")
    S_scn = S_all

st.markdown("---")

# ---------------- 5) 개선 효과 결과 ----------------
st.header("5) 개선 효과 결과")

A0 = calc_aurc(S_all, H)
A1 = calc_aurc(S_scn, H) if 'S_scn' in locals() else A0
dA = A1 - A0
ratio = (A1/A0 - 1.0)*100.0 if A0>0 else np.nan
cls = "pos" if dA >= 0 else "neg"

k1,k2,k3 = st.columns(3)
with k1: st.markdown(f"<div class='card'><div class='label'>Baseline AURC</div><div class='value'>{A0:.2f}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='card'><div class='label'>Scenario AURC</div><div class='value'>{A1:.2f}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='card'><div class='label'>ΔAURC / 개선율</div>"
                     f"<div class='value'><span class='badge {cls}' style='font-size:20px; padding:1px 14px;'>{dA:+.2f} / {ratio:+.1f}%</span></div>"
                     f"</div>", unsafe_allow_html=True)

x_vals = list(range(0, int(H)+1))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=x_vals, y=[1.0] + list(S_all),
    mode="lines+markers", name="현재",
    hovertemplate = f"기간: %{{x}}{label_unit}<br>생존확률: %{{y:.3f}}<extra></extra>"

))
fig2.add_trace(go.Scatter(
    x=x_vals, y=[1.0] + list(S_scn if 'S_scn' in locals() else S_all),
    mode="lines+markers", name="개선 후",
    hovertemplate = f"기간: %{{x}}{label_unit}<br>생존확률: %{{y:.3f}}<extra></extra>"
))
fig2.update_traces(line=dict(width=2))
fig2.update_layout(
    title="개선 전후 생존곡선 비교",
    xaxis_title=label_unit, yaxis_title="생존확률 S(t)",
    yaxis=dict(range=[0,1.02]), template="plotly_white",
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    #margin=dict(t=40, r=10, b=40, l=10)
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- 6) 코호트별 개선 효과 요약 ----------------
st.header("6) 코호트별 개선 효과 요약")

st.caption("비고: 날짜 컬럼은 fst_pay_date(없으면 crda)로 고정.<br>업로드가 없으면 자동으로 최신 CSV를 불러옵니다.", unsafe_allow_html=True)

def apply_simple_segments_to_df(df_sub, H, segments_with_pct, is_half, is_week):
    e = df_sub[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
    d_m = to_num(df_sub[col("done_month")]).fillna(0.0).values
    S = km_bins_timegrid(d_m, e, int(H), unit=("week" if is_week else "month") if not is_half else "month")
    A0 = calc_aurc(S, H)
    h = survival_to_hazards(S).copy()
    for (a,b,pct) in segments_with_pct:
        h[a-1:b] = h[a-1:b] * (1.0 - pct/100.0)
    S2 = hazards_to_survival(h); A1 = calc_aurc(S2, H)
    return A0, A1

if 'user_pcts' in locals():
    segs_for_apply = [(1,1,user_pcts[0]), (2,2,user_pcts[1]), (3,3,user_pcts[2])]
else:
    segs_for_apply = [(1,1,70), (2,2,60), (3,3,50)]

A0_all, A1_all = apply_simple_segments_to_df(df, H, segs_for_apply, is_half, is_week)
cohort_rows = [["전체", len(df), round(A0_all,2), round(A1_all,2), round(A1_all-A0_all,2),
                round((A1_all/A0_all-1.0)*100.0,1) if A0_all>0 else np.nan]]

cohort_col2 = (col("fst_months") or col("fst_fst_months"))
if cohort_col2 is not None:
    cm2 = pd.to_numeric(df[cohort_col2], errors="coerce")
    for m in [1,3,6,12]:
        g = df.loc[cm2==m]
        if len(g)==0: continue
        A0_g, A1_g = apply_simple_segments_to_df(g, H, segs_for_apply, is_half, is_week)
        cohort_rows.append([f"{m}개월 구매", len(g), round(A0_g,2), round(A1_g,2), round(A1_g-A0_g,2),
                            round((A1_g/A0_g-1.0)*100.0,1) if A0_g>0 else np.nan])

out = pd.DataFrame(cohort_rows, columns=["구분","N","현재 AUC","개선 후 AUC","증가량","개선율(%)"])
order = ["전체","1개월 구매","3개월 구매","6개월 구매","12개월 구매"]
out["__ord"] = pd.Categorical(out["구분"], categories=order, ordered=True)

# 스타일 표로 렌더
co = out.copy().sort_values("__ord").drop(columns="__ord", errors="ignore").reset_index(drop=True)

if co.empty:
    st.info("표에 표시할 코호트가 없습니다. (코호트 컬럼 없음 또는 필터 결과 0행)")
else:
    def _fmt_num(x):
        try: return f"{int(x):,}"
        except Exception: return x
    def _fmt_float2(x):
        try: return f"{float(x):.2f}"
        except Exception: return x
    def _fmt_pct1(x):
        try: return f"{float(x):.1f}%"
        except Exception: return x

    if "N" in co.columns: co["N"] = co["N"].map(_fmt_num)
    for c in ["현재 AUC","개선 후 AUC","증가량"]:
        if c in co.columns: co[c] = co[c].map(_fmt_float2)
    if "개선율(%)" in co.columns:
        co["개선율(%)"] = co["개선율(%)"].map(lambda v: "" if (pd.isna(v) or v=="") else _fmt_pct1(v))

    def render_cohort_table_html(df):
        headers_all = ["구분","N","현재 AUC","개선 후 AUC","증가량","개선율(%)"]
        headers = [h for h in headers_all if h in df.columns]
        html = ["<table class='summary-table'>"]
        html.append("<thead><tr>")
        for h in headers: html.append(f"<th>{h}</th>")
        html.append("</tr></thead><tbody>")
        for _, r in df.iterrows():
            html.append("<tr>")
            for h in headers:
                cls = "row-title" if h == "구분" else "col-right"
                html.append(f"<td class='{cls}'>{r[h]}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "\n".join(html)

    st.markdown("<div class='result-card'>" + render_cohort_table_html(co) + "</div>", unsafe_allow_html=True)

    if st.toggle("원본 DataFrame 보기", value=False):
        st.dataframe(out.drop(columns="__ord", errors="ignore"), use_container_width=True)


