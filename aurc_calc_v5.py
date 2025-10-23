\
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as _fm
import plotly.graph_objects as go
import re
import os, glob
import calendar as _cal
import html

st.set_page_config(page_title="이탈률 · AURC 통합 계산기", page_icon="📈", layout="wide")

# === App width limit (robust across Streamlit versions) ===
st.markdown("""
<style>
/* Cap main content width on wide screens */
.block-container{max-width: 1200px !important; margin: 0 auto !important; padding-left: 1rem; padding-right: 1rem;}
/* Fallback selectors for older/newer versions */
.main .block-container{max-width: 1300px !important; margin: 0 auto !important;}
[data-testid="stAppViewContainer"] .main .block-container{max-width: 1300px !important; margin: 0 auto !important;}
/* Ensure charts/images don't overflow the capped width */
.block-container img, .block-container svg, .block-container canvas {max-width: 100%; height: auto;}
.badge-miss{
  display:inline-block; padding:2px 8px; margin:2px 6px 2px 0;
  font-size:12px; line-height:18px; border-radius:9999px;
  background:#FEE2E2; color:#B91C1C; border:1px solid #FECACA;
  font-weight:600; font-family: inherit;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ──────────────── 도표 디자인 개선 (summary-table) ──────────────── */
.summary-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 14px;
  background: #ffffff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* 헤더 스타일 */
.summary-table thead th {
  background: linear-gradient(90deg, #eaf1ff 0%, #f5f9ff 100%);
  color: #2c3e50;
  text-align: center;
  font-weight: 600;
  padding: 12px 10px;
  border-bottom: 2px solid #dde3ed;
}

/* 본문(행) 스타일 */
.summary-table tbody tr:nth-child(odd) {
  background-color: #fdfdfd;
}
.summary-table tbody tr:nth-child(even) {
  background-color: #f7f9fb;
}

/* 셀 스타일 */
.summary-table tbody td {
  padding: 10px 10px;
  border-bottom: 1px solid #ebeff3;
  color: #333;
}

/* 행 hover 시 강조 */
.summary-table tbody tr:hover {
  background-color: #eef4ff;
  transition: background-color 0.2s ease;
}

/* 제목 열 (좌측 고정 컬럼 느낌) */
.summary-table .row-title {
  font-weight: 600;
  color: #34495e;
  background-color: #f0f4fa;
}

/* 숫자 정렬 컬럼 */
.summary-table .col-right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* 둥근 모서리 */
.summary-table thead th:first-child {
  border-top-left-radius: 10px;
}
.summary-table thead th:last-child {
  border-top-right-radius: 10px;
}
.summary-table tbody tr:last-child td:first-child {
  border-bottom-left-radius: 10px;
}
.summary-table tbody tr:last-child td:last-child {
  border-bottom-right-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# === Onui CI Theme (Drop-in: no logic changes) ================================
# Plotly 전역 테마/컬러
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.update(
    # 폰트/여백/범례
    font=dict(family="Pretendard, 'Noto Sans KR', sans-serif", size=13, color="#1E293B"),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right"),
    # 배경
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    # Hover 라벨(부드러운 스카이톤)
    hoverlabel=dict(bgcolor="#E0F2FE", font_size=12, font_color="#1E3A8A"),
)
# Onui 컬러 팔레트 (Primary Blue, Deep Navy, Mint 등)
pio.templates["plotly_white"].layout.colorway = [
    "#3B82F6",  # Primary Blue
    "#1E3A8A",  # Deep Navy
    "#10B981",  # Mint (improvement)
    "#60A5FA",  # Soft Blue
    "#6366F1",  # Indigo
    "#93C5FD",  # Light Blue
]

# CSS (헤더/메트릭/테이블/구분선 등 Onui 스타일)
st.markdown("""
<style>
/* 헤더 계열: 딥네이비 */
h1, h2, h3, h4 {
  color: #1E3A8A !important;
  font-family: Pretendard, 'Noto Sans KR', sans-serif !important;
}

/* 구분선(---)을 블루톤으로 */
hr { border: 1px solid #3B82F6 !important; margin: 1.2rem 0 !important; }

/* Metric 스타일: 값/라벨/델타 */
[data-testid="stMetricValue"] { color: #1E3A8A !important; font-weight: 700 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #3B82F6 !important; font-weight: 600 !important; font-size: 0.92rem !important; }
[data-testid="stMetricDelta"] { color: #10B981 !important; font-weight: 600 !important; }

/* 표(.summary-table) – 카드형 + 오누이 톤 */
.summary-table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  background: #ffffff; border-radius: 10px; overflow: hidden;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05); font-size: 14px;
}
.summary-table thead th {
  background: #F3F4F6; color: #1E3A8A; font-weight: 700;
  text-align: center; padding: 12px 10px; border-bottom: 2px solid #E5E7EB;
}
.summary-table tbody tr:nth-child(odd) { background-color: #FFFFFF; }
.summary-table tbody tr:nth-child(even){ background-color: #F9FAFB; }
.summary-table tbody tr:hover { background-color: #E0F2FE; transition: background-color .2s; }
.summary-table tbody td { padding: 10px 10px; border-bottom: 1px solid #EBEFF3; color: #334155; }
.summary-table .row-title { font-weight: 600; color: #1F2937; background-color: #F1F5F9; }
.summary-table .col-right { text-align: right; font-variant-numeric: tabular-nums; }

/* 섹션 카드(원하면 기존 .section에 적용됨) */
.section {
  background: #ffffff; border: 1px solid #E5E7EB; border-radius: 12px;
  padding: 12px 14px; box-shadow: 0 1px 3px rgba(27,31,35,.04); margin: 1rem 0;
}
            
/* 컴포넌트 박스 간 하단 간격(차트/표 등) */
.element-container { margin-bottom: 1.25rem !important; }
</style>
""", unsafe_allow_html=True)
# ============================================================================ 


# 3) Plotly Hover & 가이드라인(스파이크) 기본값(필요 시 개별 그래프에서 override 가능)
#   - 기존 그래프 코드에 별도 수정 없이, 아래 설정이 기본으로 적용됨

def _apply_interactive_defaults(fig, y_top=1.02, y_title="생존확률 S(t)"):
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(align="left"),
        yaxis=dict(range=[0, y_top]),
        xaxis_title=fig.layout.xaxis.title.text or "",
        yaxis_title=y_title if (fig.layout.yaxis.title.text or "") == "" else fig.layout.yaxis.title.text,
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig


# =====================
# Shared small helpers
# =====================
def _to_num(x):
    return pd.to_numeric(x, errors="coerce")

def _pct(n, d):
    return (float(n) / float(d) * 100.0) if (d and d > 0) else float("nan")

def _pick_col(df, names):
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    # fallback: 부분일치
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(n.lower() in lc for n in names):
            return c
    return None

def _parse_date_text(s: str):
    s = (s or "").strip()
    s = re.sub(r"[./]", "-", s)
    s = re.sub(r"\s+", "", s)
    ts = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts)


# ==== Column requirements & small helpers (no behavior change) ====
REQUIRED_COLS = [
    "lecture_vt_No","fst_pay_date","tutoring_state",
    "done_month","step","reactiveornot","cycle_count",
]

def _missing_required(df, required=REQUIRED_COLS):
    low = {str(c).strip().lower(): c for c in df.columns}
    return [r for r in required if r.lower() not in low]

def assert_required(df, show_badges=True):
    miss = _missing_required(df)
    if miss:
        if show_badges:
            st.markdown("#### 📋 누락된 필수 컬럼")
            st.markdown("".join(f"<span class='badge-miss'>{m}</span>" for m in miss), unsafe_allow_html=True)
        st.error("필수 컬럼이 누락되어 분석을 진행할 수 없습니다. 위 항목을 추가해 다시 업로드 해주세요.")
        st.stop()

def raw_total_from_reactive(df, reactive_col_name="reactiveornot"):
    if reactive_col_name in df.columns:
        return int((df[reactive_col_name].astype(str) != "T").sum())
    return len(df)

# 공통 STOP 정의 (원 코드와 동일 값)
STOP = {"finish","auto_finish","done"}

# =====================
# 0) 공통 업로더 + 공통 날짜 필터
# =====================
st.title("이탈률 · AURC 통합 계산기")

up = st.file_uploader("CSV 업로드 (한 번만 업로드하면 두 탭에서 공통 사용)", type=["csv"], key="shared_uploader")
if up is None: 
    st.markdown("""
    #### 📋 필수 컬럼 목록
    - lvt: `lecture_vt_No`  
    - 첫 결제일: `fst_pay_date`  
    - 과외상태: `tutoring_state`  
    - done_month: `done_month`  
    - 단계: `step`  
    - 이탈여부: `reactiveornot`  
    - cycle_count: `cycle_count`
    """)
    st.info("CSV를 업로드하면 분석이 시작됩니다.")
    st.stop()

# 인코딩 시도 (원 코드는 기본 read_csv였지만 공통 업로더에서 안정성 위해 가벼운 시도)
df_shared = None
for enc in ["utf-8","cp949","euc-kr"]:
    try:
        df_shared = pd.read_csv(up, encoding=enc)
        break
    except Exception:
        pass
if df_shared is None:
    st.error("인코딩 문제로 파일을 열 수 없습니다. UTF-8/CP949/EUC-KR 등으로 저장 후 다시 시도해주세요.")
    st.stop()

st.markdown(f"불러온 파일: **{html.escape(up.name)}** · 행 수 {len(df_shared):,}")
df_shared.rename(columns=lambda c: str(c).strip(), inplace=True)

# 업로드 직후 1회 컬럼 요구사항 검사
assert_required(df_shared)
# ---- 공통 날짜 필터 (fst_pay_date만) ----
date_col = _pick_col(df_shared, ["fst_pay_date"])
if date_col is None:
    st.error("필수 컬럼(fst_pay_date)이 없습니다.")
    st.stop()

dt = pd.to_datetime(df_shared[date_col], errors="coerce")
if not dt.notna().any():
    st.error("fst_pay_date에 유효한 날짜가 없습니다.")
    st.stop()

min_dt = pd.to_datetime(dt.min()).date()
max_dt = pd.to_datetime(dt.max()).date()


st.subheader("공통 날짜 필터 설정")

c1, c2 = st.columns(2)
with c1:
    start = st.date_input("시작일 (YYYY/DD/MM)", value=min_dt, key="glob_start")
with c2:
    end   = st.date_input("종료일 (YYYY/DD/MM)", value=max_dt, key="glob_end")

end_incl = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
mask = (dt >= pd.Timestamp(start)) & (dt <= end_incl)
df_filtered = df_shared.loc[mask].copy()
date_caption = f"필터: {start} ~ {end} (fst_pay_date 기준) · 행 수: {len(df_filtered):,}"
 
st.caption(date_caption)

if pd.Timestamp(start) > pd.Timestamp(end):
    st.error("시작일이 종료일보다 클 수 없습니다.")
    st.stop()


# ===================================================================
# Tab 1: churn - baseline definitions & presentation
# ===================================================================
def tab_churn(df):
    st.subheader("필수 컬럼")

    assert_required(df, show_badges=False)
    lecture_col  = _pick_col(df, ["lecture_vt_No"])
    fst_col      = _pick_col(df, ["fst_pay_date"])
    state_col    = _pick_col(df, ["tutoring_state"])
    dm_col       = _pick_col(df, ["done_month"])

    step_col     = _pick_col(df, ["step"])
    reactive_col = _pick_col(df, ["reactiveornot"])
    cycle_col    = _pick_col(df, ["cycle_count"])# done_month 클린 (원 코드 동일)
    dm_clean = (
        df[dm_col].astype(str)
          .str.replace("\\u00A0", "", regex=False)
          .str.replace("\\t", "", regex=False)
          .str.replace(",", "", regex=False)
          .replace(r"^\\s*$", "0", regex=True)
    )
    df = df.copy()
    df[dm_col] = pd.to_numeric(dm_clean, errors="coerce").fillna(0.0)

    st.markdown(f"""
        - lvt: `{lecture_col}`  
        - 첫 결제일: `{fst_col}`  
        - 과외상태: `{state_col}`  
        - done_month: `{dm_col}`  
        - 단계: `{step_col}`  
        - 이탈여부: `{reactive_col}`  
        - cycle_count: `{cycle_col}`
    """)


    # 3) 요약 — 분모는 필터 후 전체 행수(raw_total)
    st.subheader("요약")
    # 원 코드에서는 reactiveornot != 'T'를 사용
    raw_total = raw_total_from_reactive(df, reactive_col if (reactive_col in df.columns if reactive_col is not None else False) else 'reactiveornot')
    events_mask = df[state_col].astype(str).str.lower().isin(STOP).values
    dm_vals = pd.to_numeric(df[dm_col], errors="coerce").fillna(0.0).values

    # dm1 구성요소 (탭1 정의)
    dm1_components = 0
    if (step_col is not None) and (reactive_col is not None):
        step_vals = _to_num(df[step_col])
        react_p   = df[reactive_col].astype(str).str.upper().str.strip().eq("P")
        dm_ser    = _to_num(df[dm_col]).fillna(0.0)

        pre_match   = int(((step_vals <= 2) & react_p).sum())
        post_match  = int(((step_vals == 3) & react_p).sum())
        if cycle_col is not None:
            cycle_vals = _to_num(df[cycle_col])
            first_to_second_before = int((((dm_ser > 0) & (dm_ser <= 0.25)) & (cycle_vals == 2) & react_p).sum())
        else:
            first_to_second_before = 0
        first_after_dm1  = int((((dm_ser > 0) & (dm_ser <= 1.0)) & react_p).sum())
        second_after_dm1 = max(0, first_after_dm1 - first_to_second_before)
        dm1_components   = pre_match + post_match + first_to_second_before + second_after_dm1

    dm_b2 = int((((dm_vals > 1.0) & (dm_vals <= 2.0)) & events_mask).sum())
    dm_b3 = int((((dm_vals > 2.0) & (dm_vals <= 3.0)) & events_mask).sum())
    dm_b4 = int((((dm_vals > 3.0) & (dm_vals <  4.0)) & events_mask).sum())
    dm_lt4 = int(dm1_components + dm_b2 + dm_b3 + dm_b4)

    c1, c2, c3, c4 = st.columns(4)

    metric_card_html = """
    <div style="background-color:#f8f9fa;border-radius:10px;padding:10px 0;text-align:center;">
    <div style="font-size:16px;font-weight:600;color:#555;">{label}</div>
    <div style="font-size:22px;font-weight:700;color:#000;">{value}</div>
    <div style="font-size:20px;font-weight:600;color:#007bff;margin-top:4px;">{pct}</div>
    </div>
    """
    with c1:
        st.markdown(metric_card_html.format(label="신규 활성 수업 수(raw)", value=f"{raw_total:,}", pct=""), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html.format(label="DM 1 총이탈(1이하)", value=f"{dm1_components:,}", pct=f"{_pct(dm1_components, raw_total):.2f}%"), unsafe_allow_html=True)
    with c3:
        dm_leq3 = int(dm1_components + dm_b2 + dm_b3)
        st.markdown(metric_card_html.format(label="DM 3 총이탈(3이하)", value=f"{dm_leq3:,}", pct=f"{_pct(dm_leq3, raw_total):.2f}%"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html.format(label="DM 4 총이탈(4미만)", value=f"{dm_lt4:,}", pct=f"{_pct(dm_lt4, raw_total):.2f}%"), unsafe_allow_html=True)

    # 4) 윈도우 상세 (표시용) — 원 코드 유지
    st.subheader("윈도우 상세")
    # 단계별 표 (탭1 정의)
    def _build_step_rows(df_local, churn_local):
        rows = []
        if (step_col is None) or (reactive_col is None):
            keys = ["결제","과외 신청서","결제 직후 매칭 전","매칭 직후 첫 수업 전","첫 수업 후 2회차 전",
                    "2회차 수업 후 dm 1.0 이하","매칭 직후 dm 1.0 이하","첫 수업 후 dm 1.0 이하"]
            for k in keys:
                rows.append([k, 0, f"{_pct(0,churn_local):.2f}%"])
            return rows, 0

        step_vals = _to_num(df_local[step_col])
        react_p   = df_local[reactive_col].astype(str).str.upper().str.strip().eq("P")
        dm_ser    = _to_num(df_local[dm_col]).fillna(0.0)
        pay_drop         = int(((step_vals == 1) & react_p).sum())
        apply_drop       = int(((step_vals == 2) & react_p).sum())
        pre_match_drop   = int(((step_vals <= 2) & react_p).sum())
        post_match_drop  = int(((step_vals == 3) & react_p).sum())
        if cycle_col is not None:
            cycle_vals = _to_num(df_local[cycle_col])
            first_to_second_before = int((((dm_ser > 0) & (dm_ser <= 0.25)) & (cycle_vals == 2) & react_p).sum())
        else:
            first_to_second_before = 0
        first_after_dm1  = int((((dm_ser > 0) & (dm_ser <= 1.0)) & react_p).sum())
        second_after_dm1 = max(0, first_after_dm1 - first_to_second_before)
        match_after_dm1  = int(post_match_drop + first_to_second_before + second_after_dm1)
        dm1_comp_local   = int(pre_match_drop + post_match_drop + first_to_second_before + second_after_dm1)

        rows = [
            ["결제",                          pay_drop,               f"{_pct(pay_drop,churn_local):.2f}%"],
            ["과외 신청서",                    apply_drop,             f"{_pct(apply_drop,churn_local):.2f}%"],
            ["결제 직후 매칭 전",              pre_match_drop,         f"{_pct(pre_match_drop,churn_local):.2f}%"],
            ["매칭 직후 첫 수업 전",            post_match_drop,        f"{_pct(post_match_drop,churn_local):.2f}%"],
            ["첫 수업 후 2회차 전",             first_to_second_before, f"{_pct(first_to_second_before,churn_local):.2f}%"],
            ["2회차 수업 후 dm 1.0 이하",       second_after_dm1,       f"{_pct(second_after_dm1,churn_local):.2f}%"],
            ["매칭 직후 dm 1.0 이하",           match_after_dm1,        f"{_pct(match_after_dm1,churn_local):.2f}%"],
            ["첫 수업 후 dm 1.0 이하",          first_after_dm1,        f"{_pct(first_after_dm1,churn_local):.2f}%"],
        ]
        return rows, dm1_comp_local

    step_rows, dm1_comp_used = _build_step_rows(df, raw_total)
    def _table_wide(rowdata, title):
        labels = [r[0] for r in rowdata]
        counts = ["" if r[1] is None else f"{int(r[1]):,}" for r in rowdata]
        rates  = [r[2] for r in rowdata]
        t = [f"<h4 style='margin-top:0.2em'>{html.escape(str(title))}</h4><table class='summary-table'>"]
        t.append("<thead><tr><th>지표</th>")
        for lbl in labels:
            t.append(f"<th>{html.escape(str(lbl))}</th>")
        t.append("</tr></thead><tbody>")
        t.append("<tr><td class='row-title'>총이탈</td>")
        for c in counts:
            t.append(f"<td class='col-right'>{c}</td>")
        t.append("</tr>")
        t.append("<tr><td class='row-title'>이탈률(%)</td>")
        for r in rates:
            t.append(f"<td class='col-right'>{html.escape(str(r))}</td>")
        t.append("</tr>")
        t.append("</tbody></table>")
        return "".join(t)

    st.markdown(f"<div class='section'>{_table_wide(step_rows, '단계별 이탈 현황')}</div>", unsafe_allow_html=True)

    dm_leq3 = int(dm1_comp_used + dm_b2 + dm_b3)
    bucket_rows = [
        ["dm 1 (dm≤1)", dm1_comp_used, f"{_pct(dm1_comp_used, raw_total):.2f}%"],
        ["dm 2 (1<dm≤2)", dm_b2, f"{_pct(dm_b2, raw_total):.2f}%"],
        ["dm 3 (2<dm≤3)", dm_b3, f"{_pct(dm_b3, raw_total):.2f}%"],
        ["dm 4 (3<dm<4)", dm_b4, f"{_pct(dm_b4, raw_total):.2f}%"],
        ["dm 3이하 (dm≤3)", dm_leq3, f"{_pct(dm_leq3, raw_total):.2f}%"],
        ["dm 4미만 (dm<4)", dm_lt4, f"{_pct(dm_lt4, raw_total):.2f}%"],
    ]
    st.markdown(f"<div class='section'>{_table_wide(bucket_rows, 'DM별 이탈 현황')}</div>", unsafe_allow_html=True)

    # 5) 월별 보기 (원 코드 유지)
    st.subheader("월별 보기 (선택)")
    show_monthly = st.checkbox("월별로 나눠보기", value=False, key="churn01_monthly")
    if show_monthly:
        # 월 산출은 fst_pay_date 기준
        month_key = fst_col
        _fst_series = pd.to_datetime(df[month_key], errors="coerce")
        df["_month"] = _fst_series.dt.to_period("M")
        months = sorted(df["_month"].dropna().unique())
        for per in months:
            dfm = df[df["_month"] == per].copy()
            if dfm.empty:
                continue
            if reactive_col is not None and reactive_col in dfm.columns:
                raw_total_m = int((dfm[reactive_col].astype(str) != "T").sum())
            else:
                raw_total_m = len(dfm)
            step_rows_m, dm1_m = _build_step_rows(dfm, raw_total_m)
            dm_vals_m = _to_num(dfm[dm_col]).fillna(0.0).values
            events_mask_m = dfm[state_col].astype(str).str.lower().isin(STOP).values
            dm_b2_m = int((((dm_vals_m > 1.0) & (dm_vals_m <= 2.0)) & events_mask_m).sum())
            dm_b3_m = int((((dm_vals_m > 2.0) & (dm_vals_m <= 3.0)) & events_mask_m).sum())
            dm_b4_m = int((((dm_vals_m > 3.0) & (dm_vals_m <  4.0)) & events_mask_m).sum())
            bucket_rows_m = [
                ["dm 1 (dm≤1)", dm1_m, f"{_pct(dm1_m, raw_total_m):.2f}%"],
                ["dm 2 (1<dm≤2)", dm_b2_m, f"{_pct(dm_b2_m, raw_total_m):.2f}%"],
                ["dm 3 (2<dm≤3)", dm_b3_m, f"{_pct(dm_b3_m, raw_total_m):.2f}%"],
                ["dm 4 (3<dm<4)", dm_b4_m, f"{_pct(dm_b4_m, raw_total_m):.2f}%"],
                ['dm 3이하 (dm≤3)', int(dm1_m + dm_b2_m + dm_b3_m), f"{_pct(int(dm1_m + dm_b2_m + dm_b3_m), raw_total_m):.2f}%"],
                ["dm 4미만 (dm<4)", int(dm1_m + dm_b2_m + dm_b3_m + dm_b4_m), f"{_pct(int(dm1_m + dm_b2_m + dm_b3_m + dm_b4_m), raw_total_m):.2f}%"],
            ]
            with st.expander(f"월별 보기 · {per}  (행수: {raw_total_m:,})", expanded=False):
                st.markdown(_table_wide(step_rows_m, "단계별 이탈 현황"), unsafe_allow_html=True)
                st.markdown(_table_wide(bucket_rows_m, "DM별 이탈 현황"), unsafe_allow_html=True)

    st.caption("분모=기간 필터 후 T를 뺀 행수(raw). 퍼센트=소수점 2자리. 정렬: 결제→과외 신청서→결제직후 매칭전→매칭직후 첫수업전→첫수업 후 2회차 전→2회차 후 dm≤1→매칭 직후 dm≤1→첫 수업 후 dm≤1→dm1→dm2→dm3→dm4→dm<4.")



# ===================================================================
# Tab 2: KM 기반 (AURC/ΔAURC) — DM 요약은 탭1 정의로 산출
# ===================================================================
def tab_km_app(df):
    # Fonts
    _kor_candidates = ["AppleGothic","Malgun Gothic","NanumGothic","NanumBarunGothic","Noto Sans CJK KR","Noto Sans KR","Pretendard"]
    _avail = set(f.name for f in _fm.fontManager.ttflist)
    for _nm in _kor_candidates:
        if _nm in _avail:
            matplotlib.rcParams["font.family"] = _nm
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    # st.subheader("1) 날짜 필터 (공통 적용)")
    # st.caption(date_caption)

    # Column mapping
    orig_cols = df.columns.tolist()
    lowmap = {c.lower(): c for c in orig_cols}
    def col(name): return lowmap.get(name.lower())

    need = ["done_month","tutoring_state"]
    miss = [n for n in need if col(n) is None]
    if miss:
        st.error(f"필수 컬럼 누락: {miss}")
        return

    # --- Tab1과 동일한 분모 규칙 적용: reactiveornot != "T" ---
    reactive_col_km = col("reactiveornot")
    if reactive_col_km is not None and reactive_col_km in df.columns:
        mask_raw = df[reactive_col_km].astype(str) != "T"
        df_km = df.loc[mask_raw].copy()
    else:
        df_km = df.copy()
    raw_total_km = len(df_km)
    
    # ---------------- KM helpers (원 코드 유지) ----------------
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
        for i in range(H):
            auc += prev; prev = S[i]
        return float(auc)

    def aurc_half_trapezoid(S, H):
        H = int(H)
        if H <= 0: return 0.0
        y = np.empty(2*H + 1, float); y[0] = 1.0; y[1] = 1.0
        for m in range(1, H):
            y[2*m] = float(S[m-1]); y[2*m+1] = float(S[m-1])
        y[2*H] = float(S[H-1])
        return float(np.trapz(y, dx=0.5))

    def hazards_to_survival(h):
        S=[]; s=1.0
        for v in h:
            s *= (1.0 - float(v)); S.append(s)
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

    # 3) KM 분석
    st.subheader("1) 현재 생존분석 (KM)")
    unit = st.radio("시간 격자(단위)", ["월(표준)", "주(표준)"], horizontal=True, index=0, key="km01_unit")
    is_week = unit.startswith("주"); is_half = unit.startswith("0.5")
    if is_week:
        H = st.number_input("Horizon (주)", min_value=8, max_value=240, value=52, step=1, key="km01_H_week")
    else:
        H = st.number_input("Horizon (개월)", min_value=6, max_value=60, value=36, step=1, key="km01_H_month")

    state = df_km[col("tutoring_state")].astype(str).str.lower()
    events = state.isin(STOP).values
    dur_m = _to_num(df_km[col("done_month")]).fillna(0.0).values

    S_all = km_bins_timegrid(dur_m, events, int(H), unit=("week" if is_week else "month") if not is_half else "month")
    def calc_aurc(S, H):
        return aurc_sum_left(S, H)

    A_all = calc_aurc(S_all, H)
    n_total = len(df_km); n_stop = int(events.sum()); n_active = n_total - n_stop
    label_unit = "주" if is_week else "개월"; aurc_label = f"{label_unit} 합"

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("분석 대상", f"{n_total:,}")
    with c2: st.metric("중단 수업", f"{n_stop:,}")
    with c3: st.metric("활성 수업", f"{n_active:,}")
    with c4: st.metric(f"AURC ({aurc_label}; 0~{int(H)})", f"{A_all:.2f}")

    # KM 생존곡선 (Plotly)
    x_vals = np.arange(0, int(H)+1)
    fig_km = go.Figure()
    fig_km.add_trace(go.Scatter(
        x=x_vals,
        y=[1.0] + list(S_all),
        mode="lines+markers",
        name="현재(KM)",
        line=dict(width=2),
        line_shape="linear",
        hovertemplate="t=%{x}<br>S(t)=%{y:.4f}<extra></extra>"
    ))
    fig_km.update_layout(
        title="현재 KM 생존곡선",
        xaxis_title=label_unit,
        yaxis_title="생존확률 S(t)",
        yaxis=dict(range=[0,1.02]),
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(align="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right")
    )
    fig_km.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig_km.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    st.plotly_chart(fig_km, use_container_width=True)


    # 3-1) DM 요약 (탭1 정의 동일)
    st.subheader("DM 요약")
    st.caption("탭 1에서 정의한 DM 구간별 이탈 요약과 동일한 방식으로 산출됩니다.")
    reactive_col_km = col("reactiveornot")  # already used above
    dm_vals_km = _to_num(df_km[col("done_month")]).fillna(0.0).values
    events_mask_km = df_km[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
    dm_b2_km = int((((dm_vals_km > 1.0) & (dm_vals_km <= 2.0)) & events_mask_km).sum())
    dm_b3_km = int((((dm_vals_km > 2.0) & (dm_vals_km <= 3.0)) & events_mask_km).sum())
    dm_b4_km = int((((dm_vals_km > 3.0) & (dm_vals_km <  4.0)) & events_mask_km).sum())

    step_col_km = col("step"); cycle_col_km = col("cycle_count")
    dm1_components_km = 0
    if (step_col_km is not None) and (reactive_col_km is not None):
        step_vals_km = _to_num(df_km[step_col_km])
        react_p_km   = df_km[reactive_col_km].astype(str).str.upper().str.strip().eq("P")
        dm_ser_km    = _to_num(df_km[col("done_month")]).fillna(0.0)
        pre_match_km  = int(((step_vals_km <= 2) & react_p_km).sum())
        post_match_km = int(((step_vals_km == 3) & react_p_km).sum())
        if cycle_col_km is not None:
            cycle_vals_km = _to_num(df_km[cycle_col_km])
            first_to_second_before_km = int((((dm_ser_km > 0) & (dm_ser_km <= 0.25)) & (cycle_vals_km == 2) & react_p_km).sum())
        else:
            first_to_second_before_km = 0
        first_after_dm1_km  = int((((dm_ser_km > 0) & (dm_ser_km <= 1.0)) & react_p_km).sum())
        second_after_dm1_km = max(0, first_after_dm1_km - first_to_second_before_km)
        dm1_components_km   = pre_match_km + post_match_km + first_to_second_before_km + second_after_dm1_km

    dm_leq3_km = int(dm1_components_km + dm_b2_km + dm_b3_km)
    dm_lt4_total_km = int(dm_leq3_km + dm_b4_km)

    dm_tbl = pd.DataFrame({
        "구간": ["dm 1 (dm≤1)", "dm 2 (1<dm≤2)", "dm 3 (2<dm≤3)", "dm 4 (3<dm<4)", "dm 3이하 (dm≤3)", "dm 4미만 (dm<4)"],
        "총이탈": [dm1_components_km, dm_b2_km, dm_b3_km, dm_b4_km, dm_leq3_km, dm_lt4_total_km],
        "이탈률(%)": [
            f"{_pct(dm1_components_km, raw_total_km):.2f}%",
            f"{_pct(dm_b2_km, raw_total_km):.2f}%",
            f"{_pct(dm_b3_km, raw_total_km):.2f}%",
            f"{_pct(dm_b4_km, raw_total_km):.2f}%",
            f"{_pct(dm_leq3_km, raw_total_km):.2f}%",
            f"{_pct(dm_lt4_total_km, raw_total_km):.2f}%"]
    })
    st.dataframe(dm_tbl, use_container_width=True)

    # 4) 현재 AUC 요약 표
    st.subheader("2) 현재 AURC 분석 결과")

    # --- 현재 AURC 분석 그래프 (전체 & 코호트별) ---
    st.markdown("**생존곡선 (전체 & 코호트별)**")
    x_vals = np.arange(0, int(H)+1)
    figc = go.Figure()
    # 전체 생존곡선
    figc.add_trace(go.Scatter(x=x_vals, y=[1.0] + list(S_all), mode="lines+markers", name="전체"))
    # 코호트 생존곡선
    cohort_col_plot = col("fst_months")
    if cohort_col_plot is not None:
        cm = _to_num(df_km[cohort_col_plot])
        for m in [1,3,6,12]:
            g = df_km.loc[np.isfinite(cm) & (np.isclose(cm, m, atol=0.25))]
            if len(g) == 0:
                continue
            e = g[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
            d = _to_num(g[col("done_month")]).fillna(0.0).values
            Sg = km_bins_timegrid(d, e, int(H), unit=("week" if is_week else "month") if not is_half else "month")
            figc.add_trace(go.Scatter(x=x_vals, y=[1.0] + list(Sg), mode="lines+markers", name=f"{m}개월 구매"))
    figc.update_traces(line=dict(width=2))
    figc.update_layout(
        title="현재 AURC 생존곡선",
        xaxis_title=label_unit,
        yaxis_title="생존확률 S(t)",
        yaxis=dict(range=[0,1.02]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right")
    )
    # === 코호트별 색상만 변경 (선명도 개선) ===
    cohort_colors = {
        "전체": "#1E3A8A",
        "1개월 구매": "#3B82F6",
        "3개월 구매": "#F59E0B",
        "6개월 구매": "#10B981",
        "12개월 구매": "#EF4444"
    }

    for trace in figc.data:
        if trace.name in cohort_colors:
            trace.update(line=dict(color=cohort_colors[trace.name], width=2))

    st.plotly_chart(figc, use_container_width=True)

    # AURC summary table
    def _make_S(df_sub):
        e = df_sub[col("tutoring_state")].astype(str).str.lower().isin(STOP).values
        d_m = _to_num(df_sub[col("done_month")]).fillna(0.0).values
        unit_here = ("week" if is_week else "month") if not is_half else "month"
        return km_bins_timegrid(d_m, e, int(H), unit=unit_here), int(e.sum()), len(df_sub)
    def median_survival_index(S):
        for i,v in enumerate(S, start=1):
            if v <= 0.5: return i
        return float("inf")
    def _median_in_months(S):
        idx = median_survival_index(S)
        if np.isinf(idx): return float("inf")
        return round(idx/4.0, 1) if is_week and not is_half else round(float(idx), 1)
    def _row(label, df_sub):
        S, n_stop, n = _make_S(df_sub)
        auc = aurc_half_trapezoid(S, H) if is_half else aurc_sum_left(S, H)
        med = _median_in_months(S); stop_rate = (n_stop/n*100.0) if n>0 else np.nan
        return {"구분": label, "샘플 수": f"{n:,}", "중단율": f"{stop_rate:.1f}%",
                f"AUC ({int(H)}개월)": f"{auc:.2f}", "중위 생존기간": ("∞" if np.isinf(med) else f"{med:.1f}")}
    rows = [_row("전체", df)]
    cohort_col_for_table = (col("fst_months") or col("fst_fst_months"))
    if cohort_col_for_table is not None:
        cm = _to_num(df[cohort_col_for_table])
        for m in [1,3,6,12]:
            g = df.loc[cm==m]
            if len(g): rows.append(_row(f"{m}개월 구매", g))
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 5) 구간 개선 목표 설정 (월 격자만 Tab1 정의 기반 이탈률 표시)
    st.subheader("3) 구간 개선 목표 설정")
    seg_tabA = st.container()

    with seg_tabA:
        st.caption("표시는 Tab1 정의(구성요소 dm1 + 이벤트 dm2/3) 기준으로 현재 이탈률을 보여줍니다.")

        # 현재 구간 이탈률 (표시 전용)
        cur_M1 = _pct(dm1_components_km, raw_total_km)
        cur_M2 = _pct(dm_b2_km,          raw_total_km)
        cur_M3 = _pct(dm_b3_km,          raw_total_km)

        c1,c2,c3 = st.columns(3)
        with c1:
            m1p = st.number_input("M1 개선율(%)", 0, 100, 70, 1, key="km01_m1p")
            st.metric("현재 이탈률(M1)", f"{cur_M1:.2f}%")
        with c2:
            m2p = st.number_input("M2 개선율(%)", 0, 100, 60, 1, key="km01_m2p")
            st.metric("현재 이탈률(M2)", f"{cur_M2:.2f}%")
        with c3:
            m3p = st.number_input("M3 개선율(%)", 0, 100, 50, 1, key="km01_m3p")
            st.metric("현재 이탈률(M3)", f"{cur_M3:.2f}%")

        # === 전이효과(Spillover) 반영 토글: 카드/가이드테이블의 TO-BE 표시에만 영향 ===
        spill = st.toggle(
            "전이효과 반영",
            value=False,  # 기본 OFF
            help="ON: 앞 구간 개선이 뒤 구간 母수 증가까지 반영(시스템 전이). OFF: 각 구간만 초기분모 기준으로 직접 축소."
        )

        # --- 전이효과 설명 (Expander) ---
        with st.expander("💡 전이효과(Spillover) 계산 방식 안내", expanded=False):
            st.markdown("""
            ### 🧭 전이효과란?
            앞 단계의 개선이 다음 단계의 유지율에도 영향을 주는 **연쇄 효과**를 의미합니다.  
            예를 들어 M1 구간을 개선하면, M2·M3 구간으로 더 많은 학생이 넘어가게 됩니다.

            ---

            #### 🎯 OFF (기본) — 구간별 직접 비교
            - 각 구간(M1, M2, M3)을 **서로 독립적**으로 계산합니다.  
            - 예: “M2 이탈률을 5% 줄이겠다” → **M2 구간만 정확히 5% 감소**  
            - M1을 개선해도 M2·M3에는 **영향이 없습니다.**  
            - ✅ **팀별 KPI 확인 / 구간별 개선 가이드용**으로 적합합니다.

            ---

            #### 🔗 ON — 실제 흐름(전체 시나리오) 반영
            - M1을 개선하면 M2·M3 구간으로 **더 많은 학생이 진입**합니다.  
            - 즉, 앞 구간 개선이 **뒤 구간의 잔존 고객 수(母수)** 까지 늘리는 효과를 반영합니다.  
            - 따라서 M2의 확률은 줄더라도, 진입 인원이 늘면 **절대 이탈 수는 커질 수도 있습니다.**  
            - ✅ **전체 유지기간(AURC)** 기준으로,  
            개선 시 전체 고객이 **평균적으로 얼마나 오래 남는지(ΔAURC)** 분석할 때 적합합니다.

            ---

            💡 **요약**
            | 구분 | 의미 | 활용 목적 |
            |------|------|-----------|
            | **OFF (기본)** | 각 구간만 독립적으로 개선 | 구간별 성과 / 팀 KPI |
            | **ON** | 앞 구간 개선의 파급효과 반영 | 전체 잔존 시나리오 분석 |
            """)

        # 개선은 KM 해저드 기반으로 적용 (표시 기준과 계산 엔진을 분리)
        def hazards_to_survival(h):
            S=[]; s=1.0
            for v in h:
                s *= (1.0 - float(v)); S.append(s)
            return np.array(S, float)

        def survival_to_hazards(S):
            h=[]; prev=1.0
            for cur in S:
                cur = float(cur)
                val = 0.0 if prev <= 0 else (1.0 - cur/prev)
                h.append(max(0.0, min(1.0, val)))
                prev = cur
            return np.array(h, float)

        h_base = survival_to_hazards(S_all).copy()
        h_imp  = h_base.copy()
        for i, p in enumerate([m1p, m2p, m3p]):
            if i < len(h_imp):
                h_imp[i] = h_imp[i] * (1.0 - p/100.0)

        # 개선 후 전체 생존곡선 (AURC/비교그래프용: 항상 전이 반영)
        S_scn = hazards_to_survival(h_imp)

        def _bucket_loss(S, m):
            try: m = int(m)
            except: return 0.0
            if m < 1 or not hasattr(S, "__len__") or len(S) == 0: return 0.0
            if m == 1:
                return float(max(0.0, 1.0 - float(S[0])))
            if (m-1) >= len(S): return 0.0
            prev_, cur_ = (float(S[m-2]) if m-2 >= 0 else 1.0), float(S[m-1])
            return float(max(0.0, prev_ - cur_))

        def _loss_from(prev_survival: float, hazard: float) -> float:
            return float(prev_survival) * float(max(0.0, min(1.0, hazard)))

        # --- 카드용 TO-BE ---
        if spill:
            # 전이 ON: 개선된 전체 생존곡선에서 월손실 비율
            tob_M1 = _bucket_loss(S_scn, 1) * 100.0
            tob_M2 = _bucket_loss(S_scn, 2) * 100.0
            tob_M3 = _bucket_loss(S_scn, 3) * 100.0
            st.caption("전이효과 ON: 앞 구간(M1) 개선 시 뒤 구간(M2·M3)으로 더 많이 넘어가는 효과까지 반영된 전체 시나리오입니다.")

        else:
            # 전이 OFF: 현재 DM 구간 이탈률(cur_M*)을 기준으로 직접 축소 (0% 개선 시 완전히 동일)
            tob_M1 = cur_M1 * (1 - m1p / 100.0)
            tob_M2 = cur_M2 * (1 - m2p / 100.0)
            tob_M3 = cur_M3 * (1 - m3p / 100.0)
            st.caption("전이효과 OFF: 각 구간(M1·M2·M3)을 독립적으로 계산해, 입력한 개선율만큼 해당 구간의 이탈률만 줄어듭니다.")

        # === AURC 계산에 사용할 생존곡선 선택 ===
        # 전이효과 OFF일 때는 카드 기준으로 구한 TO-BE(M1~M3)를 이용해
        # 각 구간별 해저드를 역산해 생존곡선을 새로 구성합니다.
        if spill:
            S_scn_for_aurc = S_scn  # 전이효과 ON: 기존 그대로 사용
        else:
            h_off = h_base.copy()

            # 현재 생존률(prev)과 카드에서 구한 절대 손실률(TO-BE%)로 역산
            prev_M1 = 1.0
            prev_M2 = float(S_all[0]) if len(S_all) >= 1 else 0.0
            prev_M3 = float(S_all[1]) if len(S_all) >= 2 else 0.0

            loss1 = float(tob_M1) / 100.0
            loss2 = float(tob_M2) / 100.0
            loss3 = float(tob_M3) / 100.0

            if len(h_off) >= 1 and prev_M1 > 0:
                h_off[0] = max(0.0, min(1.0, loss1 / prev_M1))
            if len(h_off) >= 2 and prev_M2 > 0:
                h_off[1] = max(0.0, min(1.0, loss2 / prev_M2))
            if len(h_off) >= 3 and prev_M3 > 0:
                h_off[2] = max(0.0, min(1.0, loss3 / prev_M3))
            # 나머지 월(M4+)은 기존 해저드 유지
            S_scn_for_aurc = hazards_to_survival(h_off)

        # 카드 출력
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.metric("TO-BE 이탈률(M1)", f"{tob_M1:.2f}%", delta=f"{(tob_M1 - cur_M1):+.2f}pp")
            st.caption(f"≈ 절감 {max(0.0, (cur_M1 - tob_M1)/100.0 * raw_total_km):,.0f}건")
        with gc2:
            st.metric("TO-BE 이탈률(M2)", f"{tob_M2:.2f}%", delta=f"{(tob_M2 - cur_M2):+.2f}pp")
            st.caption(f"≈ 절감 {max(0.0, (cur_M2 - tob_M2)/100.0 * raw_total_km):,.0f}건")
        with gc3:
            st.metric("TO-BE 이탈률(M3)", f"{tob_M3:.2f}%", delta=f"{(tob_M3 - cur_M3):+.2f}pp")
            st.caption(f"≈ 절감 {max(0.0, (cur_M3 - tob_M3)/100.0 * raw_total_km):,.0f}건")


        # 표: M1..min(12, H) 구간별 가이드 (현재/TO-BE/Δpp/절감 건수)
        max_rows = int(min(12, H))
        rows = []
        for m in range(1, max_rows+1):
            if   m == 1: cur = _pct(dm1_components_km, raw_total_km)
            elif m == 2: cur = _pct(dm_b2_km,          raw_total_km)
            elif m == 3: cur = _pct(dm_b3_km,          raw_total_km)
            else:
                cur = _bucket_loss(S_all, m) * 100.0  # KM 기반 현재값

            if spill or m >= 4:
                tob = _bucket_loss(S_scn, m) * 100.0
            else:
                if m == 1:
                    prev_surv = 1.0
                elif m == 2:
                    prev_surv = float(S_all[0]) if len(S_all) >= 1 else 0.0
                else:  # m == 3
                    prev_surv = float(S_all[1]) if len(S_all) >= 2 else 0.0
                hazm = float(h_imp[m-1]) if (m-1) < len(h_imp) else 0.0
                tob = _loss_from(prev_surv, hazm) * 100.0

            dpp = tob - cur
            saved = max(0.0, (cur - tob)/100.0 * raw_total_km)
            rows.append([m, f"{cur:.2f}%", f"{tob:.2f}%", f"{dpp:+.2f}pp", f"{saved:,.0f}"])

        guide_df = pd.DataFrame(rows, columns=["월(M)", "현재 이탈률", "TO-BE 이탈률", "Δpp", "절감(건)"])
        with st.expander("구간별 TO-BE 가이드 테이블 (M1~M12)", expanded=False):
            st.dataframe(guide_df, use_container_width=True)

    # 6) 개선 효과 결과
    st.subheader("4) 개선 효과 결과")

    with st.expander("📈 ΔAURC(평균 유지기간 증가량) 해석 가이드", expanded=False):
            st.markdown("""
            ΔAURC는 **‘개선 전후 생존곡선 면적 차이’**, 즉 평균 유지기간 증가량을 의미합니다.  
            전이효과(Spillover) ON/OFF에 따라 해석이 달라집니다.

            | 모드 | 계산 방식 | 해석 포인트 | 활용 용도 |
            |------|-------------|--------------|------------|
            | **OFF (기본)** | 각 구간을 독립적으로 개선 | “M1 10% 개선 → 평균 유지 +0.8개월” | 팀별 KPI / 단기 목표 |
            | **ON** | 앞 구간 개선이 뒤 구간에도 전이 | “M1 개선으로 전체 유지 +0.6개월” | 전사 전략 / ROI 판단 |

            💬 **요약:**  
            - OFF는 **직접 개선 효과** 중심, 수치상 더 크게 보일 수 있습니다.  
            - ON은 **실제 잔존 구조 변화**를 반영하며, 더 현실적인 시나리오입니다.
            """)
    
    A0 = calc_aurc(S_all, H)
    A1 = calc_aurc(locals().get("S_scn_for_aurc", S_all), H)
    dA = A1 - A0
    ratio = (A1/A0 - 1.0)*100.0 if A0 > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    with c1: 
        st.metric("Baseline AURC (현재 기준)", f"{A0:.2f}")
        st.caption("실제 데이터 기반 AURC")
    with c2: 
        st.metric("Scenario AURC (개선 시나리오)", f"{A1:.2f}")
        st.caption("입력한 개선율 반영 후 AURC")
    with c3:
        st.metric("ΔAURC / 개선율", f"{dA:+.2f}", f"{ratio:+.1f}%")
        st.caption("AURC 개선폭(절대값·상대비율)")
    
    # 비교 그래프
    x_vals = list(range(0, int(H)+1))
    fig2 = go.Figure()

    # 현재
    fig2.add_trace(go.Scatter(
        x=x_vals, y=[1.0] + list(S_all),
        mode="lines+markers", name="현재",
        line=dict(width=2), line_shape="linear"
    ))
    # 개선 후
    fig2.add_trace(go.Scatter(
        x=x_vals, y=[1.0] + list(locals().get("S_scn_for_aurc", S_all)),
        mode="lines+markers", name="개선 후",
        line=dict(width=2), line_shape="linear"
    ))

    fig2.data[0].update(hovertemplate="%{x}개월<br>현재 S(t)=%{y:.4f}<extra></extra>")
    fig2.data[1].update(hovertemplate="개선 후 S(t)=%{y:.4f}<extra></extra>")

    fig2.update_layout(
        title="개선 전후 생존곡선 비교",
        xaxis_title=label_unit,
        yaxis_title="생존확률 S(t)",
        yaxis=dict(range=[0, 1.02]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
        hoverlabel=dict(align="left")
    )
    fig2.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig2.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    
    for t in fig2.data:
        if t.name == "현재":
            t.update(line=dict(color="#2563EB", width=3),
                    marker=dict(size=6, color="#2563EB"),
                    name="현재 (Baseline)")
        elif t.name == "개선 후":
            t.update(line=dict(color="#F97316", width=3),
                    marker=dict(size=6, color="#F97316"),
                    name="개선 후 (Scenario)")

    st.plotly_chart(fig2, use_container_width=True)


# =====================
# Layout (Tabs)
# =====================
tab1, tab2 = st.tabs(["📊 이탈률 (Churn Rate)", "📈 AURC (KM 기반)"])

with tab1:
    tab_churn(df_filtered)

with tab2:
    tab_km_app(df_filtered)
