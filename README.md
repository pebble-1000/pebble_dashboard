# 🧮 AURC · 이탈률 통합 계산기

**오누이(Onui)** 내부 데이터 분석용 Streamlit 앱입니다.  
고객 이탈률(Churn Rate)과 AURC(Area Under Retention Curve)를 한 번에 분석하고,  
구간별 개선 시나리오(ΔAURC)를 시각적으로 비교할 수 있습니다.

---

## 🚀 실행 방법

### 1️⃣ 환경 준비
```bash
# 가상환경 생성 및 활성화 (선택)
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 필수 라이브러리 설치
pip install -r requirements.txt
```

> **주의:** Python 3.9 이상 권장  
> Streamlit ≥ 1.32, Plotly ≥ 5.20

---

### 2️⃣ 실행
```bash
streamlit run aurc_calc.py
```

앱이 실행되면 브라우저에서 아래 주소로 자동 열립니다:
```
http://localhost:8501
```

---

## 🧾 주요 기능

### 📊 탭 1: 이탈률 (Churn Rate)
- CSV 업로드 한 번으로 전체 데이터 공통 사용
- `reactiveornot != "T"` 기준으로 분석 대상 자동 필터링
- **단계별 이탈 현황(G01~G04)** 및 **DM별 요약(dm≤1, dm≤3, dm<4)**
- **월별 보기(선택)** — `fst_pay_date` 기준 자동 그룹핑
- 오누이 CI 기반 테이블 디자인 (블루·민트 계열)

---

### 📈 탭 2: AURC (KM 기반)
- KM (Kaplan–Meier) 방식으로 생존곡선 산출
- 코호트별 생존곡선 (1/3/6/12개월)
- **AURC 계산** 및 **중위 생존기간**
- 개선 시나리오(M1/M2/M3) 적용 → ΔAURC 시각화
- Hover 시 한 X축 기준으로 **현재 vs 개선 후 생존확률** 동시 표시

---

## 🎨 디자인 가이드 (오누이 CI 반영)
- 폰트: Pretendard / Noto Sans KR
- Primary: `#3B82F6` (Blue)  
  Accent: `#10B981` (Mint Green)  
  Text: `#1E3A8A` (Deep Navy)
- 표/카드/메트릭은 통일된 라운드형 디자인

---

## 🧩 입력 데이터 형식

| 컬럼명 | 설명 | 필수 여부 |
|--------|------|------------|
| `lecture_vt_No` | 과외 ID | ✅ |
| `fst_pay_date` | 첫 결제일 | ✅ |
| `tutoring_state` | 과외 상태 (finish/auto_finish/done 포함 시 종료) | ✅ |
| `done_month` | 종료까지의 경과개월 | ✅ |
| `step` | 단계 코드(G01~G04) | ✅ |
| `reactiveornot` | 이탈 여부 (‘T’ 제외 대상) | ✅ |
| `cycle_count` | 회차 카운트 | ✅ |
| `fst_months` | 코호트(1/3/6/12개월 등) | ⚙️ 선택 |

---

## 📊 결과 해석
| 지표 | 의미 |
|------|------|
| **AURC** | 잔존 곡선의 면적, 높을수록 유지력 우수 |
| **ΔAURC** | 개선 시나리오 대비 AURC 증가분 |
| **중위 생존기간** | 잔존율 50%에 도달하는 시점 |
| **DM 1/3/4** | 구간별 이탈 누적 비중 요약 |

---

## 🧠 참고
- KM 기반 생존곡선은 **센서링(아직 종료되지 않은 고객)**을 포함합니다.  
- 개선 시나리오의 M1/M2/M3은 각각 **초기·중기·후기 이탈률 개선율**을 의미합니다.
- ΔAURC는 AURC 상승분을 통해 **잔존 기간(개월)** 증가 효과를 직관적으로 확인할 수 있습니다.

---

## 🛠️ 버전 관리
| 버전 | 주요 변경사항 |
|------|----------------|
| **v0.1** | 초기 버전 (단순비 + KM 탭) |
| **v0.4 (current)** | 공통 업로더 + 날짜 필터 + 오누이 CI 디자인 적용 |
| **v0.5 (예정)** | 개선율 자동 시뮬레이션 / ΔAURC 시각화 향상 |

---

## 👥 담당자
- 데이터 분석팀: **천우석**
- 프로젝트: *고객 이탈 예측 & AURC 계산기 개발*
- 문의: internal@onuii.ai

---

### 🧭 예시 실행 화면
> (추가 예정: `screenshots/main_ui.png`, `screenshots/km_curve.png`, `screenshots/churn_summary.png`)
