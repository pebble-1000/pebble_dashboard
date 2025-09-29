# AURC 계산기 — One-Page PDF Style + Cohorts (FULL v4)

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 주요 기능
- 월/주 단위 토글, 동적 Horizon
- G01~G04 구간 개선율(기본: G02=70%, G03=60%, G04=50%)
- 생존/ΔS/Hazard 시각화
- `fst_months` 코호트(1/3/6/12) 비교
- AURC 결과표, Data QA, 벤치마크 비교, 의사결정 카드(Go/검토/보류)
