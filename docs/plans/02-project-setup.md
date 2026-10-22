# 02. 프로젝트 구조와 실행 환경

- 예상 시간: 4시간
- 선행 단계: [01. 데이터 선정](01-data-selection.md)
- 다음 단계: [03. 분석 파이프라인](03-data-pipeline.md)

## 목표

데이터 획득부터 분석 결과까지 다른 환경에서도 재현할 수 있는 저장소 구조를 만든다.

## 작업

1. 다음 파일과 디렉터리를 준비한다.

   ```text
   README.md
   requirements.txt
   src/
     __init__.py
     pipeline.py
   notebooks/
     analysis_report.ipynb
   data/
     README.md
     processed/
   outputs/
     figures/
   ```

2. Python 3.8 이상을 기준으로 NumPy, Pandas, Matplotlib, Seaborn을 의존성에 명시한다. Plotly는 08단계의 보너스 차트에만 사용한다.
3. 원본과 가공 데이터, 이미지, 출력 차트의 경로를 상대 경로로 관리한다. 개인 컴퓨터의 절대 경로를 코드와 노트북에 넣지 않는다.
4. 대용량 원본 데이터와 이미지의 저장소 포함 여부를 라이선스와 용량 기준으로 정한다. 포함하지 않으면 `data/README.md`에 획득 방법과 위치를 설명한다.
5. `README.md`에 설치 → 데이터 준비 → 노트북 실행 순서의 초안을 적는다.
6. 금지 라이브러리(OpenCV, Pillow, scikit-learn, NLTK, 자동 EDA 라이브러리)를 의존성에 추가하지 않는다.

## 산출물

- 디렉터리 구조와 의존성 파일
- 데이터 경로 규칙과 실행 명령 초안
- 빈 코드/노트북 파일 또는 실행 가능한 최소 뼈대

## 완료 기준

- [ ] 새 Python 환경에서 의존성을 설치할 수 있다.
- [ ] 문서의 절차로 데이터 경로를 준비하고 로드할 수 있다.
- [ ] 저장소에 개인 절대 경로나 이용 조건에 어긋나는 데이터가 없다.
