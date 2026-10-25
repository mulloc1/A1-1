# H&M 패션 데이터 분석

H&M Personalized Fashion Recommendations 데이터를 이용한 Python 분석 과제입니다. 재현 가능한 실행 환경, 전처리·이미지 피처·RFM 파이프라인과 실행 결과가 저장된 EDA·고객 세분화 노트북을 제공합니다.

## 1. 환경 설치

[uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치한 뒤 저장소 루트에서 실행합니다. Python 버전은 `.python-version`의 3.11을 사용합니다.

```sh
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.txt
```

위 설치 명령은 처음 환경을 만들 때 한 번만 실행하면 됩니다. 가상환경을 `activate`할 필요 없이 `--python .venv/bin/python`으로 설치 대상을 지정합니다. 이후에는 아래 JupyterLab 실행 명령만 사용하면 됩니다. `requirements.txt`가 바뀌거나 `.venv`를 삭제했을 때는 설치 명령을 다시 실행합니다.

직접 의존성은 `requirements.in`, 고정된 전체 의존성은 `requirements.txt`에 있습니다. 의존성을 변경할 때는 다음 명령으로 잠금 파일을 다시 생성합니다.

```sh
uv pip compile requirements.in --python-version 3.11 --output-file requirements.txt
```

Plotly는 선택 과제의 인터랙티브 차트에서만 사용합니다.

## 2. 원본 데이터 준비

[Kaggle 대회 데이터 페이지](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)에서 이용 조건을 확인하고 원본을 받아 다음 위치에 압축 해제합니다.

```text
data/h-and-m-personalized-fashion-recommendations/
├── articles.csv
├── customers.csv
├── transactions_train.csv
└── images/
```

노트북은 저장소 루트를 현재 작업 디렉터리로 사용하는 상대 경로를 기준으로 합니다. `article_id`는 앞자리 0을 유지하도록 문자열로 읽습니다. 컬럼 계약과 표본 추출 원칙은 [data/README.md](data/README.md)에 있습니다.

원본은 [Kaggle 대회 규칙](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules)에 따라 비상업적·교육용으로 사용합니다. 원본 CSV와 이미지를 재배포하지 않으며, 원본 및 로컬에서 만든 가공 데이터는 Git에서 제외합니다.

## 3. 노트북 실행

저장소 루트에서 JupyterLab을 시작합니다.

```sh
.venv/bin/python -m jupyter lab
```

[`notebooks/analysis_report.ipynb`](notebooks/analysis_report.ipynb)를 열어 셀을 처음부터 순서대로 실행합니다. 노트북은 고객 10,000명과 이미지 상품 1,500개를 시드 42로 선택하고, 원본 거래를 청크로 읽어 EDA와 RFM 세분화를 재현합니다. 표·차트 출력은 노트북에도 저장돼 있어 원본 데이터 없이 집계 결과를 검토할 수 있습니다.

현재 파이프라인은 가공 데이터를 파일로 저장하지 않습니다. 차트 PNG는 Git에서 제외된 `outputs/figures/`에 로컬로 생성합니다.

## 4. RFM 고객 세분화와 인사이트 초안

분석 기준일은 **2020-09-23**입니다. 선택 고객 10,000명 중 유효한 구매 기록이 있는 **9,918명**을 세분화했고, 기록이 없는 82명은 RFM 계산에서 제외했습니다. Frequency는 주문 횟수가 아니라 **서로 다른 구매일 수**이며, 상품 구매 기록 행 수는 별도 지표입니다. Monetary는 원본 `price`를 합산한 값입니다. 가격의 통화·환산 기준과 반품·취소 상태를 알 수 없으므로 아래의 가격값 합계는 확정 매출액이 아닙니다.

R·F·M은 동점에 평균 순위를 주는 1~5점 규칙으로 계산합니다. 낮은 Recency와 높은 Frequency·Monetary가 높은 점수입니다. 규칙과 세그먼트별 고객 수·평균 RFM·가격값 합계 비율은 [분석 노트북](notebooks/analysis_report.ipynb)의 6절에서 확인할 수 있습니다. `Recent`는 최근 가입, `Dormant`는 실제 이탈 확정을 뜻하지 않습니다.

### 인사이트 1: VIP 구매 경험 유지

- **근거:** VIP는 **2,468명(구매 고객의 24.88%)**이고 원본 가격값 합계의 **66.07%**를 차지합니다. 평균 구매일 수는 **17.09일**입니다.
- **실행:** VIP에 맞는 재입고 알림이나 선호 상품 추천을 시험합니다. 반복 구매 유지에 도움이 될 것이라는 **가설**이며, 관찰된 효과는 아닙니다.
- **검증:** 알림 동의를 받은 고객을 무작위 대조군과 비교해 60일 내 재구매율과 기록 가격값 합계의 증가분을 확인합니다. 발송 비용과 반품 정보도 필요합니다.

### 인사이트 2: 이전 단골의 구매 공백 확인

- **근거:** `At risk`는 **1,027명(10.35%)**으로 평균 구매일 수가 **5.40일**이지만 마지막 구매 후 평균 **376.18일**이 지났습니다.
- **실행:** 이 그룹에 과거 구매 카테고리와 관련된 상품 알림을 시험합니다. 다시 구매할 가능성을 높일 수 있다는 **가설**입니다.
- **검증:** 알림 동의 여부와 현재 활동 가능 여부를 확인하고, 무작위 대조군 대비 60일 내 재구매율을 측정합니다. 오래 구매하지 않은 이유는 이 데이터만으로 알 수 없습니다.

### 인사이트 3: 최근 구매 고객의 다음 구매 유도

- **근거:** `Recent`는 **725명(7.31%)**이고 평균 구매일 수는 **1.41일**입니다. 이 그룹의 원본 가격값 합계 비율은 **1.45%**입니다.
- **실행:** 최근 구매 상품과 어울리는 상품 추천을 시험합니다. 다음 구매로 이어질 수 있다는 **가설**입니다.
- **검증:** 추천을 받지 않은 대조군과 60일 내 추가 구매일 발생률을 비교합니다. 가입일이 없어 이들이 신규 고객인지는 별도로 확인해야 합니다.

이 수치는 선택한 고객 표본의 관측 결과입니다. 세그먼트가 작은 경우 일부 고객의 큰 구매값에 평균이 민감할 수 있으며, 캠페인 효과는 위 검증 전까지 주장하지 않습니다.
