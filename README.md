# H&M 패션 데이터 분석

H&M Personalized Fashion Recommendations 데이터를 이용한 Python 분석 과제입니다. 현재 단계에서는 재현 가능한 실행 환경과 데이터 접근용 노트북을 제공합니다. 전처리, 이미지 분석, RFM 및 시각화는 이후 단계에서 구현합니다.

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

`notebooks/analysis_report.ipynb`를 열어 환경 확인 셀을 실행합니다. 현재는 CSV 세 개와 이미지 폴더의 존재 및 CSV 앞부분 읽기만 확인합니다. 전체 거래 스캔, 이미지 배열 변환, 피처 생성은 이슈 3에서 진행합니다.

분석 중 생성하는 가공 데이터는 `data/processed/`에 로컬로 저장합니다. 그림을 저장할 위치는 `outputs/figures/`입니다.
