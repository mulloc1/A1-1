"""H&M 원본을 보존하며 표본, 이미지 피처, RFM을 계산한다."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from matplotlib.image import imread


class DataAnalyzer:
    """원본 표본(raw_*)과 분석용 표를 분리해 관리하는 H&M 파이프라인."""

    ARTICLE_COLUMNS = {"article_id", "prod_name", "product_group_name", "detail_desc"}
    CUSTOMER_COLUMNS = {"customer_id", "club_member_status", "age"}
    TRANSACTION_COLUMNS = {"t_dat", "customer_id", "article_id", "price", "sales_channel_id"}

    # 원본 위치와 표본 크기, 이미지 크기 등을 정하고 결과를 담을 공간을 준비한다.
    def __init__(
        self,
        data_dir: str | Path = "data/h-and-m-personalized-fashion-recommendations",
        *,
        customer_count: int = 10_000,
        image_count: int = 1_500,
        seed: int = 42,
        chunk_size: int = 500_000,
        crop_shape: tuple[int, int] = (1_728, 704),
        downsample: tuple[int, int] = (27, 11),
        image_reader: Callable[[str | Path], np.ndarray] = imread,
    ) -> None:
        if customer_count <= 0 or image_count < 0 or chunk_size <= 0:
            raise ValueError("customer_count와 chunk_size는 양수, image_count는 0 이상이어야 합니다.")
        if any(value <= 0 for value in (*crop_shape, *downsample)):
            raise ValueError("crop_shape와 downsample의 각 값은 양수여야 합니다.")
        if any(size % step for size, step in zip(crop_shape, downsample)):
            raise ValueError("crop_shape는 downsample 간격으로 나누어떨어져야 합니다.")
        self.data_dir = Path(data_dir)
        self.customer_count = customer_count
        self.image_count = image_count
        self.seed = seed
        self.chunk_size = chunk_size
        self.crop_shape = crop_shape
        self.downsample = downsample
        self.image_reader = image_reader
        # raw_*는 원본에서 읽은 표본이고, 나머지는 가공한 결과다.
        self.raw_articles: pd.DataFrame | None = None
        self.raw_customers: pd.DataFrame | None = None
        self.raw_transactions: pd.DataFrame | None = None
        self.customers: pd.DataFrame | None = None
        self.transactions: pd.DataFrame | None = None
        self.products: pd.DataFrame | None = None
        self.image_features: pd.DataFrame | None = None
        self.image_arrays: np.ndarray | None = None
        self.rfm: pd.DataFrame | None = None
        self.reference_date: pd.Timestamp | None = None
        self.quality_report: dict[str, object] = {}

    # CSV를 본격적으로 읽기 전에 헤더만 확인해 필요한 컬럼이 있는지 검사한다.
    @staticmethod
    def _require_columns(path: Path, required: set[str]) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"원본 CSV가 없습니다: {path}")
        missing = sorted(required - set(pd.read_csv(path, nrows=0).columns))
        if missing:
            raise ValueError(f"{path.name}의 필수 컬럼이 없습니다: {missing}")

    # 전체 고객에서 고정된 수만큼 뽑고, 선택된 고객의 정보만 다시 읽는다.
    def _sample_customers(self, path: Path) -> tuple[set[str], pd.DataFrame]:
        ids = pd.read_csv(path, usecols=["customer_id"], dtype={"customer_id": "string"})["customer_id"]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("customers.csv의 customer_id에 결측 또는 중복이 있습니다.")
        if len(ids) < self.customer_count:
            raise ValueError(f"요청한 고객 {self.customer_count}명보다 원본 고객이 적습니다.")
        # ID를 정렬한 뒤 seed를 적용하면 같은 파일에서 매번 같은 고객을 뽑는다.
        ordered = np.sort(ids.to_numpy(dtype=str))
        chosen = np.random.default_rng(self.seed).choice(
            ordered, size=self.customer_count, replace=False
        )
        chosen_ids = set(chosen.tolist())
        # 전체 고객 정보는 모두 보관하지 않고, 선택된 행만 모은다.
        chunks = pd.read_csv(path, chunksize=self.chunk_size, dtype={"customer_id": "string"})
        customers = pd.concat(
            [chunk.loc[chunk["customer_id"].isin(chosen_ids)].copy() for chunk in chunks],
            ignore_index=True,
        ).sort_values("customer_id", ignore_index=True)
        return chosen_ids, customers

    # 큰 거래 CSV를 조각별로 읽으며 선택 고객의 모든 거래와 전체 최종 거래일을 찾는다.
    def _sample_transactions(self, path: Path, ids: set[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
        selected_chunks: list[pd.DataFrame] = []
        last_date: pd.Timestamp | None = None
        chunks = pd.read_csv(
            path, chunksize=self.chunk_size,
            dtype={"customer_id": "string", "article_id": "string", "t_dat": "string"},
        )
        for chunk in chunks:
            # 기준일은 표본 고객만이 아니라 원본 전체의 마지막 유효 날짜를 따른다.
            if not chunk.empty:
                distinct_dates = chunk["t_dat"].dropna().unique()
                dates = pd.to_datetime(distinct_dates, format="%Y-%m-%d", errors="coerce")
                if len(dates):
                    latest = dates.max()
                    if pd.notna(latest) and (last_date is None or latest > last_date):
                        last_date = latest
            # 고객은 무작위로 뽑지만 그 고객의 거래 행은 기간 전체를 빠짐없이 모은다.
            selected = chunk.loc[chunk["customer_id"].isin(ids)]
            if not selected.empty:
                selected_chunks.append(selected.copy())
        if last_date is None:
            raise ValueError("transactions_train.csv에 거래 날짜가 없습니다.")
        # 마지막 거래 당일에 산 고객도 Recency가 1일이 되도록 다음 날을 쓴다.
        reference_date = last_date + pd.Timedelta(days=1)
        if not selected_chunks:
            raise ValueError("선택된 고객의 거래 기록이 없습니다.")
        return pd.concat(selected_chunks, ignore_index=True), reference_date

    # 원본을 읽고 검증한 뒤 고객·거래·상품 표와 이미지 피처, RFM 결과를 만든다.
    def load_data(self, *, include_images: bool = True) -> DataAnalyzer:
        """CSV를 읽어 고객 표본과 분석용 표, 선택적으로 이미지 피처를 만든다."""
        articles_path = self.data_dir / "articles.csv"
        customers_path = self.data_dir / "customers.csv"
        transactions_path = self.data_dir / "transactions_train.csv"
        for path, columns in (
            (articles_path, self.ARTICLE_COLUMNS),
            (customers_path, self.CUSTOMER_COLUMNS),
            (transactions_path, self.TRANSACTION_COLUMNS),
        ):
            self._require_columns(path, columns)
        # 상품 ID를 숫자로 읽으면 앞자리 0이 사라지므로 문자열로 고정한다.
        raw_articles = pd.read_csv(articles_path, dtype={"article_id": "string"})
        if raw_articles.empty or raw_articles["article_id"].isna().any():
            raise ValueError("articles.csv가 비었거나 article_id에 결측이 있습니다.")
        if raw_articles["article_id"].duplicated().any():
            raise ValueError("articles.csv의 article_id에 중복이 있습니다.")
        if not raw_articles["article_id"].str.fullmatch(r"\d{10}").all():
            raise ValueError("article_id는 앞자리 0을 포함한 10자리 문자열이어야 합니다.")

        ids, raw_customers = self._sample_customers(customers_path)
        raw_transactions, reference_date = self._sample_transactions(transactions_path, ids)
        # 분석에서 쓸 표준 컬럼명으로 바꾸되 raw_transactions는 그대로 둔다.
        transactions = raw_transactions.rename(columns={
            "t_dat": "order_date", "article_id": "product_id",
            "price": "amount", "sales_channel_id": "sales_channel",
        }).copy()
        # 날짜 오류와 0 이하 가격을 판별해 RFM에 사용할 유효 거래만 남긴다.
        transactions["order_date"] = pd.to_datetime(
            transactions["order_date"], format="%Y-%m-%d", errors="coerce"
        )
        transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
        invalid_date = transactions["order_date"].isna()
        invalid_amount = transactions["amount"].isna() | transactions["amount"].le(0)
        invalid_id = (
            transactions["customer_id"].isna()
            | transactions["product_id"].isna()
            | ~transactions["product_id"].str.fullmatch(r"\d{10}").fillna(False)
        )
        transactions = transactions.loc[~(invalid_date | invalid_amount | invalid_id)].copy()
        if transactions.empty:
            raise ValueError("선택된 고객의 유효한 거래 기록이 없습니다.")
        missing_products = ~transactions["product_id"].isin(raw_articles["article_id"])
        if missing_products.any():
            examples = transactions.loc[missing_products, "product_id"].drop_duplicates().head(5).tolist()
            raise ValueError(f"상품 표에 없는 article_id가 거래에 있습니다: {examples}")

        # 실제 표본 고객이 구매한 상품만 가져와 이름과 관찰 가격 피처를 붙인다.
        products = raw_articles.loc[
            raw_articles["article_id"].isin(transactions["product_id"]),
            ["article_id", "prod_name", "product_group_name", "detail_desc"],
        ].rename(columns={
            "article_id": "product_id", "prod_name": "product_name",
            "product_group_name": "category", "detail_desc": "description",
        }).copy()
        products["product_name_word_count"] = products["product_name"].fillna("").str.split().str.len()
        # reference_price는 정가가 아니라 표본 거래에서 관찰된 가격의 중앙값이다.
        prices = transactions.groupby("product_id", sort=False)["amount"].median()
        products["reference_price"] = products["product_id"].map(prices)
        products["image_path"] = products["product_id"].map(
            lambda product_id: self.data_dir / "images" / product_id[:3] / f"{product_id}.jpg"
        )
        products["image_path"] = products["image_path"].map(
            lambda path: path if path.is_file() else None
        )
        # 고객 나이는 회원 상태별 중앙값으로, 그 값도 없으면 전체 중앙값으로 채운다.
        customers, missing_report = self.handle_missing_values(
            raw_customers, "age", "club_member_status", strategy="median"
        )
        self.raw_articles = raw_articles
        self.raw_customers = raw_customers
        self.raw_transactions = raw_transactions
        self.customers = customers
        self.transactions = transactions.reset_index(drop=True)
        self.products = products.reset_index(drop=True)
        self.reference_date = reference_date
        self.image_features = None
        self.image_arrays = None
        # 제거된 거래와 대치 건수를 남겨 나중에 처리 전후를 확인할 수 있게 한다.
        self.quality_report = {
            "sampled_customers": len(raw_customers),
            "raw_transaction_rows": len(raw_transactions),
            "valid_transaction_rows": len(transactions),
            "invalid_date_rows": int(invalid_date.sum()),
            "invalid_amount_rows": int(invalid_amount.sum()),
            "invalid_id_rows": int(invalid_id.sum()),
            "age_imputation": missing_report,
            "image_errors": [],
        }
        if include_images and self.image_count:
            self.extract_image_features()
        self.calculate_rfm()
        return self

    # 숫자 컬럼의 빈 값을 같은 그룹의 통계값으로 채우고 처리 건수를 보고한다.
    @staticmethod
    def handle_missing_values(
        frame: pd.DataFrame, target: str, group: str, *,
        strategy: Literal["mean", "median"] = "median",
    ) -> tuple[pd.DataFrame, dict[str, int | str]]:
        """숫자 결측을 그룹 통계, 이어 전체 통계로 채운 복사본을 반환한다."""
        missing = [name for name in (target, group) if name not in frame.columns]
        if missing:
            raise KeyError(f"결측 처리 컬럼이 없습니다: {missing}")
        if strategy not in ("mean", "median"):
            raise ValueError("strategy는 'mean' 또는 'median'이어야 합니다.")
        if not pd.api.types.is_numeric_dtype(frame[target]):
            raise TypeError(f"{target}은 숫자형 컬럼이어야 합니다.")
        if frame.empty or frame[target].notna().sum() == 0:
            raise ValueError(f"{target}에 대치에 사용할 유효한 값이 없습니다.")
        # 복사본에만 값을 채우므로 호출자가 넘긴 원본 표는 바뀌지 않는다.
        result = frame.copy(deep=True)
        before = int(result[target].isna().sum())
        # transform은 각 행이 속한 그룹의 평균 또는 중앙값을 행마다 돌려준다.
        group_values = result.groupby(group, dropna=False)[target].transform(strategy)
        result[target] = result[target].fillna(group_values)
        after_group = int(result[target].isna().sum())
        # 그룹 전체가 비어 있으면 그룹 통계도 없으므로 전체 통계로 한 번 더 채운다.
        global_value = getattr(frame[target], strategy)()
        result[target] = result[target].fillna(global_value)
        after = int(result[target].isna().sum())
        report: dict[str, int | str] = {
            "target": target, "group": group, "strategy": strategy,
            "before_missing": before, "group_filled": before - after_group,
            "global_filled": after_group - after, "after_missing": after,
        }
        return result, report

    # IQR로 이상치의 범위를 계산하고 범위 밖의 행을 표시만 한다.
    @staticmethod
    def detect_outliers(
        frame: pd.DataFrame, column: str, *, iqr_multiplier: float = 1.5,
    ) -> tuple[pd.Series, dict[str, float | int]]:
        """IQR 경계 밖의 행을 표시한다. 결측값은 이상치로 간주하지 않는다."""
        if column not in frame.columns:
            raise KeyError(f"이상치 탐지 컬럼이 없습니다: {column}")
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column}은 숫자형 컬럼이어야 합니다.")
        if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier는 유한한 양수여야 합니다.")
        values = frame[column].dropna()
        if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column}에 유한한 숫자 값이 필요합니다.")
        # IQR은 3사분위수와 1사분위수의 차이다. 이 값으로 정상 범위를 정한다.
        q1, q3 = (float(values.quantile(q)) for q in (0.25, 0.75))
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        mask = ((frame[column] < lower) | (frame[column] > upper)).fillna(False)
        return mask, {
            "q1": q1, "q3": q3, "iqr": iqr, "lower": lower,
            "upper": upper, "outlier_count": int(mask.sum()),
        }

    # detect_outliers가 찾은 행을 제거하거나 경계값으로 바꾼 새 표를 만든다.
    @classmethod
    def treat_outliers(
        cls, frame: pd.DataFrame, column: str, *,
        method: Literal["drop", "clip"] = "clip", iqr_multiplier: float = 1.5,
    ) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
        """원본은 그대로 두고 이상치를 제거하거나 경계값으로 절단한다."""
        if method not in ("drop", "clip"):
            raise ValueError("method는 'drop' 또는 'clip'이어야 합니다.")
        mask, bounds = cls.detect_outliers(frame, column, iqr_multiplier=iqr_multiplier)
        # drop은 행 수가 줄고, clip은 행을 유지하며 값만 경계 안으로 옮긴다.
        result = frame.loc[~mask].copy(deep=True) if method == "drop" else frame.copy(deep=True)
        if method == "clip":
            result[column] = result[column].clip(bounds["lower"], bounds["upper"])
        report: dict[str, float | int | str] = {
            **bounds, "method": method, "before_rows": len(frame), "after_rows": len(result),
            "before_min": float(frame[column].min()), "before_max": float(frame[column].max()),
            "after_min": float(result[column].min()) if not result.empty else float("nan"),
            "after_max": float(result[column].max()) if not result.empty else float("nan"),
        }
        return result, report

    # 이미지 한 장을 읽어 크기가 달라도 같은 크기의 RGB 배열로 바꾼다.
    def _read_image(self, path: Path) -> np.ndarray:
        image = np.asarray(self.image_reader(path))
        crop_h, crop_w = self.crop_shape
        step_h, step_w = self.downsample
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"RGB 3채널 이미지가 아닙니다: {image.shape}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"비어 있는 이미지입니다: {image.shape}")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("픽셀에 유한한 숫자 값이 필요합니다.")
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
            image = image * 255
        if image.min() < 0 or image.max() > 255:
            raise ValueError("픽셀 값은 0~255 범위여야 합니다.")
        # 폭이 좁은 이미지도 있으므로 실제 크기 안에서 중앙 영역을 자른다.
        actual_h = min(image.shape[0], crop_h)
        actual_w = min(image.shape[1], crop_w)
        top = (image.shape[0] - actual_h) // 2
        left = (image.shape[1] - actual_w) // 2
        center = image[top : top + actual_h, left : left + actual_w, :]
        # NumPy 인덱스로 픽셀을 고르게 골라 모든 이미지를 설정된 동일 크기로 맞춘다.
        rows = np.linspace(0, actual_h - 1, crop_h // step_h).astype(int)
        cols = np.linspace(0, actual_w - 1, crop_w // step_w).astype(int)
        return np.asarray(center[rows[:, None], cols[None, :], :], dtype=np.uint8)

    # 선택된 상품 이미지들을 쌓고 이미지별 전체·RGB별 평균과 표준편차를 계산한다.
    def extract_image_features(self) -> pd.DataFrame:
        """이미지 배열을 쌓아 전체 및 RGB별 픽셀 통계를 벡터 연산으로 계산한다."""
        if self.products is None:
            raise RuntimeError("이미지 피처 계산 전에 load_data()를 실행하세요.")
        eligible = self.products.loc[
            self.products["image_path"].notna(), ["product_id", "image_path"]
        ].sort_values("product_id", ignore_index=True)
        if len(eligible) < self.image_count:
            raise ValueError(f"이미지가 있는 거래 상품 {len(eligible)}개는 요청한 {self.image_count}개보다 적습니다.")
        if self.image_count == 0:
            raise ValueError("image_count가 0이면 이미지 피처를 계산할 수 없습니다.")
        # 고객 표본과 마찬가지로 정렬 후 seed를 적용해 같은 이미지를 다시 뽑는다.
        positions = np.random.default_rng(self.seed).choice(
            len(eligible), size=self.image_count, replace=False
        )
        chosen = eligible.iloc[np.sort(positions)].reset_index(drop=True)
        arrays: list[np.ndarray] = []
        errors: list[dict[str, str]] = []
        # 파일 읽기는 반복하되, 픽셀 통계는 아래에서 묶음 배열에 한 번에 계산한다.
        for product_id, path in chosen.itertuples(index=False, name=None):
            try:
                arrays.append(self._read_image(path))
            except (OSError, ValueError, TypeError) as exc:
                errors.append({"product_id": str(product_id), "reason": str(exc)})
        self.quality_report["image_errors"] = errors
        if errors:
            raise ValueError(f"이미지 {len(errors)}개를 처리하지 못했습니다: {errors[:5]}")
        stacked = np.stack(arrays, axis=0)
        # 배열 축: (이미지 번호, 높이, 너비, RGB). 1~3축을 줄이면 이미지당 한 값이다.
        overall_mean = np.mean(stacked, axis=(1, 2, 3))
        overall_std = np.std(stacked, axis=(1, 2, 3))
        rgb_mean = np.mean(stacked, axis=(1, 2))
        rgb_std = np.std(stacked, axis=(1, 2))
        features = pd.DataFrame({
            "product_id": chosen["product_id"].to_numpy(),
            "image_mean": overall_mean, "image_std": overall_std,
            "image_r_mean": rgb_mean[:, 0], "image_g_mean": rgb_mean[:, 1],
            "image_b_mean": rgb_mean[:, 2], "image_r_std": rgb_std[:, 0],
            "image_g_std": rgb_std[:, 1], "image_b_std": rgb_std[:, 2],
        })
        # 계산된 숫자와 축소한 이미지 배열을 상품 ID에 맞춰 상품 표에 붙인다.
        products = self.products.drop(columns=[c for c in features if c != "product_id"], errors="ignore")
        products = products.merge(features, on="product_id", how="left", validate="one_to_one")
        products["image_array"] = None
        rows = products.set_index("product_id").index.get_indexer(chosen["product_id"])
        image_column = products.columns.get_loc("image_array")
        for row, array in zip(rows, stacked):
            products.iat[row, image_column] = array
        self.products = products
        self.image_features = features
        self.image_arrays = stacked
        self.quality_report["image_feature_rows"] = len(features)
        self.quality_report["image_array_shape"] = tuple(stacked.shape)
        return features.copy(deep=True)

    # 고객마다 마지막 구매일·고유 구매일 수·가격 합계를 RFM 지표로 계산한다.
    def calculate_rfm(self, *, reference_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """유효한 표본 거래에서 고유 구매일 기반 RFM을 계산한다."""
        if self.transactions is None or self.reference_date is None:
            raise RuntimeError("RFM 계산 전에 load_data()를 실행하세요.")
        date = pd.Timestamp(reference_date) if reference_date is not None else self.reference_date
        if pd.isna(date) or date <= self.transactions["order_date"].max():
            raise ValueError("RFM 기준일은 표본의 마지막 구매일보다 나중이어야 합니다.")
        # 같은 날 여러 상품을 산 기록은 금액에는 모두 더하고 빈도는 하루로 센다.
        summary = self.transactions.groupby("customer_id", sort=True).agg(
            last_purchase=("order_date", "max"),
            frequency_days=("order_date", "nunique"),
            monetary=("amount", "sum"),
            transaction_rows=("amount", "size"),
        )
        summary["recency_days"] = (date - summary["last_purchase"]).dt.days
        summary["reference_date"] = date
        self.rfm = summary.reset_index()[[
            "customer_id", "recency_days", "frequency_days", "monetary",
            "transaction_rows", "last_purchase", "reference_date",
        ]]
        return self.rfm.copy(deep=True)
