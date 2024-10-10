from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from autolgb.basic import (
    CategoricalFeatureConfiguration,
    FeatureNameConfiguration,
    InitScoreType,
    LabelType,
    Task,
    TrainDataType,
    WeightType,
)


class Dataset:
    def __init__(
        self,
        data: TrainDataType,
        label: LabelType | None = None,
        weight: WeightType | None = None,
        init_score: InitScoreType | None = None,
        feature_name: FeatureNameConfiguration = "auto",
        categorical_feature: CategoricalFeatureConfiguration = "auto",
        params: dict[str, Any] | None = None,
        free_raw_data: bool = True,
    ) -> None:
        self.data = data
        self.label = label
        self.weight = weight
        self.init_score = init_score
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params = params
        self.free_raw_data = free_raw_data

        if label is not None:
            _task_str = type_of_target(y=label)
            _task = getattr(Task, _task_str)
            if _task is None:
                raise ValueError("Unsupported task.")

            self.task: Task = _task
            if isinstance(label, pd.DataFrame):
                if label.shape[1] > 1:
                    raise ValueError("Dimension of label must 1")
                self.label = pd.Series(label)

            self._is_encode_needed = self.task in {Task.binary, Task.multiclass}
            if self.is_encode_needed:
                self._label_encoder = LabelEncoder()
                self.label = self._label_encoder.fit_transform(self.label)

    @property
    def dtrain(self) -> lgb.Dataset:
        return lgb.Dataset(
            data=self.data,
            label=self.label,
            weight=self.weight,
            init_score=self.init_score,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            params=self.params,
            free_raw_data=self.free_raw_data,
        )

    @property
    def dpredict(self) -> TrainDataType:
        return self.data

    @property
    def is_encode_needed(self) -> bool:
        return getattr(self, "_is_encode_needed", False)

    @property
    def label_encoder(self) -> LabelEncoder:
        if not self.is_encode_needed:
            raise ValueError("No label encoder exists.")
        return self._label_encoder
