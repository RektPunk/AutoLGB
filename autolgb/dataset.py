from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from autolgb.basic import (
    CategoricalFeatureConfiguration,
    FeatureNameConfiguration,
    GroupType,
    InitScoreType,
    LabelType,
    PositionType,
    Task,
    TrainDataType,
    WeightType,
)


@dataclass
class Dataset:
    data: TrainDataType
    label: LabelType | None = None
    reference: "Dataset" | None = None
    weight: WeightType | None = None
    group: GroupType | None = None
    init_score: InitScoreType | None = None
    feature_name: FeatureNameConfiguration = "auto"
    categorical_feature: CategoricalFeatureConfiguration = "auto"
    params: dict[str, Any] | None = None
    free_raw_data: bool = True
    position: PositionType | None = None

    def __post_init__(self):
        if self.reference is None:
            _task_str = type_of_target(y=self.label)
            _task = getattr(Task, _task_str)
            if _task is None:
                raise ValueError("Unsupported task.")
            self.task: Task = _task
        else:
            self.task = self.reference.task

        if self.label:
            if isinstance(self.label, pd.DataFrame):
                if self.label.shape[1] > 1:
                    raise ValueError("Dimension of label must 1")
                self.label = pd.Series(self.label)

            if self.reference is None:
                self.is_encode_needed = self.task in {Task.binary, Task.multiclass}
                if self.is_encode_needed:
                    self.label_encoder = LabelEncoder()
                    self.label = self.label_encoder.fit_transform(self.label)
            else:
                self.is_encode_needed = self.reference.is_encode_needed
                self.label_encoder = self.reference.label_encoder
                self.label = self.reference.label_encoder.transform(self.label)

    @property
    def dtrain(self) -> lgb.Dataset:
        return lgb.Dataset(
            data=self.data,
            label=self.label,
            reference=self.reference.dtrain if self.reference else None,
            weight=self.weight,
            group=self.group,
            init_score=self.init_score,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            params=self.params,
            free_raw_data=self.free_raw_data,
            position=self.position,
        )

    @property
    def dpredict(self) -> TrainDataType:
        return self.data

    @property
    def is_encode_needed(self) -> bool:
        return getattr(self, "is_encode_needed", False)
