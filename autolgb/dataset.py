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
    """Dataset in AutoLGB. AutoLGB does not train on raw data as LightGBM.
    It discretizes continuous features into histogram bins, tries to combine categorical features, and automatically handles missing and infinite values.
    This class handles that preprocessing, and holds that alternative representation of the input data."""

    def __init__(
        self,
        data: TrainDataType,
        label: LabelType | None = None,
        weight: WeightType | None = None,
        init_score: InitScoreType | None = None,
        feature_name: FeatureNameConfiguration = "auto",
        categorical_feature: CategoricalFeatureConfiguration = "auto",
        params: dict[str, Any] = {"verbose": -1},
        free_raw_data: bool = True,
    ) -> None:
        """Initializes the Dataset object.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, Sequence, list of Sequence, list of numpy array
            Data source of Dataset.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM) or a LightGBM Dataset binary file.
        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow Array, pyarrow ChunkedArray or None, optional (default=None)
            Label of the data.
        weight : list, numpy 1-D array, pandas Series, pyarrow Array, pyarrow ChunkedArray or None, optional (default=None)
            Weight for each instance. Weights should be non-negative.
        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow Array, pyarrow ChunkedArray, pyarrow Table (for multi-class task) or None, optional (default=None)
            Init score for Dataset.
        feature_name : list of str, or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame or pyarrow Table, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        params : dict or None, optional (default=None)
            Other parameters for Dataset.
        free_raw_data : bool, optional (default=True)
            If True, raw data is freed after constructing inner Dataset.
        """
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
            _task: Task = getattr(Task, _task_str)
            if _task is None:
                raise ValueError("Unsupported task.")

            self.task = _task
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
        """Return the input data into a lgb.Dataset object for training."""
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
        """Returns the input data as-is for use in predictions."""
        return self.data

    @property
    def is_encode_needed(self) -> bool:
        """Checks if label encoding is needed for the given task (only for binary and multiclass tasks)."""
        return getattr(self, "_is_encode_needed", False)

    @property
    def label_encoder(self) -> LabelEncoder:
        """Returns the LabelEncoder object used for encoding target labels, if applicable."""
        if not self.is_encode_needed:
            raise ValueError("No label encoder exists.")
        return self._label_encoder
