from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse
from lightgbm.basic import Sequence

TrainDataType = (
    str
    | Path
    | np.ndarray
    | pd.DataFrame
    | scipy.sparse.spmatrix
    | Sequence
    | list[Sequence]
    | list[np.ndarray]
)
LabelType = list[float] | list[int] | np.ndarray | pd.Series | pd.DataFrame
WeightType = list[float] | list[int] | np.ndarray | pd.Series
InitScoreType = list[float] | list[list[float]] | np.ndarray | pd.Series | pd.DataFrame
CategoricalFeatureConfiguration = list[str] | list[int] | Literal["auto"]
FeatureNameConfiguration = list[str] | Literal["auto"]


class Task(IntEnum):
    binary: int = 1
    multiclass: int = 2
    continuous: int = 3


class Objective(StrEnum):
    binary: str = "binary"
    multiclass: str = "multiclass"
    regression: str = "regression"
