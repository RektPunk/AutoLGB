from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse
from lightgbm.basic import Sequence

TrainDataType = (
    np.ndarray
    | pd.DataFrame
    | scipy.sparse.spmatrix
    | Sequence
    | list[Sequence]
    | list[np.ndarray]
)
LabelType = list[float] | list[int] | np.ndarray | pd.Series | pd.DataFrame
WeightType = list[float] | list[int] | np.ndarray | pd.Series
PositionType = np.ndarray | pd.Series
InitScoreType = list[float] | list[list[float]] | np.ndarray | pd.Series | pd.DataFrame
GroupType = list[float] | list[int] | np.ndarray | pd.Series
CategoricalFeatureConfiguration = list[str] | list[int] | Literal["auto"]
FeatureNameConfiguration = list[str] | Literal["auto"]
