from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from scipy.sparse import spmatrix
from sklearn.model_selection import BaseCrossValidator

from autolgb.basic import Objective, Task
from autolgb.dataset import Dataset


def param_update(params: dict[str, Any], train_set: Dataset) -> dict[str, Any]:
    """Set objective and num_class for multiclass in params."""
    if "objective" in params:
        return params
    _params = deepcopy(params)
    _task_objective_mapper: dict[Task, Objective] = {
        Task.binary: Objective.binary,
        Task.multiclass: Objective.multiclass,
        Task.continuous: Objective.regression,
    }
    _objective = _task_objective_mapper[train_set.task]
    _params["objective"] = _objective.value
    if train_set.task == Task.multiclass:
        _num_class = len(train_set.label_encoder.classes_)
        _params["num_class"] = _num_class

    if "verbosity" not in params:
        _params["verbosity"] = -1
    return _params


class Engine:
    """A wrapper for LightGBM training, prediction, cross-validation, and hyperparameter optimization.
    It provides a streamlined interface for training LightGBM models, making predictions,
    performing cross-validation, and optimizing hyperparameters using Optuna."""

    def __init__(self, params: dict[str, Any] = {}, num_boost_round: int = 100):
        """Initializes the Engine object.

        Parameters
        ----------
        params : dict
            Parameters for training.
            Values passed through ``params`` take precedence over those supplied via arguments.
        num_boost_round : int, optional (default=100)
            Number of boosting iterations.
        """
        self.params = params
        self.num_boost_round = num_boost_round

    def fit(
        self,
        train_set: Dataset,
        init_model: str | Path | lgb.Booster | None = None,
        keep_training_booster: bool = False,
        callbacks: list[Callable] | None = None,
    ) -> None:
        """Trains a LightGBM model on the given dataset.

        Parameters
        ----------
        train_set : Dataset
            Data to be trained on.
        init_model : str, pathlib.Path, Booster or None, optional (default=None)
            Filename of LightGBM model or Booster instance used for continue training.
        keep_training_booster : bool, optional (default=False)
            Whether the returned Booster will be used to keep training.
            If False, the returned value will be converted into _InnerPredictor before returning.
            This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
            When your model is very large and cause the memory error,
            you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
            You can still use _InnerPredictor as ``init_model`` for future continue training.
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        """
        self.is_encode_needed = train_set.is_encode_needed
        if self.is_encode_needed:
            self.label_encoder = train_set.label_encoder

        _params = param_update(params=self.params, train_set=train_set)
        self.booster = lgb.train(
            params=_params,
            train_set=train_set.dtrain,
            num_boost_round=self.num_boost_round,
            init_model=init_model,
            keep_training_booster=keep_training_booster,
            callbacks=callbacks,
        )
        self._is_fitted = True

    def predict(
        self,
        data: Dataset,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        """Makes predictions on the provided dataset.

        Parameters
        ----------
        data : Dataset
            Data source for prediction.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
            .. note::
                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        data_has_header : bool, optional (default=False)
            Whether the data has header.
            Used only if data is str.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.

        Returns
        -------
        result : numpy array, scipy.sparse.spmatrix or list of scipy.sparse.spmatrix
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).

        """
        self.__check_fitted()
        _predict = self.booster.predict(
            data=data.dpredict,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            data_has_header=data_has_header,
            validate_features=validate_features,
        )
        if (
            raw_score
            or pred_leaf
            or pred_contrib
            or isinstance(_predict, spmatrix | list)
            or not self.is_encode_needed
        ):
            return _predict

        class_index = (
            np.argmax(_predict, axis=1)
            if len(self.label_encoder.classes_) > 2
            else np.round(_predict).astype(int)
        )
        return self.label_encoder.inverse_transform(class_index)

    def cv(
        self,
        train_set: Dataset,
        folds: Iterable[tuple[np.ndarray, np.ndarray]]
        | BaseCrossValidator
        | None = None,
        nfold: int = 5,
        shuffle: bool = True,
        init_model: str | lgb.Path | lgb.Booster | None = None,
        fpreproc: Callable[
            [lgb.Dataset, lgb.Dataset, dict[str, Any]],
            tuple[lgb.Dataset, lgb.Dataset, dict[str, Any]],
        ]
        | None = None,
        seed: int = 0,
        callbacks: list[Callable] | None = None,
        eval_train_metric: bool = False,
        return_cvbooster: bool = False,
    ) -> dict[str, list[float] | lgb.CVBooster]:
        """Perform cross-validation on the provided training set.

        Parameters
        ----------
        train_set : Dataset
            Data to be trained on.
        folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
            If generator or iterator, it should yield the train and test indices for each fold.
            If object, it should be one of the scikit-learn splitter classes
            (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            and have ``split`` method.
            This argument has highest priority over other data split arguments.
        nfold : int, optional (default=5)
            Number of folds in CV.
        shuffle : bool, optional (default=True)
            Whether to shuffle before splitting data.
        init_model : str, pathlib.Path, Booster or None, optional (default=None)
            Filename of LightGBM model or Booster instance used for continue training.
        fpreproc : callable or None, optional (default=None)
            Preprocessing function that takes (dtrain, dtest, params)
            and returns transformed versions of those.
        seed : int, optional (default=0)
            Seed used to generate the folds (passed to numpy.random.seed).
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        eval_train_metric : bool, optional (default=False)
            Whether to display the train metric in progress.
            The score of the metric is calculated again after each training step, so there is some impact on performance.
        return_cvbooster : bool, optional (default=False)
            Whether to return Booster models trained on each fold through ``CVBooster``.

        Returns
        -------
        eval_results : dict
            History of evaluation results of each metric.
            The dictionary has the following format:
            {'valid metric1-mean': [values], 'valid metric1-stdv': [values],
            'valid metric2-mean': [values], 'valid metric2-stdv': [values],
            ...}.
            If ``return_cvbooster=True``, also returns trained boosters wrapped in a ``CVBooster`` object via ``cvbooster`` key.
            If ``eval_train_metric=True``, also returns the train metric history.
            In this case, the dictionary has the following format:
            {'train metric1-mean': [values], 'valid metric1-mean': [values],
            'train metric2-mean': [values], 'valid metric2-mean': [values],
            ...}.
        """
        _params = param_update(params=self.params, train_set=train_set)
        return lgb.cv(
            params=_params,
            train_set=train_set.dtrain,
            num_boost_round=self.num_boost_round,
            folds=folds,
            nfold=nfold,
            stratified=True
            if train_set.task in {Task.binary, Task.multiclass}
            else False,
            shuffle=shuffle,
            init_model=init_model,
            fpreproc=fpreproc,
            seed=seed,
            callbacks=callbacks,
            eval_train_metric=eval_train_metric,
            return_cvbooster=return_cvbooster,
        )

    def optimize(
        self,
        train_set: Dataset,
        ntrial: int = 10,
        folds: Iterable[tuple[np.ndarray, np.ndarray]]
        | BaseCrossValidator
        | None = None,
        nfold: int = 5,
        shuffle: bool = True,
        init_model: str | lgb.Path | lgb.Booster | None = None,
        fpreproc: Callable[
            [lgb.Dataset, lgb.Dataset, dict[str, Any]],
            tuple[lgb.Dataset, lgb.Dataset, dict[str, Any]],
        ]
        | None = None,
        seed: int = 0,
        callbacks: list[Callable] | None = None,
    ) -> None:
        """Optimizes the model's hyperparameters using Optuna and cross-validation.

        Parameters
        ----------
        train_set : Dataset
            Data to be trained on.
        ntrial : int, optional (default=10)
            Number of optimize trials.
        folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
            If generator or iterator, it should yield the train and test indices for each fold.
            If object, it should be one of the scikit-learn splitter classes
            (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            and have ``split`` method.
            This argument has highest priority over other data split arguments.
        nfold : int, optional (default=5)
            Number of folds in CV.
        shuffle : bool, optional (default=True)
            Whether to shuffle before splitting data.
        init_model : str, pathlib.Path, Booster or None, optional (default=None)
            Filename of LightGBM model or Booster instance used for continue training.
        fpreproc : callable or None, optional (default=None)
            Preprocessing function that takes (dtrain, dtest, params)
            and returns transformed versions of those.
        seed : int, optional (default=0)
            Seed used to generate the folds (passed to numpy.random.seed).
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        """
        _task_metric_mapper: dict[Task, str] = {
            Task.binary: "valid binary_logloss-mean",
            Task.multiclass: "valid multi_logloss-mean",
            Task.continuous: "valid l2-mean",
        }
        _metric_key = _task_metric_mapper[train_set.task]

        def _study_func(trial: optuna.Trial) -> float:
            _study_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 20.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 20.0),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "num_boost_round": trial.suggest_int("num_boost_round", 80, 120),
            }
            _params = param_update(params=_study_params, train_set=train_set)
            _num_boost_round = _params.pop("num_boost_round")
            _cv_results = lgb.cv(
                params=_params,
                train_set=train_set.dtrain,
                num_boost_round=_num_boost_round,
                folds=folds,
                nfold=nfold,
                stratified=True
                if train_set.task in {Task.binary, Task.multiclass}
                else False,
                shuffle=shuffle,
                init_model=init_model,
                fpreproc=fpreproc,
                seed=seed,
                callbacks=callbacks,
            )
            return min(_cv_results[_metric_key])

        study = optuna.create_study(direction="minimize")
        study.optimize(_study_func, n_trials=ntrial)

        _best_params = study.best_params
        self.params = param_update(params=_best_params, train_set=train_set)
        self.num_boost_round = self.params.pop("num_boost_round")

    def __check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotImplementedError("fit is not finished.")
