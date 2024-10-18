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
    """Set objective and num_class for multiclass in params"""
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
    def __init__(self, params: dict[str, Any] = {}, num_boost_round: int = 100):
        self.params = params
        self.num_boost_round = num_boost_round

    def fit(
        self,
        train_set: Dataset,
        init_model: str | Path | lgb.Booster | None = None,
        keep_training_booster: bool = False,
        callbacks: list[Callable] | None = None,
    ) -> None:
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
        ):
            return _predict

        if self.is_encode_needed:
            if len(self.label_encoder.classes_) > 2:
                class_index = np.argmax(_predict, axis=1)
                return self.label_encoder.inverse_transform(class_index)
            else:
                class_index = np.round(_predict).astype(int)
                return self.label_encoder.inverse_transform(class_index)
        return _predict

    def cv(
        self,
        train_set: lgb.Dataset,
        folds: Iterable[tuple[np.ndarray, np.ndarray]]
        | BaseCrossValidator
        | None = None,
        nfold: int = 5,
        stratified: bool = True,
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
        _params = param_update(params=self.params, train_set=train_set)
        return lgb.cv(
            params=_params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            folds=folds,
            nfold=nfold,
            stratified=stratified,
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
