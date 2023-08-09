"""
Программа: Тренировка данных
Версия: 1.0
"""

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from optuna import Study
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def objective_ctb(trial, X, y, N_FOLDS, random_state):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [300]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.2945510204081633]
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "l2_leaf_reg": trial.suggest_uniform("l2_leaf_reg", 1e-5, 1e2),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "random_state": random_state,
    }

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)

    cv_predicts = np.empty(N_FOLDS)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cat_feat = X.select_dtypes("category").columns.tolist()

        train_data = Pool(data=X_train, label=y_train, cat_features=cat_feat)

        model = CatBoostRegressor(**params, allow_writing_files=False)
        model.fit(
            train_data,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=100,
            verbose=0,
        )

        preds = model.predict(X_test)
        cv_predicts[idx] = mean_absolute_error(y_test, preds)

    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [CatBoostRegressor tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="Cat_01")
    function = lambda trial: objective_ctb(
        trial, x_train, y_train, kwargs["k_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
) -> CatBoostRegressor:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: CatBoostRegressor
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    cat_features = x_train.select_dtypes("category").columns.tolist()

    # training optimal params
    clf = CatBoostRegressor(
        **study.best_params,
        cat_features=cat_features,
        verbose=False,
        allow_writing_files=False,
    )
    clf.fit(x_train, y_train, verbose=0)

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=clf, metric_path=metric_path)
    return clf
