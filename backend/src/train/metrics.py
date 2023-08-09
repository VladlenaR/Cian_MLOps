"""
Программа: Получение метрик
Версия: 1.0
"""
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_dict_metrics(y_test: pd.Series, y_predict: pd.Series) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :param y_probability: предсказанные вероятности
    :return: словарь с метриками
    """
    dict_metrics = {
        "MAE": round(mean_absolute_error(y_test, y_predict), 3),
        "MSE": round(mean_squared_error(y_test, y_predict), 3),
        "R2 adjusted": round(r2_score(y_test, y_predict), 3),
        "MAPE_%": round(np.mean(np.abs((y_predict - y_test) / y_test)) * 100, 3),
        "WAPE_%": round(np.sum(np.abs(y_predict - y_test)) / np.sum(y_test) * 100, 3),
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame,
    data_y: pd.Series,
    model: object,
    metric_path: str,
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(y_test=data_y, y_predict=model.predict(data_x))
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
