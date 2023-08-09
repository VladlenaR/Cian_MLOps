"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import json
import os

import joblib
import requests
import streamlit as st
from optuna.visualization import plot_optimization_history, plot_param_importances


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"MAE": 0, "MSE": 0, "R2 adjusted": 0, "MAPE_%": 0, "WAPE_%": 0}

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    mae, mse, r2, mape, wape = st.columns(5)
    mae.metric(
        "MAE",
        new_metrics["MAE"],
        f"{new_metrics['MAE']-old_metrics['MAE']:.3f}",
    )
    mse.metric(
        "MSE",
        new_metrics["MSE"],
        f"{new_metrics['MSE']-old_metrics['MSE']:.3f}",
    )
    r2.metric(
        "R2 adjusted",
        new_metrics["R2 adjusted"],
        f"{new_metrics['R2 adjusted']-old_metrics['R2 adjusted']:.3f}",
    )
    mape.metric(
        "MAPE_%",
        new_metrics["MAPE_%"],
        f"{new_metrics['MAPE_%']-old_metrics['MAPE_%']:.3f}",
    )
    wape.metric(
        "WAPE_%",
        new_metrics["WAPE_%"],
        f"{new_metrics['WAPE_%']-old_metrics['WAPE_%']:.3f}",
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
