"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO

import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для вводы данных, используем уникальные значения
    author = st.sidebar.selectbox("Автор", (unique_df["author"]))
    floor = st.sidebar.slider(
        "Этаж", min_value=min(unique_df["floor"]), max_value=max(unique_df["floor"])
    )
    floors_count = st.sidebar.slider(
        "Кол-во этажей",
        min_value=min(unique_df["floors_count"]),
        max_value=max(unique_df["floors_count"]),
    )
    rooms_count = st.sidebar.slider(
        "Кол-во комнат",
        min_value=min(unique_df["rooms_count"]),
        max_value=max(unique_df["rooms_count"]),
    )
    total_meters = st.sidebar.number_input(
        "Площадь",
        min_value=min(unique_df["total_meters"]),
        max_value=max(unique_df["total_meters"]),
    )
    district = st.sidebar.selectbox("Район", (unique_df["district"]))
    street = st.sidebar.selectbox("Улица", (unique_df["street"]))
    underground = st.sidebar.selectbox("Метро", (unique_df["underground"]))
    residential_complex = st.sidebar.selectbox(
        "Название комплекса", (unique_df["residential_complex"])
    )

    dict_data = {
        "author": author,
        "floor": floor,
        "floors_count": floors_count,
        "rooms_count": rooms_count,
        "total_meters": total_meters,
        "district": district,
        "street": street,
        "underground": underground,
        "residential_complex": residential_complex,
    }

    st.write(
        f"""### Данные квартиры:\n
    1) Автор: {dict_data['author']}
    2) Этаж: {dict_data['floor']}
    3) Кол-во этажей: {dict_data['floors_count']}
    4) Кол-во комнат: {dict_data['rooms_count']}
    5) Площадь: {dict_data['total_meters']}
    6) Район: {dict_data['district']}
    7) Улица: {dict_data['street']}
    8) Метро: {dict_data['underground']}
    9) Название комплекса: {dict_data['residential_complex']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        if dict_data["floor"] > dict_data["floors_count"]:
            st.error("Выбранный этаж не может превышать общее количество этажей в доме")
        else:
            result = requests.post(endpoint, timeout=8000, json=dict_data)
            json_str = json.dumps(result.json())
            output = json.loads(json_str)
            st.write(f"## {output}")
            st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())
