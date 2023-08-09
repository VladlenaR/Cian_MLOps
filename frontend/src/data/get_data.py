"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

import io
from io import BytesIO
from typing import Dict, Tuple

import pandas as pd
import streamlit as st


def get_dataset(path: str, sep: str = None, encoding: str = None) -> pd.DataFrame:
    """
    Читает файл CSV и возвращает его содержимое в виде датафрейма.
    :param path: путь к папке, содержащей файлы CSV
    :param sep: опциональный разделитель столбцов
    :param encoding: опциональная кодировка файла
    :return: датафрейм
    """
    data = pd.read_csv(path, sep=sep, encoding=encoding)
    return data


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_csv(data, sep=";")
    st.write("Dataset load")
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_csv(dataset_bytes_obj, index=False, sep=";")
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files
