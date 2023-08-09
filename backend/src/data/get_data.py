"""
Программа: получение данных из файла
Версия: 1.0
"""

import pandas as pd


def read_df(path: str, sep: str = None, encoding: str = None) -> pd.DataFrame:
    """
    Читает файл CSV и возвращает его содержимое в виде датафрейма.
    :param path: путь к папке, содержащей файлы CSV
    :param sep: опциональный разделитель столбцов
    :param encoding: опциональная кодировка файла
    :return: датафрейм
    """
    data = pd.read_csv(path, sep=sep, encoding=encoding)
    return data
