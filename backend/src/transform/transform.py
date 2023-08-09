"""
Программа: предобработка данных
тренеровочных и тестовых данных
Версия: 1.0
"""

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


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


def drop_duplicates_df(data: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Удаляет дубликаты из указанного датафрейма
    :param data: исходный датафрейм
    :return: датафрейм
    """

    data.drop_duplicates(inplace=True, subset=subset)
    return data


def drop_column(data: pd.DataFrame, column_name: list) -> pd.DataFrame:
    """
    Удаляет указанные столбецы из датафрейма.
    :param data: исходный датафрейм
    :param column_name: имя столбца, который нужно удалить
    :return: датафрейм с удаленным столбцом
    """
    for i in column_name:
        if i in data.columns:
            data = data.drop([i], axis=1)
        else:
            continue
    return data


def rename_columns(data: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Переименовывает столбцы указанного датафрейма согласно заданному отображению.
    :param data: исходный датафрейм
    :param column_mapping: словарь с отображением старых и новых имен столбцов
    :return: датафрейм
    """
    data = data.rename(columns=column_mapping)
    return data


def add_column_from_mapping(
    data: pd.DataFrame, column: str, mapping_column: str
) -> pd.DataFrame:
    """
    Добавляет новый признак с количеством уникальных значений в указанный датафрейм,
    используя значения из другого столбца.
    :param data: исходный датафрейм
    :param column: имя нового столбца, который будет добавлен
    :param mapping_column:имя столбца, из которого будут взяты значения для отображения
    :return: датафрейм с добавленным новым столбцом
    """
    data[column] = data[mapping_column].map(data[mapping_column].value_counts())
    return data


def add_column_limit(
    data: pd.DataFrame, column_name: str, new_column_name: str, threshold: int = 2
) -> pd.DataFrame:
    """
    Добавляет новый столбец в указанный датафрейм, который указывает,
    превышает ли значение столбца заданный порог.
    :param data: исходный датафрейм
    :param column_name: имя столбца 'author_count'
    :param threshold: пороговое значение (по умолчанию 2)
    :return: датафрейм с добавленным столбцом 'author_more'
    """

    data[new_column_name] = data[column_name].apply(lambda x: 1 if x > threshold else 0)
    return data


def get_floor_position(data: pd.Series) -> int:
    """
    Определяет позицию этажа
    0 - среднии этажи
    1 - первый этаж
    2 - последний этаж
    :param data: датафрейм
    :return: бинаризованные значения
    """
    if data["floor"] == 1:
        return 1
    elif data["floor"] == data["floors_count"]:
        return 2
    else:
        return 0


def get_house_category(data: pd.Series) -> int:
    """
    Эта функция подразделяет дома а категории в зависимости от количества этажей.
    По этажности жилые дома подразделяют на:
    - малоэтажные (1 - 2 этажа)
    - средней этажности (3 - 5 этажей)
    - многоэтажные (6-10)
    - повышенной этажности (11 - 16 этажей)
    - высотные (16-50 этажей)
    - очень высотные (более 50 этажей)
    :param data: датафрейм
    :return: бинаризованные значения
    """

    if data <= 2:
        return 1
    elif 2 < data <= 5:
        return 2
    elif 5 < data <= 10:
        return 3
    elif 10 < data <= 16:
        return 4
    elif 16 < data <= 50:
        return 5
    elif data > 50:
        return 6


def remove_correlated_features(
    data: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """
    Удаляет признаки из указанного датафрейма, у которых корреляция превышает заданный порог.
    :param data: исходный датафрейм
    :param threshold: пороговое значение корреляции (по умолчанию 0.9)
    :return: датафрейм с удаленными признаками с высокой корреляцией
    """
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(
        pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    data.drop(high_corr_features, axis=1, inplace=True)

    return data


def replace_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Изменяет некорректные значения
    :param data: исходный датафрейм
    :return: исправленный датафрейм
    """
    data["district"] = data["district"].apply(lambda x: None if "." in str(x) else x)
    columns_to_fill = [
        "district",
        "street",
        "underground",
        "residential_complex",
        "author",
    ]
    for col in columns_to_fill:
        data[col] = data[col].fillna("не указано")
    return data


def fill_na_values(data: pd.DataFrame, fill_na_val: dict) -> pd.DataFrame:
    """
    Заполнение пропусков заданными значениями
    :param data: датафрейм
    :param fill_na_val: словарь с названиями признаков и значением, которым нужно заполнить пропуки
    :return: датафрейм
    """
    return data.fillna(fill_na_val)


def replace_dot(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Заменяет запятые на точки в указанных столбцах датафрейма.
    :param data: исходный датафрейм
    :param columns: список имен столбцов, в которых нужно заменить запятые на точки
    :return: pd.DataFrame, датафрейм с выполненной заменой
    """
    for column in columns:
        data[column] = data[column].apply(lambda x: x.replace(",", "."))
    return data


def transform_types(data: pd.DataFrame, change_col_types: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return: датафрейм
    """
    return data.astype(change_col_types, errors="raise")


def population_density(data, population_column, square_column, new_column_name):
    """
    Вычисляет плотность населения для каждой записи в датафрейме и добавляет
    новый столбец с результатами.
    :param data: исходный датафрейм
    :param population_column: имя столбца с населением
    :param square_column: имя столбца с площадью
    :param new_column_name: имя нового столбца
    :return: датафрейм
    """

    data[new_column_name] = data.apply(
        lambda x: round(x[population_column] / x[square_column], 2), axis=1
    )
    return data


def replace_single_value(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Заменяет уникальные значения столбца, которые встречаются только один раз,
    на строку 'None' внутри указанного датафрейма
    :param data: датафрейм
    :param column: имя столбца, в котором выполняется замена
    :return: датафрейм
    """
    counts = data[column].value_counts()
    for index, value in counts.items():
        if value == 1:
            data[column].replace({index: "None"}, inplace=True)
    return data


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Разные признаки"
    return data[column_sequence]


def save_unique_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def read_multiple_files(file_list: list[dict]) -> list:
    """
    Считывает несколько файлов CSV и возвращает список DataFrame для каждого файла.
    :param file_list : Список словарей, где каждый словарь содержит информацию о файле и его параметрах.
    :return: список DataFrame для каждого файла.
    """
    data_frames = []
    for file_info in file_list:
        file_path = file_info["path"]
        sep = file_info.get("sep", ",")
        encoding = file_info.get("encoding", "utf-8")
        data = pd.read_csv(file_path, sep=sep, encoding=encoding)
        data_frames.append(data)
    return data_frames


def df_merge(
    data: pd.DataFrame,
    data_list: list,
    columns: list,
    left_on_list: list,
    right_on_list: list,
    how: str,
) -> pd.DataFrame:
    """
    Объединяет таблицу `data` с несколькими таблицами из списка `data_list` по указанным столбцам.
    :param data: основная таблица
    :param data_list: список таблиц для объединения с основной таблицей
    :param columns: список столбцов из дополннительных таблиц, которые нужно объединить с основной
    :param left_on_list: список столбцов для объединения в `data` для каждой таблицы
    :param right_on_list: список столбцов для объединения в таблицах из `data_list` для каждой таблицы
    :param how: способ объединения
    :return: объединенную таблицу
    """

    for d, col, left, right in zip(data_list, columns, left_on_list, right_on_list):
        if col is None:
            col = list(d.columns)
        data = data.merge(
            d[col], left_on=left, right_on=right, how=how
        ).drop_duplicates()
    return data


def pipeline_preprocess(
    data: pd.DataFrame, flg_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :param rename_col: словарь с отображением старых и новых имен столбцов
    :param flg_evaluate: флаг для evaluate
    :return: датасет
    """
    data = drop_column(data, kwargs["drop_columns"])
    data = replace_data(data)
    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_data(
            data=data,
            drop_columns=kwargs["drop_columns"],
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )
    # добавляем дополнительные признаки к основной таблице из дополнительных
    underground_line, geo, eco, rating = read_multiple_files(kwargs["df_add"])
    df_add = [underground_line, eco, rating, geo]

    data = df_merge(
        data,
        df_add,
        kwargs["df_add_col"],
        kwargs["df_add_left"],
        kwargs["df_add_right"],
        kwargs["how"],
    )
    # удаляем дубликаты
    data = drop_duplicates_df(data)
    # удаляем столбцы после соединения таблиц
    data = drop_column(data, kwargs["drop_columns"])
    # переименовываем столбцы
    data = rename_columns(data, kwargs["rename_columns"])
    # добавляем новые признаки
    data = add_column_from_mapping(data, "line_count", "line")
    data = add_column_from_mapping(data, "author_count", "author")
    data = add_column_limit(data, "author_count", "author_more")
    data["floor_position"] = data.apply(lambda x: get_floor_position(x), axis=1)
    data["house_category"] = data["floors_count"].apply(lambda x: get_house_category(x))
    # удаляем коррелирующие признаки
    data = remove_correlated_features(data)
    # заполняем пропуски
    data = fill_na_values(data, kwargs["columns_fill_na"])
    # заменяем запятые на точки
    data = replace_dot(data, kwargs["dot_replace"])
    # изменяем типы данных
    for i, v in kwargs["change_col_types"].items():
        if i in data.columns:
            data = transform_types(data, {i: v})
        else:
            continue
    # добавляем новый признак
    data = population_density(data, "population", "square", "population_density")
    # изменяем некорректно заполненый параметр
    data = replace_single_value(data, "district")
    return data
