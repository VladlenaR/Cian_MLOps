"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import streamlit as st
import yaml
from src.data.get_data import get_dataset, load_data
from src.evaluate.evaluate import evaluate_from_file, evaluate_input
from src.plotting.charts import create_boxplot, get_sort_barplot
from src.preprocess.preprocess import pipeline_preprocess
from src.train.training import start_training

CONFIG_PATH = "../config/params.yaml"


def main_page():
    """
    Страница с описанием проекта
    """

    st.markdown("# Описание проекта")
    st.title("MLOps project:  Определение цен на квартиры в Москве")
    st.write(
        """
        Задача предсказания цен на квартиры в Москве - задача регрессии

        Она может быть важна для агентств недвижимости, инвесторов, застройщиков и 
        других участников рынка недвижимости. На основе предсказаний цен можно делать
        более точные прогнозы доходности инвестиций, планировать стратегии продаж и 
        закупки объектов недвижимости, а также определять ценообразование на рынке."""
    )

    st.markdown(
        """
        ### Описание основных полей 
            - author - автор объявления
            - floor - этаж, на котором находится квартира
            - floors_count - общее количество этажей в доме
            - rooms_count - количество комнат в квартире
            - total_meters - общая площадь
            - **price - стоимость квартиры (целевая переменная)**
            - district - район, в котором находится квартира
            - street - улица
            - underground - метро
            - residential_complex - название жилого комплекса
    """
    )

    st.markdown(
        """
        ### Описание дополнительных полей 
            - line - ветка, на которой находится метро
            - area - округ, в котором находится квартира
            - eco_rating - экологический рейтинг района, в котром находится квартира
            - insufficient_infrastructure - недостаточно инфраструктуры, %
            - convenient_for_life - удобность для жизни, %
            - very_convenient_for_life - очень комфортный для жизни, %
            - few_entertainment - недостаточно мест для досуга и развлечений, %
            - cultural - оценка культурных мест, %
            - entertainment - оценка развлекательных мест, %
            - cultural_entertainment - оценка культурно-развлекательных мест, %
            - residential_infrastructure_rating - рейтинг жилой инфраструктуры 
            - entertainment_infrastructure_rating - рейтинг развлекательной инфраструктуры 
            - square - площадь района
            - population - численность населения в районе
            - housing_fund_area - площадь жилфонда
            - line_count - количество пересадочных станций
            - author_count - количество объявлений у автора
            - author_more - флаг, который показывает что у автора больше двух объявлений
            - floor_position - позиция этажа (2 - последний этаж, 1 -первый этаж, 0 - середина)
            - house_category - категория дома в зависимости от количества этажей (1 - малоэтажные (1 - 2 этажа), 
            2 - средней этажности (3 - 5 этажей), 3 - многоэтажные (6-10), 4 - повышенной этажности (11 - 16 этажей), 
            5 - высотные (16-50 этажей), 6 - очень высотные (более 50 этажей))
            - population_density - плотность населения района
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis")

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    # load and write dataset
    data = get_dataset(path=config["preprocessing"]["train_path"], sep=";")
    data = pipeline_preprocess(data=data, flg_evaluate=False, **preprocessing_config)
    st.write(data.head())

    # plotting with checkbox
    district_price = st.sidebar.checkbox("Цена в зависимости от района")
    cat_home_price = st.sidebar.checkbox(
        "Цена в зависимости от категории этажности дома"
    )
    floor_price = st.sidebar.checkbox("Цена в зависимости от этажа")
    floor_position_price = st.sidebar.checkbox("Цена в зависимости от положения этажа")
    population_density_price = st.sidebar.checkbox(
        "Цена в зависимости от плотности населения"
    )
    line_price = st.sidebar.checkbox("Цена в зависомости от линии метро")

    if district_price:
        st.pyplot(
            create_boxplot(
                data=data,
                x="district",
                y="price",
                order=data["district"].unique()[:30],
                title="Цена в зависимости от района",
                xlabel="Район",
                ylabel="Цена за квартиру",
                ylim=(0, 400000000),
                figsize=(13, 7),
                rotation=True,
            )
        )
        st.markdown("""- в центре квартиры дороже всего""")
    if floor_price:
        st.pyplot(
            get_sort_barplot(
                data=data,
                x="floor",
                y="price",
                xlabel="Этаж",
                ylabel="Цена",
                title="Цена в зависимости от этажа",
                figsize=(20, 9),
            )
        )
        st.markdown(
            """
        - чем ниже этаж, тем больше количество квартир
        - квартиры на очень высоких этажах встречаются редко, возможно поэтому цена на них дороже
        - в среднем самые дорогие квартиры на 7 этаже
        - в среднем самые дешёвые квартиры на 32 этаже, возможно это связано с тем, что это последний этаж в некоторых домах
        - в высоких домах (скорее всего новостройки) до 40 этажей, чем выше этаж, тем ниже цена. Возможно, это связано с тем что людям некомфортно жить так высоко"""
        )
    if cat_home_price:
        st.markdown(
            """
        Дома разделаются в зависимости от количества этажей на категории:
        - 1 - малоэтажные (1 - 2 этажа)
        - 2 - средней этажности (3 - 5 этажей)
        - 3 - многоэтажные (6-10)
        - 4 - повышенной этажности (11 - 16 этажей)
        - 5 - высотные (16-50 этажей)
        - 6 - очень высотные (более 50 этажей)"""
        )
        st.pyplot(
            create_boxplot(
                data=data,
                x="house_category",
                y="price",
                title="Цена в зависимости от категории этажности дома",
                xlabel="Категория дома",
                ylabel="Цена",
                ylim=(0, 500000000),
            )
        )
        st.markdown(
            """
        - самые дорогие квартиры в домах, в которых 6-10 этажей
        - в очень высотных домах квартиры по цене находятся на втором месте. В таких домах начальная цена квартир самая дорогая  
        - в остальных домах средняя цена примерно одинакова, различие только в разбросе цен, возможно она связана с тем на каком этаже находится квартира
        - больше всего квартир в домах от 16 до 40 этажей, возможно в связи экономии места"""
        )
    if floor_position_price:
        st.write(
            """
        - 0 - середина,
        - 1 -первый этаж,
        - 2 - последний этаж"""
        )
        st.pyplot(
            create_boxplot(
                data=data,
                x="floor_position",
                y="price",
                title="Цена в зависимости от положения этажа",
                xlabel="Положение этажа",
                ylabel="Цена",
                ylim=(0, 500000000),
            )
        )
        st.markdown(
            """
        - квартиры на первом этаже самые дешёвые
        - средняя цена квартир на последнем этаже примерно такая же, как и на других этажах, но имеется хвост в сторону увеличения цены, т к в некоторых домах последний этаж это пентхаус
        - за счёт того, что обычные квартиры на последнем этаже меньше ценятся, начальная цена меньше чем на других этажах"""
        )
    if population_density_price:
        st.pyplot(
            get_sort_barplot(
                data=data,
                x="population_density",
                y="price",
                xlabel="Плотность населения",
                ylabel="Цена",
                title="Цена в зависимости от плотности населения",
                figsize=(20, 9),
                rotation=True,
            )
        )
        st.markdown(
            """
        - зависимости между ценой и плотностью населения нет"""
        )
    if line_price:
        st.pyplot(
            create_boxplot(
                data=data,
                x="line",
                y="price",
                title="Цена в зависомости от линии метро",
                xlabel="Линия метро",
                ylabel="Цена",
                ylim=(0, 1000000000),
                rotation=True,
                figsize=(17, 7),
            )
        )
        st.markdown(
            """
        - линия метро влияет на цену квартиры
        - на Кольцевой линии самый большой порог цены и самая высокая средняя цена
        - на Калининской линии самый большой разброс цен, потому что она идёт от центра до края Москвы
        - самая дешёвые квартиры на Некрасовской линии"""
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model CatBoost")
    # get params
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
