"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_sort_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    ascending: bool = False,
    limit: int = None,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    order: bool = False,
    figsize: tuple = (10, 7),
    rotation: bool = False,
) -> matplotlib.figure.Figure:
    """
    Строит график столбчатой диаграммы с сортировкой по заданному столбцу
    :param data: датафрейм
    :param x: параметр по x
    :param y:параметр по xy
    :param ascending: флаг сортировки
    :param limit: количество выводимых записей
    :param xlabel: название по x
    :param ylabel: название по y
    :param title: азвание графика
    :param order: флаг нужно ли делать сортировку
    :param figsize: размер графика
    :param rotation: флаг для поворота названий значений на 90 градусов
    :return: поле рисунка
    """
    fig = plt.figure(figsize=figsize)
    if order == True:
        ax = sns.barplot(
            x=x,
            y=y,
            data=data,
            order=data.sort_values(y, ascending=ascending)[x][:limit],
        )
    else:
        ax = sns.barplot(x=x, y=y, data=data)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if rotation == True:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig


def create_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    ylim: tuple = None,
    rotation: bool = False,
    order=None,
    figsize: tuple = (10, 7),
) -> matplotlib.figure.Figure:
    """
    Создает boxplot на основе данных.
    :param data: данные для создания диаграммы
    :param x: имя столбца для оси x
    :param y: имя столбца для оси y
    :param title: заголовок диаграммы
    :param xlabel: название оси x
    :param ylabel: название оси y
    :param ylim: пределы оси y в формате
    :param rotation: флаг для поворота названий значений на 90 градусов
    :param order: порядок районов на оси x (по умолчанию None)
    :param figsize: размеры фигуры (по умолчанию (13, 7))
    :return: поле рисунка
    """

    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(x=x, y=y, data=data, order=order)
    plt.title(title, fontsize=15, pad=10)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if ylim is not None:
        ax.set_ylim(ylim)
    if rotation is True:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return fig
