"""
Программа: Модель для предсказания цен на квартиры в Москве
Версия: 1.0
"""

import warnings

import optuna
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from src.evaluate.evaluate import pipeline_evaluate
from src.pipelines.pipeline import pipeline_training
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yaml"


class InsuranceCustomer(BaseModel):
    """
    Признаки для получения результатов модели
    """

    author: str
    floor: int
    floors_count: int
    rooms_count: int
    total_meters: float
    district: str
    street: str
    underground: str
    residential_complex: str


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(customer: InsuranceCustomer):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            customer.author,
            customer.floor,
            customer.floors_count,
            customer.rooms_count,
            customer.total_meters,
            customer.district,
            customer.street,
            customer.underground,
            customer.residential_complex,
        ]
    ]

    cols = [
        "author",
        "floor",
        "floors_count",
        "rooms_count",
        "total_meters",
        "district",
        "street",
        "underground",
        "residential_complex",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    result = f"Цена данной квартиры {round(predictions)} рублей"

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)
