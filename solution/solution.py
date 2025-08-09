import numpy as np
from typing import List, Union

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Загружаем модель YOLOv8
# model = YOLO("best2.pt")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path='best.pt',
    confidence_threshold=0.33,
    device="cuda:0",  # or 'cuda:0'
)

def sahi_to_xywh(sahi_result):
    """
    Конвертирует SAHI ObjectPrediction сразу в формат [xc, yc, w, h, conf, cls_id]
    """
    boxes = []
    for pred in sahi_result.object_prediction_list:
        # Получаем координаты в формате SAHI
        xc, yc, w, h = pred.bbox.to_xywh()

        formatted = {
                'xc': xc / sahi_result.image_width,
                'yc': yc / sahi_result.image_height,
                'w': w / sahi_result.image_width,
                'h': h / sahi_result.image_height,
                'label': 0,
                'score': pred.score.value
            }
        boxes.append(formatted)

    return boxes


def infer_image_bbox(image: np.ndarray) -> List[dict]:
    """Функция для получения ограничивающих рамок объектов на изображении.

    Args:
        image (np.ndarray): Изображение, на котором будет производиться инференс.

    Returns:
        List[dict]: Список словарей с координатами ограничивающих рамок и оценками.
        Пример выходных данных:
        [
            {
                'xc': 0.5,
                'yc': 0.5,
                'w': 0.2,
                'h': 0.3,
                'label': 0,
                'score': 0.95
            },
            ...
        ]
    """

    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=1280,
        slice_width=1280,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    
    # result_numpy = []

    # Преобразуем результаты в numpy массивы
    # for res in result:
    #     result_numpy.append(res.cpu().numpy())
    
    # Если есть результаты, обрабатываем их
    res_list = []
    for pred in result.object_prediction_list:
        # Получаем координаты в формате SAHI
        x_min, y_min, w, h = pred.bbox.to_xywh()

        xc = x_min + w / 2
        yc = y_min + h / 2

        formatted = {
                'xc': xc / result.image_width,
                'yc': yc / result.image_height,
                'w': w / result.image_width,
                'h': h / result.image_height,
                'label': 0,
                'score': pred.score.value
            }
        res_list.append(formatted)

    return res_list


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Список изображений или одно изображение.

    Returns:
        List[List[dict]]: Список списков словарей с результатами предикта 
        на найденных изображениях.
        Пример выходных данных:
        [
            [
                {
                    'xc': 0.5,
                    'yc': 0.5,
                    'w': 0.2,
                    'h': 0.3,
                    'label': 0,
                    'score': 0.95
                },
                ...
            ],
            ...
        ]
    """    
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    # Обрабатываем каждое изображение из полученного списка
    for image in images:        
        image_results = infer_image_bbox(image)
        results.append(image_results)
    
    return results
