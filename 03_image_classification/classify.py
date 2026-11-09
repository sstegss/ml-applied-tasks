"""Классификация изображений по 1000 классам ImageNet моделями из Keras Applications."""

import argparse
import os
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # убрать служебные сообщения TF про CUDA

import keras
import numpy as np
from PIL import Image

# Имя модели -> (класс модели, функция предобработки, размер входа)
MODELS = {
    "efficientnetb0": (keras.applications.EfficientNetB0,
                       keras.applications.efficientnet.preprocess_input, 224),
    "resnet50": (keras.applications.ResNet50,
                 keras.applications.resnet50.preprocess_input, 224),
}
DEFAULT_MODEL = "efficientnetb0"


@lru_cache
def get_model(model_name: str = DEFAULT_MODEL):
    """Загрузить модель один раз и дальше брать из кэша."""
    return MODELS[model_name][0](weights="imagenet")


def load_image(path: str, size: int = 224) -> np.ndarray:
    """Уменьшить по короткой стороне до size / 0.875 и вырезать центр size×size, как при оценке на ImageNet."""
    img = Image.open(path).convert("RGB")
    scale = size / 0.875 / min(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype="float32")


def predict_proba(images: np.ndarray, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Вероятности 1000 классов для батча картинок формы (N, size, size, 3), значения 0–255."""
    _, preprocess, _ = MODELS[model_name]
    return get_model(model_name).predict_on_batch(preprocess(images.copy()))


def classify(path: str, model_name: str = DEFAULT_MODEL, top: int = 5) -> list[dict]:
    """Вернуть top самых вероятных классов: WNID, название и вероятность."""
    size = MODELS[model_name][2]
    probs = predict_proba(load_image(path, size)[None], model_name)
    decoded = keras.applications.imagenet_utils.decode_predictions(probs, top=top)[0]
    return [{"wnid": wnid, "label": label, "score": round(float(score), 4)}
            for wnid, label, score in decoded]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="картинки (jpg, png, ...)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    for path in args.files:
        print(path)
        for r in classify(path, args.model, args.top):
            print(f"  {r['score']:.3f}  {r['label']} ({r['wnid']})")
