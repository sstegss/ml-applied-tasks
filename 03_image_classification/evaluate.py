"""Оценка моделей Keras Applications на картинках Imagenette.

Большая часть Imagenette (и train, и val) взята из обучающей выборки ImageNet,
на которой учились модели. Поэтому по умолчанию оцениваем только на 500 файлах
ILSVRC2012_val_*: это вся валидационная часть ImageNet для 10 классов Imagenette.

Метрики: top-1 и top-5 accuracy по всем 1000 классам ImageNet,
скорость: картинок в секунду батчами и время на одну картинку.
"""

import argparse
import json
import random
import time
from pathlib import Path

import keras
import numpy as np

from classify import MODELS, get_model, load_image, predict_proba

DATA_DIR = Path(__file__).parent / "data" / "imagenette2-320"


# Тот же файл, что использует decode_predictions: {"0": ["n01440764", "tench"], ...}
CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"


def load_class_index() -> list[tuple[str, str]]:
    """Список (WNID, название) для 1000 классов ImageNet по порядку выходов модели."""
    path = keras.utils.get_file("imagenet_class_index.json", CLASS_INDEX_URL, cache_subdir="models")
    with open(path) as f:
        index = json.load(f)
    return [tuple(index[str(i)]) for i in range(1000)]


def wnid_to_index() -> dict[str, int]:
    """Номер класса ImageNet по WNID: папки Imagenette названы WNID-ами."""
    return {wnid: i for i, (wnid, _) in enumerate(load_class_index())}


def load_samples(data_dir: Path, subset: str, per_class: int | None) -> list[tuple[Path, str]]:
    """unseen: картинки из validation ImageNet (лежат в train/ и val/ Imagenette).
    seen: картинки из train ImageNet в val/ Imagenette, модель видела их при обучении."""
    samples = []
    for class_dir in sorted((data_dir / "val").iterdir()):
        wnid = class_dir.name
        if subset == "unseen":
            files = sorted(data_dir.glob(f"*/{wnid}/ILSVRC2012_val_*.JPEG"))
        else:
            files = sorted(class_dir.glob(f"{wnid}_*.JPEG"))
        samples += [(f, wnid) for f in files[:per_class]]
    return samples


def evaluate(model_name: str, samples: list[tuple[Path, str]], batch_size: int) -> dict:
    size = MODELS[model_name][2]
    index = wnid_to_index()
    labels = np.array([index[wnid] for _, wnid in samples])
    get_model(model_name)  # загрузка модели не входит в замер

    # Одна картинка за раз: задержка ответа на запрос, включая чтение и предобработку
    single = random.Random(0).sample(samples, 30)
    predict_proba(load_image(single[0][0], size)[None], model_name)  # прогрев
    start = time.perf_counter()
    for path, _ in single:
        predict_proba(load_image(path, size)[None], model_name)
    ms_per_image = (time.perf_counter() - start) / len(single) * 1000

    top5 = []
    start = time.perf_counter()
    for i in range(0, len(samples), batch_size):
        batch = np.stack([load_image(p, size) for p, _ in samples[i:i + batch_size]])
        probs = predict_proba(batch, model_name)
        top5.append(np.argsort(probs, axis=1)[:, ::-1][:, :5])
    elapsed = time.perf_counter() - start
    top5 = np.concatenate(top5)

    hit1 = top5[:, 0] == labels
    hit5 = (top5 == labels[:, None]).any(axis=1)
    wnids = np.array([wnid for _, wnid in samples])
    return {
        "model": model_name,
        "params": get_model(model_name).count_params(),
        "top1": hit1.mean(),
        "top5": hit5.mean(),
        "per_class": {w: hit1[wnids == w].mean() for w in sorted(set(wnids))},
        "images_per_sec": len(samples) / elapsed,
        "ms_per_image": ms_per_image,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--subset", choices=["unseen", "seen"], default="unseen",
                        help="unseen: validation ImageNet (500 шт.), seen: train ImageNet из val/ Imagenette")
    parser.add_argument("--per-class", type=int, default=None, help="картинок на класс (по умолчанию все)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    samples = load_samples(args.data, args.subset, args.per_class)
    print(f"Imagenette, {args.subset}: {len(samples)} картинок, 10 классов\n")

    results = []
    for model_name in args.models:
        r = evaluate(model_name, samples, args.batch_size)
        results.append(r)
        print(f"{model_name}: top-1 {r['top1']:.3f}, top-5 {r['top5']:.3f}, "
              f"{r['images_per_sec']:.1f} карт/с батчами, {r['ms_per_image']:.0f} мс на одну", flush=True)

    print(f"\n{'Модель':<16}{'Параметров':>12}{'top-1':>8}{'top-5':>8}{'карт/с':>8}{'мс/карт':>9}")
    for r in results:
        print(f"{r['model']:<16}{r['params'] / 1e6:>11.1f}M{r['top1']:>8.3f}{r['top5']:>8.3f}"
              f"{r['images_per_sec']:>8.1f}{r['ms_per_image']:>9.0f}")

    names = dict(load_class_index())
    print(f"\nTop-1 по классам:\n{'Класс':<20}" + "".join(f"{r['model']:>16}" for r in results))
    for wnid in results[0]["per_class"]:
        print(f"{names[wnid]:<20}" + "".join(f"{r['per_class'][wnid]:>16.3f}" for r in results))
