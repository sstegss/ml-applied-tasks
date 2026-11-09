"""Детекция объектов на видео моделью YOLOv5 из PyTorch Hub."""

import argparse
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

DEFAULT_MODEL = "yolov5s"


@lru_cache
def get_model(model_name: str = DEFAULT_MODEL):
    """Загрузить модель один раз и дальше брать из кэша.

    Веса кладём в папку torch.hub, иначе YOLOv5 скачивает их в текущую папку.
    """
    weights = Path(torch.hub.get_dir()) / f"{model_name}.pt"
    return torch.hub.load("ultralytics/yolov5", "custom", path=str(weights),
                          trust_repo=True, verbose=False)


def detect(frame: np.ndarray, model_name: str = DEFAULT_MODEL) -> list[dict]:
    """Найти объекты на кадре из OpenCV (BGR). Рамка задаётся как [x1, y1, x2, y2] в пикселях."""
    model = get_model(model_name)
    results = model(frame[..., ::-1])  # модель ждёт RGB
    return [{"label": model.names[int(cls)], "score": round(conf, 3), "box": [round(v) for v in box]}
            for *box, conf, cls in results.xyxy[0].tolist()]


def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{d['label']} {d['score']:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return frame


def process_video(src: str, dst: str | None = None, every: int = 1,
                  model_name: str = DEFAULT_MODEL) -> dict:
    """Пройти по видео, нарисовать рамки и при dst сохранить результат.

    every > 1: модель запускается на каждом every-м кадре, а на остальных
    остаются рамки с последнего обработанного кадра.
    """
    get_model(model_name)  # загрузка модели не входит в замер
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if dst:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    per_frame, detections = [], []
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if len(per_frame) % every == 0:
            detections = detect(frame, model_name)
        per_frame.append(detections)
        if writer:
            writer.write(draw(frame, detections))
    elapsed = time.perf_counter() - start
    cap.release()
    if writer:
        writer.release()

    return {"frames": len(per_frame), "video_fps": fps, "duration": len(per_frame) / fps,
            "processing_fps": len(per_frame) / elapsed, "detections": per_frame}


def class_stats(per_frame: list[list[dict]]) -> dict[str, dict]:
    """Сколько объектов каждого класса в среднем и максимум на одном кадре.

    Трекинга нет, поэтому один и тот же человек на 100 кадрах даёт 100 детекций,
    и «сколько всего разных людей прошло» отсюда не посчитать.
    """
    labels = sorted({d["label"] for dets in per_frame for d in dets})
    stats = {}
    for label in labels:
        counts = [sum(d["label"] == label for d in dets) for dets in per_frame]
        stats[label] = {"mean": sum(counts) / len(counts), "max": max(counts)}
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["mean"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="входное видео (mp4, avi, ...)")
    parser.add_argument("-o", "--output", help="куда сохранить видео с рамками "
                        "(по умолчанию outputs/<имя>_detected.mp4)")
    parser.add_argument("--every", type=int, default=1, help="запускать модель на каждом N-м кадре")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="yolov5n, yolov5s, yolov5m, ...")
    args = parser.parse_args()

    output = args.output or f"outputs/{Path(args.video).stem}_detected.mp4"
    r = process_video(args.video, output, args.every, args.model)
    print(f"{args.video}: {r['frames']} кадров ({r['duration']:.1f} с), "
          f"обработка {r['processing_fps']:.1f} кадров/с, сохранено в {output}")
    for label, s in class_stats(r["detections"]).items():
        print(f"  {label:<14} в среднем {s['mean']:.1f} на кадре, максимум {s['max']}")
