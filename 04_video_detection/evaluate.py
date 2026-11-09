"""Оценка YOLOv5 на нескольких видео.

Разметки у роликов нет, поэтому mAP берём из документации YOLOv5 (COCO val2017),
а сами меряем скорость на CPU и что модель находит. Для режима «модель на каждом
N-м кадре» сравниваем рамки с обработкой каждого кадра: какая доля совпала.
"""

import argparse
from pathlib import Path

from detect import DEFAULT_MODEL, class_stats, process_video

DATA_DIR = Path(__file__).parent / "data"
VIDEOS = ["crossroad.mp4", "crosswalk.mp4", "bangkok.mp4", "car-detection.mp4"]

# Из README ultralytics/yolov5, размер входа 640, COCO val2017
COCO_METRICS = {
    "yolov5n": {"params": 1.9, "map50_95": 28.0, "map50": 45.7},
    "yolov5s": {"params": 7.2, "map50_95": 37.4, "map50": 56.8},
    "yolov5m": {"params": 21.2, "map50_95": 45.4, "map50": 64.1},
}


def iou(a: list[int], b: list[int]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def agreement(reference: list[list[dict]], other: list[list[dict]], threshold: float = 0.5) -> float:
    """F1 совпадения рамок с эталоном: рамки совпали, если класс тот же и IoU >= threshold."""
    tp = fp = fn = 0
    for ref, dets in zip(reference, other):
        unmatched = list(ref)
        for d in dets:
            best = max(unmatched, key=lambda r: iou(r["box"], d["box"]) if r["label"] == d["label"] else -1,
                       default=None)
            if best and best["label"] == d["label"] and iou(best["box"], d["box"]) >= threshold:
                unmatched.remove(best)
                tp += 1
            else:
                fp += 1
        fn += len(unmatched)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", default=VIDEOS, help="файлы из data/")
    parser.add_argument("--every", nargs="+", type=int, default=[2, 3, 5],
                        help="для каких N сравнить режим «каждый N-й кадр»")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.model in COCO_METRICS:
        m = COCO_METRICS[args.model]
        print(f"{args.model} по документации: {m['params']}M параметров, "
              f"COCO mAP@0.5:0.95 {m['map50_95']}, mAP@0.5 {m['map50']}\n")

    rows = []
    for name in args.videos:
        ref = process_video(str(DATA_DIR / name), model_name=args.model)
        print(f"{name}: {ref['frames']} кадров ({ref['duration']:.1f} с, {ref['video_fps']:.0f} кадров/с), "
              f"обработка {ref['processing_fps']:.1f} кадров/с", flush=True)
        for label, s in list(class_stats(ref["detections"]).items())[:6]:
            print(f"  {label:<14} в среднем {s['mean']:.1f} на кадре, максимум {s['max']}")

        row = {"video": name, 1: (ref["processing_fps"], 1.0)}
        for n in args.every:
            r = process_video(str(DATA_DIR / name), every=n, model_name=args.model)
            row[n] = (r["processing_fps"], agreement(ref["detections"], r["detections"]))
        rows.append(row)

    steps = [1] + args.every
    print(f"\nКадров/с и F1 совпадения с обработкой каждого кадра\n{'Видео':<20}"
          + "".join(f"{'N=' + str(n):>16}" for n in steps))
    for row in rows:
        print(f"{row['video']:<20}" + "".join(f"{row[n][0]:>9.1f} / {row[n][1]:.2f}" for n in steps))
