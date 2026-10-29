"""Оценка модели тональности на тестовой части RuSentiTweet.

Метрики: accuracy, macro-F1, отчёт по классам, матрица ошибок, время работы.
"""

import time

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sentiment import predict, predict_batch

DATASET = "psytechlab/RuSentiTweet"
LABELS = ["positive", "neutral", "negative"]

# В датасете есть ещё метки skip (непонятно) и speech (речевой акт), их у модели нет
ds = load_dataset(DATASET, split="test").filter(lambda x: x["label"] in LABELS)
texts, y_true = list(ds["text"]), list(ds["label"])
print(f"Датасет: {DATASET}, test, {len(texts)} примеров")
for label in LABELS:
    print(f"  {label:8} {y_true.count(label)}")

start = time.perf_counter()
y_pred = [r["label"] for r in predict_batch(texts)]
batch_time = time.perf_counter() - start

print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.3f}\n")
print(classification_report(y_true, y_pred, labels=LABELS, digits=3))

print("Матрица ошибок (строки — истина, столбцы — предсказание):")
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
print(" " * 10 + "".join(f"{l:>10}" for l in LABELS))
for label, row in zip(LABELS, cm):
    print(f"{label:>10}" + "".join(f"{v:>10}" for v in row))

# Время по одному тексту — как в API, где запросы приходят поштучно
n = 200
start = time.perf_counter()
for text in texts[:n]:
    predict(text)
single_time = (time.perf_counter() - start) / n

print(f"\nБатчами по 32: {batch_time:.1f} с на {len(texts)} текстов, "
      f"{len(texts) / batch_time:.0f} текстов/с")
print(f"По одному: {single_time * 1000:.1f} мс на текст")
