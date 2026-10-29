"""Определение тональности русского текста: positive / neutral / negative."""

import sys

from transformers import pipeline

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

# Модель грузится один раз при импорте модуля: загрузка занимает секунды,
# а одно предсказание — миллисекунды.
clf = pipeline("text-classification", model=MODEL_NAME)


def predict(text: str) -> dict:
    """Вернуть самую вероятную метку и её вероятность."""
    result = clf(text, truncation=True)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}


def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    results = clf(texts, batch_size=batch_size, truncation=True)
    return [{"label": r["label"], "score": round(r["score"], 4)} for r in results]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python sentiment.py "текст" ["текст2" ...]')
        sys.exit(1)
    for text in sys.argv[1:]:
        r = predict(text)
        print(f"{r['label']:8} {r['score']:.2f}  {text}")
