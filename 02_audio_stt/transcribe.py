"""Распознавание русской речи моделью Whisper."""

import argparse
from functools import lru_cache

from transformers import pipeline

DEFAULT_MODEL = "openai/whisper-small"


@lru_cache
def get_asr(model_name: str = DEFAULT_MODEL):
    """Загрузить модель один раз и дальше брать из кэша."""
    return pipeline("automatic-speech-recognition", model=model_name)


def transcribe(audio, model_name: str = DEFAULT_MODEL) -> str:
    """Перевести речь в текст.

    audio — путь к файлу (wav, mp3, ...) или словарь {"raw": numpy-массив, "sampling_rate": 16000}.
    """
    if isinstance(audio, dict):
        audio = dict(audio)  # pipeline забирает ключи из словаря, не портим чужой
    asr = get_asr(model_name)
    result = asr(
        audio,
        chunk_length_s=30,  # Whisper слушает максимум 30 с, длинные записи режем на куски
        generate_kwargs={"language": "russian", "task": "transcribe"},
    )
    return result["text"].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="аудиофайлы")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="например, openai/whisper-tiny")
    args = parser.parse_args()
    for path in args.files:
        print(f"{path}: {transcribe(path, args.model)}")
