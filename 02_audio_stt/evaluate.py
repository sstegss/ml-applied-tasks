"""Оценка Whisper на русской части FLEURS (test).

Метрики: WER и CER (jiwer), скорость — RTF (время обработки / длительность аудио).
"""

import argparse
import io
import re
import time

import jiwer
import soundfile as sf
from datasets import Audio, load_dataset

from transcribe import transcribe

# В Hub FLEURS лежит как скрипт загрузки, которые datasets 5 не запускает,
# поэтому берём его автоматическую parquet-версию
DATA_URL = "hf://datasets/google/fleurs@refs/convert/parquet/ru_ru/test/0000.parquet"


def normalize(text: str) -> str:
    """Привести текст к виду эталона FLEURS: нижний регистр, без знаков препинания."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def load_samples(n: int) -> list[dict]:
    ds = load_dataset("parquet", data_files={"test": DATA_URL}, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))  # декодируем сами через soundfile
    samples = []
    for row in ds.take(n):
        array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        samples.append({"audio": {"raw": array, "sampling_rate": sr},
                        "duration": len(array) / sr,
                        "reference": row["transcription"]})
    return samples


def evaluate(model_name: str, samples: list[dict]) -> dict:
    transcribe(samples[0]["audio"], model_name)  # прогрев: загрузка модели не входит в замер

    hypotheses = []
    start = time.perf_counter()
    for s in samples:
        hypotheses.append(transcribe(s["audio"], model_name))
    elapsed = time.perf_counter() - start

    references = [normalize(s["reference"]) for s in samples]
    normalized = [normalize(h) for h in hypotheses]
    audio_time = sum(s["duration"] for s in samples)
    return {
        "model": model_name,
        "wer": jiwer.wer(references, normalized),
        "cer": jiwer.cer(references, normalized),
        "rtf": elapsed / audio_time,
        "sec_per_sample": elapsed / len(samples),
        "pairs": list(zip(references, normalized)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["openai/whisper-tiny", "openai/whisper-small"])
    parser.add_argument("-n", type=int, default=50, help="число записей")
    args = parser.parse_args()

    samples = load_samples(args.n)
    total = sum(s["duration"] for s in samples)
    print(f"FLEURS ru_ru test: {len(samples)} записей, {total / 60:.1f} мин аудио, "
          f"в среднем {total / len(samples):.1f} с\n")

    results = []
    for model_name in args.models:
        r = evaluate(model_name, samples)
        results.append(r)
        print(f"{model_name}: WER {r['wer']:.3f}, CER {r['cer']:.3f}, "
              f"RTF {r['rtf']:.2f}, {r['sec_per_sample']:.1f} с на запись", flush=True)

    print(f"\n{'Модель':<22}{'WER':>8}{'CER':>8}{'RTF':>8}")
    for r in results:
        print(f"{r['model']:<22}{r['wer']:>8.3f}{r['cer']:>8.3f}{r['rtf']:>8.2f}")

    for r in results:
        print(f"\nПримеры {r['model']} (эталон / распознано):")
        for ref, hyp in r["pairs"][:3]:
            print(f"  = {ref}\n  > {hyp}")
