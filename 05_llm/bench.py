"""Сравнение локальных LLM: скорость на CPU и простая проверка ответов.

    python bench.py                                   # qwen3.5:0.8b, 2b, 4b
    python bench.py --models qwen3.5:4b gemma4:e2b

Метрики:
- скорость обработки промпта и генерации (токенов/с) из счётчиков Ollama;
- время загрузки модели в память;
- доля правильных ответов на 8 вопросах с однозначным ответом (проверка по подстроке).
  Это не бенчмарк вроде MMLU, а быстрая проверка, что модель вообще отвечает по делу.
"""

import argparse
import statistics
import subprocess

from chat import OLLAMA_URL, SYSTEM_PROMPT, chat

# (вопрос, подстроки, хотя бы одна из которых должна быть в ответе)
QUESTIONS = [
    ("Какая столица Франции? Ответь одним словом.", ["париж", "paris"]),
    ("Сколько будет 17 умножить на 23? Напиши только число.", ["391"]),
    ("Кто написал роман «Война и мир»? Только фамилия.", ["толст"]),
    ("Переведи на английский: «кошка спит на диване». Только перевод.", ["cat", "sofa", "couch"]),
    ("Какой химический символ у золота? Только символ.", ["au"]),
    ("Назови самую большую планету Солнечной системы одним словом.", ["юпитер", "jupiter"]),
    ("Что больше: 0.9 или 0.11? Ответь числом.", ["0.9", "0,9"]),
    ("В каком году человек впервые высадился на Луну? Только год.", ["1969"]),
]

# Длинный промпт, чтобы замерить скорость обработки входа отдельно от генерации.
LONG_PROMPT = (
    "Прочитай текст и перескажи его одним предложением.\n\n"
    "Машинное обучение это раздел искусственного интеллекта, в котором программы учатся на данных, "
    "а не программируются явно. Модель подбирает параметры так, чтобы минимизировать ошибку на обучающей "
    "выборке, а затем проверяется на отложенных данных, которых не видела. Если качество на обучении "
    "высокое, а на новых данных низкое, говорят о переобучении. Чтобы его избежать, используют "
    "регуляризацию, больше данных, более простые модели и раннюю остановку обучения. "
) * 4


def ask(model: str, question: str, url: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return chat(messages, model=model, url=url)


def memory_gb(model: str) -> str:
    """Сколько памяти модель занимает по `ollama ps` (пусто, если команды нет)."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in out.splitlines():
        if line.startswith(model):
            parts = line.split()
            return f"{parts[2]} {parts[3]}" if len(parts) > 3 else ""
    return ""


def bench(model: str, url: str) -> dict:
    # Первый вызов включает загрузку весов, замеряем её отдельно.
    _, first = ask(model, "Привет! Ответь одним словом.", url)
    _, long = ask(model, LONG_PROMPT, url)
    correct, gen_tps, prompt_tps, totals, answers = 0, [], [], [], []
    for q, expected in QUESTIONS:
        answer, s = ask(model, q, url)
        ok = any(e in answer.lower() for e in expected)
        correct += ok
        gen_tps.append(s["gen_tps"])
        prompt_tps.append(s["prompt_tps"])
        totals.append(s["total_s"])
        answers.append((q, answer, ok))
    return {
        "model": model,
        "load_s": first["load_s"],
        "memory": memory_gb(model),
        "prompt_tps_long": long["prompt_tps"],
        "prompt_tokens_long": long["prompt_tokens"],
        "gen_tps": statistics.median(gen_tps),
        "median_s": statistics.median(totals),
        "correct": correct,
        "answers": answers,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"])
    parser.add_argument("--url", default=OLLAMA_URL)
    args = parser.parse_args()

    results = []
    for model in args.models:
        print(f"{model}...", flush=True)
        r = bench(model, args.url)
        results.append(r)
        for q, a, ok in r["answers"]:
            print(f"  {'+' if ok else '-'} {q}\n    > {a[:120]}")
        print(f"  верно {r['correct']}/{len(QUESTIONS)}, генерация {r['gen_tps']:.1f} ток/с, "
              f"промпт {r['prompt_tps_long']:.0f} ток/с, загрузка {r['load_s']:.1f} с\n", flush=True)

    print(f"{'Модель':<16}{'Загрузка, с':>12}{'Память':>12}{'Промпт, ток/с':>15}{'Генерация, ток/с':>18}"
          f"{'Медиана, с':>12}{'Верно':>8}")
    for r in results:
        print(f"{r['model']:<16}{r['load_s']:>12.1f}{r['memory']:>12}{r['prompt_tps_long']:>15.0f}"
              f"{r['gen_tps']:>18.1f}{r['median_s']:>12.1f}{r['correct']:>5}/{len(QUESTIONS)}")
