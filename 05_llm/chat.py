"""Локальная LLM через Ollama: вопрос из командной строки или диалог.

    python chat.py "Объясни, что такое переобучение, в двух предложениях"
    python chat.py                      # диалог, /exit для выхода
    python chat.py --model qwen3.5:0.8b -v   # -v: токены и скорость из ответа Ollama

Зависимостей нет: Ollama отдаёт HTTP API, хватает стандартной библиотеки.
"""

import argparse
import json
import sys
import urllib.request

DEFAULT_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://127.0.0.1:11434"
SYSTEM_PROMPT = "Ты помощник. Отвечай на языке вопроса, коротко и по делу."


def chat(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.2,
         num_ctx: int = 4096, think: bool = False, url: str = OLLAMA_URL) -> tuple[str, dict]:
    """Один вызов модели. Возвращает текст ответа и статистику Ollama."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Режим размышлений у Qwen3.5 на CPU стоит минуты, для коротких ответов он не нужен.
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        f"{url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    return data["message"]["content"].strip(), stats(data)


def stats(data: dict) -> dict:
    """Скорость из счётчиков Ollama: длительности приходят в наносекундах."""
    ns = 1e9
    p_tokens, g_tokens = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    p_sec, g_sec = data.get("prompt_eval_duration", 0) / ns, data.get("eval_duration", 0) / ns
    return {
        "prompt_tokens": p_tokens,
        "gen_tokens": g_tokens,
        "prompt_tps": p_tokens / p_sec if p_sec else 0.0,  # обработка промпта, токенов/с
        "gen_tps": g_tokens / g_sec if g_sec else 0.0,  # генерация, токенов/с
        "load_s": data.get("load_duration", 0) / ns,  # загрузка весов (0, если модель уже в памяти)
        "total_s": data.get("total_duration", 0) / ns,
    }


def format_stats(s: dict) -> str:
    return (f"[промпт {s['prompt_tokens']} ток. @ {s['prompt_tps']:.0f} ток/с, "
            f"ответ {s['gen_tokens']} ток. @ {s['gen_tps']:.1f} ток/с, "
            f"загрузка {s['load_s']:.1f} с, всего {s['total_s']:.1f} с]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="вопрос; без него откроется диалог")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="например, qwen3.5:0.8b или gemma4:e2b")
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--think", action="store_true", help="включить режим размышлений (медленно на CPU)")
    parser.add_argument("-v", "--verbose", action="store_true", help="печатать токены и скорость")
    args = parser.parse_args()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(text: str):
        history.append({"role": "user", "content": text})
        try:
            answer, s = chat(history, args.model, think=args.think, url=args.url)
        except OSError as e:
            sys.exit(f"Ollama не отвечает на {args.url}: {e}. Запустите `ollama serve` и `ollama pull {args.model}`.")
        history.append({"role": "assistant", "content": answer})
        print(answer)
        if args.verbose:
            print(format_stats(s))

    if args.question:
        ask(args.question)
    else:
        print(f"Модель {args.model}. Ответ на CPU может идти десятки секунд. /exit для выхода.")
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/exit":
                break
            ask(text)
