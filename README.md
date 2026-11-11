# Прикладные задачи ML на готовых моделях

ПрИнж, семестр 1, серия заданий 1 (с1сз1). Четыре прикладные задачи, решённые готовыми обученными моделями, плюс локально развёрнутая LLM.

| # | Область | Задача | Фреймворк | Модель | Результат |
|---|---------|--------|-----------|--------|-----------|
| 1 | Текст | [Тональность русских текстов](01_text_sentiment/) | Hugging Face | rubert-tiny2-russian-sentiment | macro-F1 0.617 на RuSentiTweet, 8 мс на текст |
| 2 | Аудио | [Распознавание речи](02_audio_stt/) | Hugging Face | whisper-small | WER 0.096 на FLEURS ru |
| 3 | Изображения | [Классификация изображений](03_image_classification/) | Keras Applications | EfficientNetB0 | top-1 0.780, top-5 0.956 на ImageNet-val (10 классов) |
| 4 | Видео | [Детекция объектов на видео](04_video_detection/) | PyTorch Hub | YOLOv5s | mAP@0.5 56.8 на COCO (по документации), 8–9 кадров/с на CPU |
| 5 | LLM | [Локальная LLM](05_llm/) | Ollama (llama.cpp) | qwen3.5:4b, Q4_K_M | 6.4 токена/с генерации на CPU, 3.3 ГБ |

Фреймворки: задачи 1, 2 и 4 работают на PyTorch, задача 3 на TensorFlow, LLM запускается движком Ollama. Всё считается на CPU (Intel i3-7100), видеокарта не нужна.

Решение задачи 1 (тональность) пойдёт под API на FastAPI в следующем задании: на входе строка, на выходе метка и вероятность, а модель маленькая и отвечает быстро.

Отчёт: [report/REPORT.md](report/REPORT.md)

## Быстрый старт

Требуется Python 3.12 и `ffmpeg`, для задачи 5 ещё Ollama. У каждой задачи своё окружение:

```bash
cd 01_text_sentiment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Подробности в README каждой задачи.
