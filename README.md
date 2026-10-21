# Прикладные задачи ML на готовых моделях

ПрИнж, семестр 1, серия заданий 1 (с1сз1). Четыре прикладные задачи, решённые готовыми обученными моделями.

| # | Область | Задача | Фреймворк | Модель | Метрика |
|---|---------|--------|-----------|--------|---------|
| 1 | Текст | [Тональность русских текстов](01_text_sentiment/) | Hugging Face | rubert-tiny2-russian-sentiment | accuracy, macro-F1 |
| 2 | Аудио | [Распознавание речи](02_audio_stt/) | Hugging Face | whisper-small | WER, CER |
| 3 | Изображения | [Классификация изображений](03_image_classification/) | Keras Applications | EfficientNetB0 | top-1, top-5 |
| 4 | Видео | [Детекция объектов на видео](04_video_detection/) | PyTorch Hub | YOLOv5s | mAP (COCO), FPS |

Отчёт: [report/REPORT.md](report/REPORT.md)

## Быстрый старт

Требуется Python 3.12 и `ffmpeg`. У каждой задачи своё окружение:

```bash
cd 01_text_sentiment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Подробности — в README каждой задачи.
