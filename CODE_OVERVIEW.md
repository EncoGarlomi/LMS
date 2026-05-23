
# Подробный обзор кода (функции и поведение)

Ниже — расширенная документация по каждому модулю и ключевым функциям, помещённая целиком в этот файл по запросу.

## Быстрый старт
- Убедитесь, что виртуальное окружение активно и зависимости установлены:

```bash
pip install -r requirements.txt
```

- Скопируйте `.env.example` в `.env` и заполните `BOT_TOKEN` (и опционально `ADMIN_ID`).
- Запуск бота:

```bash
python -m app.bot
```

## `app/config.py` — загрузка настроек
- `load_settings()` читает переменные окружения через `dotenv` и возвращает `Settings` (dataclass).
- Ключевые поля `Settings`:
  - `bot_token`, `admin_id` — Telegram.
  - `model_name`, `model_path`, `output_dir` — модель/чекпоинт.
  - Параметры генерации: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `max_length`.
  - Параметры тренировки: `train_batch_size`, `train_sample_limit`, `eval_sample_limit`, `train_estimated_minutes`.
  - GPU: `gpu_index`, `gpu_name_filter`.

Используйте этот модуль чтобы менять поведение приложения через `.env`.

## `app/model.py` — `CtrlNewsGenerator`
- Конструктор (`__init__(self, model_source)`) выполняет:
  - `settings = load_settings()`
  - `self.device = _resolve_cuda_device(...)` — проверяет CUDA и выбирает устройство.
  - `self.tokenizer = AutoTokenizer.from_pretrained(str(model_source))`.
  - `model = AutoModelForCausalLM.from_pretrained(...)` с `low_cpu_mem_usage=True`.
  - Устанавливает `pad_token` при необходимости и ресайзит эмбеддинги.
  - Переносит модель на `self.device` и переводит в `eval()` режим.

- Метод `generate(self, prompt, response_language, max_new_tokens, temperature, top_p, top_k)`:
  - Формирует инструкцию:
    - Пример: "Write the answer in Ukrainian. Do not use Spanish... Prompt: {prompt}".
  - Токенизирует вход и вызывает `self.model.generate(...)` с параметрами семплинга и `pad_token_id`/`eos_token_id`.
  - Декодирует результат и убирает префикс инструкции/промпта.
  - Возвращает строку-результат или сообщение "No generation was produced." если пусто.

Проблемы: отсутствие необходимых токенов, несовпадение pad/eos, нехватка памяти на GPU.

## `app/data.py` — подготовка данных
- `detect_text_column(dataset, preferred_column)`:
  - Если задан `preferred_column`, проверяет его наличие.
  - Иначе ищет распространённые имена колонок (`text`, `headline`, `title`, ...).
  - Если ничего не найдено, бросает `ValueError` с перечислением доступных колонок.

- `load_headline_texts(dataset_name, preferred_column)`:
  - Загружает датасет через `datasets.load_dataset(dataset_name)`.
  - Выбирает первую подходящую сплит-часть (train или первый доступный).
  - Определяет текстовую колонку и возвращает список строк и имя колонки.

Этот модуль используется только в `train.py` для формирования очереди/батчей.

## `app/train.py` — процесс fine-tune
Главная логика в `main()`.

- Поток:
  1. `settings = load_settings()`
  2. `_load_training_queue()` — читает `artifacts/.../training_data/queue.json` если есть, иначе вызывает `load_headline_texts()`.
  3. Выбирает батч текстов (по `TRAIN_BATCH_SIZE`), делит на train/eval и применяет лимиты `train_sample_limit`/`eval_sample_limit`.
  4. Загружает токенайзер и модель (`settings.model_name`) и при необходимости добавляет `pad_token`.
  5. Строит токенизованный `Dataset` и `DataCollatorForLanguageModeling`.
  6. Конфигурирует `TrainingArguments` (параметры сохранения, вычислений BF16 и др.) и создаёт `Trainer`.
  7. `trainer.train()` → `trainer.save_model()` и `tokenizer.save_pretrained()`.
  8. Обновляет `queue.json` и `processed.json` (переносит обработанные строки в processed).

Функции/утилиты:
- `_resolve_cuda_device(gpu_index, gpu_name_filter)` — выбрасывает исключение, если CUDA недоступна (тренировка требует GPU).
- `_training_state_paths(settings)` — возвращает пути для `queue.json` и `processed.json` и создаёт папку.
- `_load_training_queue(...)` / `_save_training_queue(...)` — сериализация/десериализация состояния очереди.

Ограничения: тренировка требует CUDA; параметры `bf16`/`fp16` настроены под конкретную среду.

## `app/bot.py` — handlers и операции бота
Ключевые функции и их роль:

- `_is_admin(update, admin_id)` — проверяет `update.effective_user.id == admin_id`.

- `_get_generator(application)` — возвращает `application.bot_data['generator']` или бросает RuntimeError.

- `_reload_generator(application)` — читает `settings` из `bot_data` и создаёт новый `CtrlNewsGenerator(settings.model_path)`.

- `_language_keyboard()` — строит `InlineKeyboardMarkup` для выбора языка.

- `_set_response_language(application, language)` — изменяет `settings.response_language` в `bot_data`.

- `_delete_message_safe(bot, chat_id, message_id)` / `_schedule_message_delete(...)` — безопасно удаляют сообщения (не критично если удаление не вдається).

- `_release_generator(application)` — освобождает модель из памяти (переносит на CPU, очищає `bot_data['generator']`, очищає кеш CUDA).

- `_start_training_process()` — запускає `subprocess.Popen([sys.executable, '-m', 'app.train'])` як окремий процес; на Windows додає `CREATE_NEW_PROCESS_GROUP`.

- `_format_training_info(settings, generator, training_process, training_started_at)` — формирует текст для `/traininfo`.

- `_monitor_training_process(application, chat_id, process)` — асинхронно очікує завершення процесу, після успіху викликає `_reload_generator()` і повідомляє чат.

Хендлери (async):
- `train_command` — запускає `_start_training_process()` (тільки для адміна), встановлює `bot_data['training_process']` і створює таск `_monitor_training_process`.
- `generate_command` — витягує `prompt` з `/generate` або аргументів, викликає `_generate_reply()` і відправляє повідомлення.
- `plain_text` — обробка звичайних текстових повідомлень як запит на генерацію.
- `status`, `training_info` — повертають інформацію лише для адміна.
- `language_command`, `language_callback` — дозволяють змінити `response_language` через inline-кнопки.

`build_application()` — основне: читає `settings`, створює `CtrlNewsGenerator`, додає хендлери і повертає `Application`.

Після запуску `application.run_polling(...)` бот починає отримувати апдейти Telegram.

## `app/cli.py` і `run_bot.bat`
- `app/cli.py` просто викликає `bot.main()` при `python -m app.cli`.
- `run_bot.bat` запускає бот через `.venv\Scripts\python.exe` і виводить повідомлення при відсутності інтерпретатора у віртуальному середовищі.

## Об artefacts
- Місце збереження моделей: `artifacts/ctrl-news/`.
- Черга та оброблені рядки: `artifacts/ctrl-news/training_data/queue.json`, `processed.json`.

## Часті помилки та як їх вирішити
- `RuntimeError: BOT_TOKEN is missing` — перевірити `.env` та `BOT_TOKEN`.
- `Conflict` при підключенні до Telegram — значить інший процес вже використовує цей BOT_TOKEN.
- Training: `CUDA required` — тренування запускайте на машині з NVIDIA GPU.
- Пам'ять GPU — зменшіть `train_batch_size` або використайте менший `model_name`.

---
Готово — файл оновлён: все описано здесь. Хочешь, переведу обратно на украинский или добавлю диаграмму потоков (Mermaid)?

