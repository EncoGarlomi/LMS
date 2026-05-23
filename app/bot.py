from __future__ import annotations

import asyncio
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.config import load_settings
    from app.model import CtrlNewsGenerator
else:
    from .config import load_settings
    from .model import CtrlNewsGenerator


HELP_TEXT = (
    "Надішли тему або короткий запит, і бот згенерує новинний заголовок на базі CTRL.\n"
    "Команди:\n"
    "/generate <тема> - згенерувати відповідь\n"
    "/status - стан моделі (лише для адміна)\n"
    "/traininfo - інформація про тренування (лише для адміна)\n"
    "/lang - вибір мови відповіді\n"
    "/train - запустити тренування (лише для адміна)"
)

logger = logging.getLogger(__name__)


def _is_admin(update: Update, admin_id: int | None) -> bool:
    user = update.effective_user
    return admin_id is not None and user is not None and user.id == admin_id


def _get_generator(application: Application) -> CtrlNewsGenerator:
    generator = application.bot_data.get("generator")
    if generator is None:
        raise RuntimeError("Model generator is not available.")
    return generator


def _reload_generator(application: Application) -> None:
    settings = application.bot_data["settings"]
    application.bot_data["generator"] = CtrlNewsGenerator(settings.model_path)


def _language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("English", callback_data="lang:English"),
            InlineKeyboardButton("Ukrainian", callback_data="lang:Ukrainian"),
        ]]
    )


def _set_response_language(application: Application, language: str) -> None:
    settings = application.bot_data["settings"]
    settings.response_language = language


async def _delete_message_safe(bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        logger.debug("Could not delete message %s in chat %s", message_id, chat_id, exc_info=True)


def _schedule_message_delete(application: Application, chat_id: int, message_id: int, delay_seconds: int = 10) -> None:
    async def _runner() -> None:
        await asyncio.sleep(delay_seconds)
        await _delete_message_safe(application.bot, chat_id, message_id)

    application.create_task(_runner())


def _release_generator(application: Application) -> None:
    generator = application.bot_data.get("generator")
    if generator is None:
        return

    model = getattr(generator, "model", None)
    if model is not None:
        model.to("cpu")
    application.bot_data["generator"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _start_training_process() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "app.train"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


def _format_training_info(
    settings,
    generator: CtrlNewsGenerator | None,
    training_process: subprocess.Popen[str] | None,
    training_started_at: datetime | None,
) -> str:
    model_path = settings.model_path
    model_source = model_path if model_path.exists() else settings.model_name
    device = getattr(generator, "device", "n/a")
    status = "running" if training_process is not None else "idle"
    checkpoint_state = "exists" if model_path.exists() else "missing"
    started_line = "n/a"
    finish_line = "n/a"

    if training_process is not None and training_started_at is not None:
        local_start = training_started_at.astimezone()
        approx_finish = local_start + timedelta(minutes=settings.train_estimated_minutes)
        started_line = local_start.strftime("%Y-%m-%d %H:%M:%S %Z")
        finish_line = approx_finish.strftime("%Y-%m-%d %H:%M:%S %Z")

    return (
        "Training info\n"
        f"Status: {status}\n"
        f"Model source: {model_source}\n"
        f"Checkpoint path: {model_path} ({checkpoint_state})\n"
        f"Output dir: {settings.output_dir}\n"
        f"Dataset: {settings.dataset_name}\n"
        f"Text column: {settings.text_column or 'auto-detect'}\n"
        f"Train samples: {settings.train_sample_limit}\n"
        f"Eval samples: {settings.eval_sample_limit}\n"
        f"Max length: {settings.max_length}\n"
        f"Language: {settings.response_language}\n"
        f"Estimated duration: {settings.train_estimated_minutes} min\n"
        f"Training started at: {started_line}\n"
        f"Approx finish time: {finish_line}\n"
        f"GPU index: {settings.gpu_index}\n"
        f"GPU filter: {settings.gpu_name_filter or 'none'}\n"
        f"Device: {device}\n"
        f"Admin ID: {settings.admin_id}\n"
        "Commands:\n"
        "/train - start training\n"
        "/status - show current bot state"
    )


async def _monitor_training_process(application: Application, chat_id: int, process: subprocess.Popen[str]) -> None:
    settings = application.bot_data["settings"]
    try:
        exit_code = await asyncio.to_thread(process.wait)
    finally:
        application.bot_data["training_process"] = None
        application.bot_data["training_started_at"] = None

    if exit_code == 0:
        try:
            _reload_generator(application)
        except Exception:
            logger.exception("Training finished, but reloading the model failed.")
            await application.bot.send_message(
                chat_id=chat_id,
                text="Тренування завершилося, але перезавантажити модель не вдалося. Перезапусти бота вручну.",
            )
            return

        await application.bot.send_message(
            chat_id=chat_id,
            text=f"Тренування завершено успішно. Модель оновлено з {settings.model_path}.",
        )
        return

    await application.bot.send_message(
        chat_id=chat_id,
        text=f"Тренування завершилося з помилкою. Код виходу: {exit_code}.",
    )


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    message = update.effective_message
    if not _is_admin(update, settings.admin_id):
        if message is not None:
            await message.reply_text("Недостатньо прав.")
        return

    if context.application.bot_data.get("training_process") is not None:
        if message is not None:
            await message.reply_text("Тренування вже запущено.")
        return

    try:
        _release_generator(context.application)
        process = _start_training_process()
        context.application.bot_data["training_process"] = process
        context.application.bot_data["training_started_at"] = datetime.now(timezone.utc)
    except Exception as exc:
        logger.exception("Failed to start training")
        if message is not None:
            await message.reply_text(f"Не вдалося запустити тренування: {exc}")
        _reload_generator(context.application)
        return

    if message is not None:
        await message.reply_text(
            f"Тренування запущено. Після завершення модель буде перечитана з {settings.model_path}."
        )

    chat = update.effective_chat
    if chat is not None:
        context.application.create_task(_monitor_training_process(context.application, chat.id, process))


async def _generate_reply(generator: CtrlNewsGenerator, prompt: str, settings) -> str:
    return await asyncio.to_thread(
        generator.generate,
        prompt,
        settings.response_language,
        settings.max_new_tokens,
        settings.temperature,
        settings.top_p,
        settings.top_k,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is not None:
        sent_message = await message.reply_text(HELP_TEXT)
        await _delete_message_safe(context.bot, message.chat_id, message.message_id)
        _schedule_message_delete(context.application, sent_message.chat_id, sent_message.message_id, 20)


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    settings = context.application.bot_data["settings"]
    sent_message = await message.reply_text(
        f"Current language: {settings.response_language}\nChoose a reply language:",
        reply_markup=_language_keyboard(),
    )
    await _delete_message_safe(context.bot, message.chat_id, message.message_id)
    _schedule_message_delete(context.application, sent_message.chat_id, sent_message.message_id, 15)


async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return

    await query.answer()
    _, language = query.data.split(":", 1)
    _set_response_language(context.application, language)

    settings = context.application.bot_data["settings"]
    await query.edit_message_text(
        f"Reply language set to: {settings.response_language}",
    )
    if query.message is not None:
        _schedule_message_delete(context.application, query.message.chat_id, query.message.message_id, 8)


async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    if context.application.bot_data.get("training_process") is not None:
        message = update.effective_message
        if message is not None:
            await message.reply_text("Тренування зараз запущено. Генерація тимчасово недоступна.")
        return

    generator = _get_generator(context.application)
    message = update.effective_message
    prompt = " ".join(context.args or []).strip()
    if not prompt and message is not None and message.text:
        prompt = message.text.replace("/generate", "").strip()
    if not prompt:
        if message is not None:
            await message.reply_text("Напиши тему після /generate.")
        return

    chat = update.effective_chat
    if chat is not None:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    result = await _generate_reply(generator, prompt, settings)
    if message is not None:
        await message.reply_text(result)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    if not _is_admin(update, settings.admin_id):
        message = update.effective_message
        if message is not None:
            await message.reply_text("Недостатньо прав.")
        return

    message = update.effective_message
    if message is None:
        return

    training_process = context.application.bot_data.get("training_process")
    generator = context.application.bot_data.get("generator")
    device = getattr(generator, "device", "n/a")

    await message.reply_text(
        f"Модель: {settings.model_path if settings.model_path.exists() else settings.model_name}\n"
        f"Пристрій: {device}\n"
        f"Admin ID: {settings.admin_id}\n"
        f"Language: {settings.response_language}\n"
        f"Training: {'running' if training_process is not None else 'idle'}",
    )


async def training_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    if not _is_admin(update, settings.admin_id):
        message = update.effective_message
        if message is not None:
            await message.reply_text("Недостатньо прав.")
        return

    message = update.effective_message
    if message is None:
        return

    info_text = _format_training_info(
        settings,
        context.application.bot_data.get("generator"),
        context.application.bot_data.get("training_process"),
        context.application.bot_data.get("training_started_at"),
    )
    sent_message = await message.reply_text(info_text)
    await _delete_message_safe(context.bot, message.chat_id, message.message_id)
    _schedule_message_delete(context.application, sent_message.chat_id, sent_message.message_id, 30)


async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    if context.application.bot_data.get("training_process") is not None:
        message = update.effective_message
        if message is not None:
            await message.reply_text("Тренування зараз запущено. Зачекай завершення.")
        return

    generator = _get_generator(context.application)
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None or not message.text:
        return

    prompt = message.text.strip()
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    result = await _generate_reply(generator, prompt, settings)
    await message.reply_text(result)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if isinstance(error, Conflict):
        logger.error(
            "Telegram polling conflict: another bot instance is already using this token. "
            "Stop the other process and restart this one.",
        )
        return

    logger.exception("Unhandled bot error", exc_info=error)


def build_application() -> Application:
    settings = load_settings()
    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN is missing in .env")

    model_source: str | Path = settings.model_path if settings.model_path.exists() else settings.model_name
    generator = CtrlNewsGenerator(model_source)

    application = Application.builder().token(settings.bot_token).build()
    application.bot_data["settings"] = settings
    application.bot_data["generator"] = generator
    application.bot_data["training_process"] = None
    application.bot_data["training_started_at"] = None

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("generate", generate_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("traininfo", training_info))
    application.add_handler(CommandHandler("lang", language_command))
    application.add_handler(CallbackQueryHandler(language_callback, pattern=r"^lang:(English|Ukrainian)$"))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))
    return application


def main() -> None:
    application = build_application()
    application.add_error_handler(on_error)
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
