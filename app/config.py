from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    bot_token: str
    admin_id: int | None
    model_name: str
    dataset_name: str
    text_column: str | None
    response_language: str
    train_estimated_minutes: int
    train_batch_size: int
    model_path: Path
    output_dir: Path
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    max_length: int
    train_sample_limit: int
    eval_sample_limit: int
    gpu_index: int
    gpu_name_filter: str | None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def load_settings() -> Settings:
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/ctrl-news/checkpoint-15"))
    return Settings(
        bot_token=os.getenv("BOT_TOKEN", "").strip(),
        admin_id=_parse_optional_int(os.getenv("ADMIN_ID")),
        model_name=os.getenv("MODEL_NAME", "salesforce/ctrl").strip(),
        dataset_name=os.getenv("DATASET_NAME", "shaurya03/tech-news-daily").strip(),
        text_column=os.getenv("TEXT_COLUMN", "").strip() or None,
        response_language=os.getenv("RESPONSE_LANGUAGE", "Ukrainian").strip(),
        train_estimated_minutes=int(os.getenv("TRAIN_ESTIMATED_MINUTES", "60")),
        train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "125")),
        model_path=model_path,
        output_dir=model_path,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "48")),
        temperature=float(os.getenv("TEMPERATURE", "0.9")),
        top_p=float(os.getenv("TOP_P", "0.92")),
        top_k=int(os.getenv("TOP_K", "50")),
        max_length=int(os.getenv("MAX_LENGTH", "32")),
        train_sample_limit=int(os.getenv("TRAIN_SAMPLE_LIMIT", "1000")),
        eval_sample_limit=int(os.getenv("EVAL_SAMPLE_LIMIT", "50")),
        gpu_index=int(os.getenv("GPU_INDEX", "0")),
        gpu_name_filter=os.getenv("GPU_NAME_FILTER", "").strip() or None,
    )
