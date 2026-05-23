from __future__ import annotations

from collections.abc import Sequence
from typing import Any


TEXT_COLUMN_CANDIDATES = (
    "text",
    "headline",
    "title",
    "content",
    "article",
    "news",
    "body",
)


def _first_dataset(dataset: Any):
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return next(iter(dataset.values()))
    return dataset


def detect_text_column(dataset: Any, preferred_column: str | None = None) -> str:
    if preferred_column:
        if preferred_column not in dataset.column_names:
            available = ", ".join(dataset.column_names)
            raise ValueError(
                f"Column '{preferred_column}' was not found. Available columns: {available}"
            )
        return preferred_column

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in dataset.column_names:
            return candidate

    available = ", ".join(dataset.column_names)
    raise ValueError(
        "Could not detect a text column automatically. "
        f"Available columns: {available}. Set TEXT_COLUMN in .env."
    )


def load_headline_texts(dataset_name: str, preferred_column: str | None = None) -> tuple[list[str], str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    first_split = _first_dataset(dataset)
    text_column = detect_text_column(first_split, preferred_column)

    texts = []
    for value in first_split[text_column]:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            texts.append(text)
    return texts, text_column
