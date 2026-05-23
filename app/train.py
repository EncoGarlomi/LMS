from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.config import load_settings
else:
    from .config import load_settings


def _resolve_cuda_device(gpu_index: int, gpu_name_filter: str | None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Run this on a machine with an NVIDIA GPU and CUDA-enabled PyTorch."
        )

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("CUDA reports zero devices even though it is available.")

    if gpu_name_filter:
        wanted = gpu_name_filter.lower()
        for index in range(device_count):
            device_name = torch.cuda.get_device_name(index)
            if wanted in device_name.lower():
                print(f"Using GPU {index}: {device_name}")
                return torch.device(f"cuda:{index}")

        available = ", ".join(torch.cuda.get_device_name(index) for index in range(device_count))
        raise RuntimeError(
            f"No CUDA device matched GPU_NAME_FILTER='{gpu_name_filter}'. Available devices: {available}"
        )

    if gpu_index < 0 or gpu_index >= device_count:
        available = ", ".join(torch.cuda.get_device_name(index) for index in range(device_count))
        raise RuntimeError(
            f"GPU_INDEX={gpu_index} is out of range. Available devices: {available}"
        )

    device_name = torch.cuda.get_device_name(gpu_index)
    print(f"Using GPU {gpu_index}: {device_name}")
    return torch.device(f"cuda:{gpu_index}")


def _take_subset(texts: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(texts) <= limit:
        return texts
    return texts[:limit]


def _training_state_paths(settings) -> tuple[Path, Path]:
    training_dir = settings.output_dir / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)
    return training_dir / "queue.json", training_dir / "processed.json"


def _load_training_queue(settings, dataset_name: str, preferred_column: str | None) -> tuple[list[str], list[str], str]:
    if __package__ is None or __package__ == "":
        from app.data import load_headline_texts
    else:
        from .data import load_headline_texts

    queue_path, processed_path = _training_state_paths(settings)

    if queue_path.exists():
        with queue_path.open("r", encoding="utf-8") as handle:
            queue_data = json.load(handle)
        pending = [str(item) for item in queue_data.get("pending", []) if str(item).strip()]
        processed = [str(item) for item in queue_data.get("processed", []) if str(item).strip()]
        text_column = str(queue_data.get("text_column", ""))
        if pending or processed:
            return pending, processed, text_column

    texts, text_column = load_headline_texts(dataset_name, preferred_column)
    processed: list[str] = []

    if processed_path.exists():
        with processed_path.open("r", encoding="utf-8") as handle:
            processed = [line.strip() for line in handle if line.strip()]

    pending = [text for text in texts if text not in set(processed)]
    _save_training_queue(settings, pending, processed, text_column)
    return pending, processed, text_column


def _save_training_queue(settings, pending: list[str], processed: list[str], text_column: str) -> None:
    queue_path, processed_path = _training_state_paths(settings)
    payload = {
        "text_column": text_column,
        "pending": pending,
        "processed": processed,
    }
    with queue_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with processed_path.open("w", encoding="utf-8") as handle:
        for item in processed:
            handle.write(item + "\n")


def _build_tokenized_dataset(tokenizer, texts: list[str], max_length: int):
    dataset = {
        "text": texts,
    }
    from datasets import Dataset

    raw_dataset = Dataset.from_dict(dataset)
    tokenized = raw_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    return tokenized


def main() -> None:
    print("Loading training dependencies...")
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    if __package__ is None or __package__ == "":
        from app.data import load_headline_texts
    else:
        from .data import load_headline_texts

    print("Loading dataset state...")

    settings = load_settings()
    pending_texts, processed_texts, text_column = _load_training_queue(
        settings,
        settings.dataset_name,
        settings.text_column,
    )

    if not pending_texts:
        raise RuntimeError("No pending training rows left. All dataset rows were already processed.")
    device = _resolve_cuda_device(settings.gpu_index, settings.gpu_name_filter)

    batch_size = max(1, settings.train_batch_size)
    batch_texts = _take_subset(pending_texts, batch_size)
    remaining_texts = pending_texts[len(batch_texts):]

    split_index = max(1, int(len(batch_texts) * 0.95))
    train_texts = _take_subset(batch_texts[:split_index], settings.train_sample_limit)
    eval_texts = _take_subset(
        batch_texts[split_index:] if split_index < len(batch_texts) else batch_texts[:1],
        settings.eval_sample_limit,
    )

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    pad_token_was_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            pad_token_was_added = True

    model_config = AutoConfig.from_pretrained(settings.model_name)
    model_config.loss_type = "ForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if pad_token_was_added:
        cast(Any, model).resize_token_embeddings(len(tokenizer))
    model = cast(Any, model).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.loss_type = "ForCausalLM"
    model.config.use_cache = False

    train_dataset = _build_tokenized_dataset(tokenizer, train_texts, settings.max_length)
    eval_dataset = _build_tokenized_dataset(tokenizer, eval_texts, settings.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(settings.output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "logging_steps": 25,
        "save_steps": 250,
        "eval_strategy": "steps",
        "eval_steps": 250,
        "save_total_limit": 2,
        "report_to": [],
        "dataloader_pin_memory": torch.cuda.is_available(),
        "bf16": True,
        "fp16": False,
        "push_to_hub": False,
    }
    training_args_factory = cast(Any, TrainingArguments)
    training_args = training_args_factory(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(settings.output_dir))
    tokenizer.save_pretrained(str(settings.output_dir))

    processed_texts.extend(batch_texts)
    _save_training_queue(settings, remaining_texts, processed_texts, text_column)

    print(f"Training completed. Dataset text column: {text_column}")
    print(f"Processed batch size: {len(batch_texts)}")
    print(f"Remaining pending rows: {len(remaining_texts)}")
    print(f"Model saved to: {settings.output_dir}")


if __name__ == "__main__":
    main()
