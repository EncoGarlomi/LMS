from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import load_settings


def _resolve_cuda_device(gpu_index: int, gpu_name_filter: str | None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if gpu_name_filter:
        wanted = gpu_name_filter.lower()
        for index in range(device_count):
            device_name = torch.cuda.get_device_name(index)
            if wanted in device_name.lower():
                print(f"Using GPU {index}: {device_name}")
                return torch.device(f"cuda:{index}")

    if 0 <= gpu_index < device_count:
        device_name = torch.cuda.get_device_name(gpu_index)
        print(f"Using GPU {gpu_index}: {device_name}")
        return torch.device(f"cuda:{gpu_index}")

    return torch.device("cuda:0")


class CtrlNewsGenerator:
    def __init__(self, model_source: str | Path) -> None:
        settings = load_settings()
        self.device = _resolve_cuda_device(settings.gpu_index, settings.gpu_name_filter)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_source))
        model_config = AutoConfig.from_pretrained(str(model_source))
        model_config.loss_type = "ForCausalLM"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_source),
            config=model_config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        prompt: str,
        response_language: str = "Ukrainian",
        max_new_tokens: int = 48,
        temperature: float = 0.9,
        top_p: float = 0.92,
        top_k: int = 50,
    ) -> str:
        cleaned_prompt = prompt.strip() or "News headline:"
        instruction = (
            f"Write the answer in {response_language}. "
            f"Do not use Spanish unless the prompt explicitly asks for Spanish. "
            f"Prompt: {cleaned_prompt}"
        )
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if decoded.startswith(instruction):
            decoded = decoded[len(instruction) :]
        elif decoded.startswith(cleaned_prompt):
            decoded = decoded[len(cleaned_prompt) :]
        return decoded.strip(" \n:-") or "No generation was produced."
