"""Shared Qwen2.5-7B-Instruct helpers used by steps 1 and 3."""

import gc
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_model, _tokenizer = None, None


def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return _model, _tokenizer


def run_batch(system_prompt: str, user_messages: list[str], batch_size: int = 4,
              max_length: int = 4096, max_new_tokens: int = 1024) -> list[str]:
    """Run the chat model over user_messages; returns raw responses (prompt stripped)."""
    model, tokenizer = load_model()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": um}],
            tokenize=False, add_generation_prompt=True,
        )
        for um in user_messages
    ]

    responses = []
    for i in range(0, len(prompts), batch_size):
        inputs = tokenizer(
            prompts[i:i + batch_size], return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )

        input_len = inputs.input_ids.shape[1]
        responses.extend(tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True))

        del inputs, generated_ids
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return responses


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model response; falls back to an error dict."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"parse_error": True, "raw_response": text}
