from transformers import pipeline, BitsAndBytesConfig
import torch

_pipe = None  # глобальная переменная для хранения загруженной модели

def load_model():
    """Загружает модель в память."""
    global _pipe
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    _pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-4B-Instruct-2507",
        device_map="auto",
        model_kwargs={"quantization_config": quantization_config},
        dtype=torch.bfloat16,
    )
    return _pipe

def get_pipe():
    """Возвращает загруженную модель (вызывается после load_model)."""
    return _pipe

def gen_message(user_text: str) -> str:
    """Генерирует ответ, используя уже загруженную модель."""
    pipe = get_pipe()
    if pipe is None:
        raise RuntimeError("Модель ещё не загружена")
    messages = [{"role": "user", "content": user_text}]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = pipe(
        prompt,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.7,
        pad_token_id=pipe.tokenizer.eos_token_id,
        generation_config=None,
        return_full_text=False,
    )
    return output[0]['generated_text'].strip()