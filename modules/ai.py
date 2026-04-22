from transformers import pipeline, BitsAndBytesConfig
import torch
import os

_pipe = None           # глобальная переменная для хранения загруженной модели
_chat_history = []     # история диалога: список {"role": "user/assistant", "content": ...}

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

def reset_history():
    """Очищает историю диалога."""
    global _chat_history
    _chat_history = []

def get_history():
    """Возвращает текущую историю диалога."""
    return _chat_history.copy()

def gen_message(user_text: str, use_history: bool = True) -> str:
    """
    Генерирует ответ, используя историю диалога.
    
    Параметры:
        user_text (str): текст сообщения пользователя.
        use_history (bool): если True, используется накопленная история;
                            если False, генерируется ответ без контекста (история не меняется).
    
    Возвращает:
        str: ответ модели.
    """
    pipe = get_pipe()
    if pipe is None:
        raise RuntimeError("Модель ещё не загружена")

    global _chat_history

    # Создаём временный список сообщений для генерации
    if use_history:
        messages = _chat_history.copy()
    else:
        messages = []

    # Добавляем новое сообщение пользователя
    messages.append({"role": "user", "content": user_text})

    # Формируем промпт с учётом всей истории
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Генерируем ответ
    output = pipe(
        prompt,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.7,
        pad_token_id=pipe.tokenizer.eos_token_id,
        generation_config=None,
        return_full_text=False,
    )
    assistant_response = output[0]['generated_text'].strip()

    # Если используем историю, сохраняем диалог
    if use_history:
        _chat_history.append({"role": "user", "content": user_text})
        _chat_history.append({"role": "assistant", "content": assistant_response})

        # Опционально: ограничиваем длину истории (чтобы не переполнить контекстное окно)
        # Например, оставляем последние 20 сообщений (10 пар)
        max_history_length = 20  # чётное число (пары user+assistant)
        if len(_chat_history) > max_history_length:
            _chat_history = _chat_history[-max_history_length:]

    return assistant_response