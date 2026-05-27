import warnings
warnings.filterwarnings(
    "ignore",
    message=".*_check_is_size will be removed.*",
    category=FutureWarning,
    module="bitsandbytes"
)
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ddgs import DDGS
from datetime import datetime
import pytz

# ----------------------------------------------------------------------
# 1. Глобальные переменные
# ----------------------------------------------------------------------
_model = None
_tokenizer = None
_chat_history = []  # хранит историю диалога (без системного сообщения)

# ----------------------------------------------------------------------
# 2. Инструменты
# ----------------------------------------------------------------------
def search_web(query: str) -> str:
    """
    Выполняет поиск через DuckDuckGo и возвращает текстовую сводку.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "По вашему запросу ничего не найдено."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Ошибка при поиске: {e}"

def get_current_datetime(query: str = "") -> str:
    """
    Возвращает точное текущее время и дату в Москве.
    Можно дополнить поддержкой других городов при необходимости.
    """
    moscow_tz = pytz.timezone("Europe/Moscow")
    now = datetime.now(moscow_tz)
    return now.strftime("Сейчас %d.%m.%Y, %H:%M (Москва).")

# Описания инструментов в формате Qwen3
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Ищет актуальную информацию в интернете. "
                "Используй для запросов о погоде, новостях, патчах, курсах валют, "
                "событиях после 2023 года и любой другой информации, требующей свежих данных."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (на русском или английском)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Возвращает текущие дату и время. "
                "Используй, когда пользователь спрашивает о текущем годе, дате, "
                "дне недели, точном времени."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Уточнение (например, 'время в Москве')"
                    }
                }
            }
        }
    }
]

# ----------------------------------------------------------------------
# 3. Загрузка модели
# ----------------------------------------------------------------------
def load_model():
    """Загружает 4-битную квантованную модель Qwen3-4B-Instruct."""
    global _model, _tokenizer

    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {0: "10GB", "cpu": "16GB"}
    
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.bfloat16,
    )
    print("Модель загружена.")
    return _model, _tokenizer

# ----------------------------------------------------------------------
# 4. Управление историей
# ----------------------------------------------------------------------
def reset_history():
    global _chat_history
    _chat_history = []

def get_history():
    return _chat_history.copy()

# ----------------------------------------------------------------------
# 5. Основная функция с автовызовом инструментов и защитой от галлюцинаций
# ----------------------------------------------------------------------
def generate_response(user_text: str, use_history: bool = True) -> str:
    """
    Генерирует ответ, при необходимости вызывая search_web или get_current_datetime.
    Запросы о текущем времени/дате обрабатываются мгновенно, без участия модели.
    Для запросов о погоде, новостях, патчах поиск принудительно выполняется,
    если модель сама его не вызвала.
    """
    global _model, _tokenizer, _chat_history

    if _model is None or _tokenizer is None:
        raise RuntimeError("Модель не загружена. Сначала вызовите load_model().")

    # ---------- Мгновенный ответ для вопросов о дате/времени ----------
    time_keywords = [
        "год", "дата", "число", "день недели", "время",
        "который час", "сегодня", "сейчас"
    ]
    if any(kw in user_text.lower() for kw in time_keywords):
        direct_answer = get_current_datetime()
        if use_history:
            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": direct_answer})
        return direct_answer

    # ---------- Подготовка сообщений ----------
    system_msg = {
        "role": "system",
        "content": (
            "Ты — полезный ассистент с доступом к поиску в интернете и точному времени. "
            "Если вопрос требует актуальной информации (погода, новости, патчи, курсы валют, "
            "события после 2023 года) — обязательно используй search_web. "
            "Если вопрос о дате или времени — используй get_current_datetime. "
            "Не пытайся угадать, если не уверен."
        )
        # Внимание: ключ "tools" здесь не передаётся! Только через параметр tools в шаблоне.
    }

    messages = [system_msg]
    if use_history:
        messages.extend(_chat_history)
    messages.append({"role": "user", "content": user_text})

    # ---------- Цикл обработки tool calling ----------
    max_tool_calls = 5
    final_response = None

    # Ключевые слова, которые гарантированно требуют поиска (если модель не вызвала сама)
    force_search_keywords = [
        "погода", "патч", "обновление", "релиз", "новост",
        "курс", "валют", "доллар", "евро", "крипто"
    ]

    for iteration in range(max_tool_calls):
        # Применяем chat template, передавая инструменты отдельно
        prompt_text = _tokenizer.apply_chat_template(
            messages,
            tools=TOOLS_DEFINITION,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = _tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Проверяем, есть ли вызовы инструментов
        if "<tool_call>" in response_text:
            # Парсим tool_calls
            tool_calls = []
            parts = response_text.split("<tool_call>")
            for part in parts[1:]:
                if "</tool_call>" in part:
                    call_json_str = part.split("</tool_call>")[0].strip()
                    try:
                        call_data = json.loads(call_json_str)
                        tool_calls.append(call_data)
                    except json.JSONDecodeError:
                        print(f"Ошибка парсинга tool call: {call_json_str}")
                        continue

            if not tool_calls:
                # Нет валидных вызовов – вероятно, финальный ответ с мусором
                final_response = response_text.replace("<tool_call>", "").replace("</tool_call>", "").replace("<|im_end|>", "").strip()
                break

            # Выполняем каждый вызов
            for call in tool_calls:
                func_name = call.get("name", "")
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except:
                        arguments = {}

                if func_name == "search_web":
                    query = arguments.get("query", "")
                    result = search_web(query)
                elif func_name == "get_current_datetime":
                    query = arguments.get("query", "")
                    result = get_current_datetime(query)
                else:
                    result = f"Неизвестный инструмент: {func_name}"

                # Добавляем сообщения ассистента и результата в messages
                tool_call_id = f"call_{len(messages)}"
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False)
                            }
                        }
                    ]
                }
                tool_result_msg = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id
                }
                messages.append(assistant_tool_msg)
                messages.append(tool_result_msg)

            # Продолжаем цикл – модель получит результаты и сгенерирует финальный ответ
            continue

        else:
            # Нет <tool_call>: проверяем, не нужно ли принудительно выполнить поиск
            need_force_search = any(
                kw in user_text.lower() for kw in force_search_keywords
            )
            if need_force_search and iteration == 0:
                # Модель проигнорировала поиск – делаем его сами
                # Формируем поисковый запрос из вопроса пользователя
                search_query = user_text.strip()
                result = search_web(search_query)
                # Добавляем фиктивный tool_call и результат в messages
                tool_call_id = f"call_{len(messages)}"
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": json.dumps({"query": search_query}, ensure_ascii=False)
                            }
                        }
                    ]
                }
                tool_result_msg = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id
                }
                messages.append(assistant_tool_msg)
                messages.append(tool_result_msg)
                # Повторяем итерацию – теперь модель должна дать ответ на основе результата
                continue
            else:
                # Обычный финальный ответ
                final_response = response_text.strip().replace("<|im_end|>", "")
                break

    if final_response is None:
        final_response = "Извините, произошла ошибка при обработке запроса."

    # Сохраняем в историю (только user и assistant, без системных и tool-сообщений)
    if use_history:
        _chat_history.append({"role": "user", "content": user_text})
        _chat_history.append({"role": "assistant", "content": final_response})
        # Ограничение длины истории
        if len(_chat_history) > 20:
            _chat_history = _chat_history[-20:]

    return final_response