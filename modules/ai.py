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
from typing import Callable, Optional

# ----------------------------------------------------------------------
# 1. Глобальные переменные
# ----------------------------------------------------------------------
_model = None
_tokenizer = None
_chat_history = []

# ----------------------------------------------------------------------
# 2. Инструменты
# ----------------------------------------------------------------------
def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "По вашему запросу ничего не найдено."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Ошибка при поиске: {e}"

def get_current_datetime(query: str = "") -> str:
    moscow_tz = pytz.timezone("Europe/Moscow")
    now = datetime.now(moscow_tz)
    return now.strftime("Сейчас %d.%m.%Y, %H:%M (Москва).")

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Ищет актуальную информацию в интернете. "
                "ОБЯЗАТЕЛЬНО используй для ЛЮБЫХ вопросов, связанных с: "
                "документами, маркировкой, Честным знаком, законами, требованиями, "
                "сертификатами, лицензиями, регуляторными нормами, ФНС, ФСС, Роспотребнадзором, "
                "отчётностью, кассами, ЭДО, ЭЦП, ФГИС, ЕГАИС, Меркурием, "
                "налогами, взносами, штрафами, проверками, изменениями в законодательстве, "
                "погодой, новостями, курсами валют, патчами, событиями после 2023 года "
                "и любой другой информацией, которая могла измениться. "
                "ЗАПРЕЩЕНО отвечать по памяти на такие вопросы — всегда ищи свежие данные."
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
# 3. Загрузка модели с колбэком прогресса
# ----------------------------------------------------------------------
def load_model(progress_callback: Optional[Callable[[str, float], None]] = None):
    """
    Загружает модель. progress_callback(message, percent) вызывается на каждом этапе.
    percent — число от 0.0 до 1.0.
    """
    global _model, _tokenizer

    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    def _cb(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)

    _cb("Подготовка конфигурации...", 0.05)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {0: "10GB", "cpu": "16GB"}

    _cb("Загрузка токенизатора...", 0.15)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    _cb("Загрузка весов модели (это займёт несколько минут)...", 0.30)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.bfloat16,
    )

    _cb("Перенос модели на устройство...", 0.90)
    # небольшой прогрев — первый проход чтобы CUDA скомпилировала графы
    dummy = _tokenizer("тест", return_tensors="pt").to(_model.device)
    with torch.no_grad():
        _model.generate(**dummy, max_new_tokens=1)

    _cb("Готово!", 1.0)
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
# 5. Основная функция с колбэком статуса
# ----------------------------------------------------------------------
def generate_response(
    user_text: str,
    use_history: bool = True,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    status_callback(message) — вызывается при смене этапа обработки,
    чтобы UI мог показать пользователю что происходит.
    """
    global _model, _tokenizer, _chat_history

    if _model is None or _tokenizer is None:
        raise RuntimeError("Модель не загружена. Сначала вызовите load_model().")

    def _status(msg: str):
        if status_callback:
            status_callback(msg)

    # ---------- Мгновенный ответ для вопросов о дате/времени ----------
    time_keywords = [
        "год", "дата", "число", "день недели", "время",
        "который час", "сегодня", "сейчас"
    ]
    if any(kw in user_text.lower() for kw in time_keywords):
        _status("Получаю текущее время...")
        direct_answer = get_current_datetime()
        if use_history:
            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": direct_answer})
        return direct_answer

    # ---------- Подготовка сообщений ----------
    system_msg = {
        "role": "system",
        "content": (
            "Ты — умный и терпеливый помощник для малого бизнеса в России. "
            "Твоя аудитория — предприниматели и владельцы небольших магазинов, складов, "
            "кафе и других малых предприятий, которые впервые сталкиваются с такими понятиями "
            "как Честный знак, маркировка товаров, ЭДО, кассы, налоги, отчётность и другие "
            "регуляторные требования. Они не имеют специального юридического или бухгалтерского "
            "образования и нуждаются в простых, понятных объяснениях без сложного жаргона. "
            "\n\n"
            "ГЛАВНОЕ ПРАВИЛО — АКТУАЛЬНОСТЬ ИНФОРМАЦИИ:\n"
            "Твои внутренние знания могут быть устаревшими. Законы, требования, сроки, "
            "штрафы и процедуры постоянно меняются. Поэтому:\n"
            "- НИКОГДА не отвечай по памяти на вопросы о документах, маркировке, "
            "Честном знаке, законах, налогах, сертификатах, проверках, ЭДО, кассах, "
            "отчётности, штрафах и любых регуляторных требованиях.\n"
            "- ВСЕГДА вызывай search_web для таких вопросов, чтобы дать актуальный ответ.\n"
            "- Если не уверен, нужен ли поиск — всё равно выполни его.\n"
            "\n"
            "КАК ОТВЕЧАТЬ:\n"
            "- Объясняй простым языком, как будто говоришь с человеком, который впервые "
            "слышит этот термин.\n"
            "- Разбивай ответ на чёткие шаги: что нужно сделать, в каком порядке, куда идти.\n"
            "- Если тема сложная — сначала коротко объясни суть, потом детали.\n"
            "- Всегда указывай источник информации (сайт, из которого взяты данные).\n"
            "- Предупреждай о штрафах и рисках, если они есть.\n"
            "- Если вопрос касается Честного знака — объясни что это такое с нуля, "
            "какие товары подлежат маркировке, как зарегистрироваться и начать работу.\n"
        )
    }

    messages = [system_msg]
    if use_history:
        messages.extend(_chat_history)
    messages.append({"role": "user", "content": user_text})

    force_search_keywords = [
        "честный знак", "маркировк", "честныйзнак", "chestny znak",
        "кмц", "гисмт", "datamatrix", "data matrix",
        "эдо", "эцп", "электронн", "фгис", "егаис", "меркури",
        "закон", "постановлени", "приказ", "норм", "требовани",
        "сертификат", "лицензи", "разрешени", "аккредитаци",
        "налог", "ндс", "ндфл", "усн", "осн", "патент", "взнос",
        "отчётност", "декларац", "фнс", "ифнс", "пфр", "фсс",
        "штраф", "проверк", "роспотребнадзор", "прокурат",
        "касс", "онлайн-касс", "54-фз", "офд",
        "погода", "патч", "обновлени", "релиз", "новост",
        "курс", "валют", "доллар", "евро", "крипто",
        "инструкци", "порядок", "процедур", "регистрац",
        "оформ", "подключ", "как работ", "как начать",
    ]

    max_tool_calls = 5
    final_response = None
    search_count = 0

    for iteration in range(max_tool_calls):
        _status("Думаю над ответом...")

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

        if "<tool_call>" in response_text:
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
                final_response = (
                    response_text
                    .replace("<tool_call>", "")
                    .replace("</tool_call>", "")
                    .replace("<|im_end|>", "")
                    .strip()
                )
                break

            for call in tool_calls:
                func_name = call.get("name", "")
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except:
                        arguments = {}

                if func_name == "search_web":
                    search_count += 1
                    query = arguments.get("query", "")
                    _status(f"Ищу в интернете: «{query[:50]}»...")
                    result = search_web(query)
                    _status("Анализирую результаты поиска...")
                elif func_name == "get_current_datetime":
                    _status("Получаю текущее время...")
                    query = arguments.get("query", "")
                    result = get_current_datetime(query)
                else:
                    result = f"Неизвестный инструмент: {func_name}"

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

            continue

        else:
            user_lower = user_text.lower()
            need_force_search = any(kw in user_lower for kw in force_search_keywords)

            if need_force_search and iteration == 0:
                search_query = user_text.strip()
                if "честный знак" in user_lower or "маркировк" in user_lower:
                    search_query = f"Честный знак маркировка {search_query} {datetime.now().year}"
                elif any(kw in user_lower for kw in ["закон", "налог", "штраф", "касс"]):
                    search_query = f"{search_query} {datetime.now().year} актуально"

                _status(f"Ищу в интернете: «{search_query[:50]}»...")
                result = search_web(search_query)
                _status("Анализирую результаты поиска...")

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
                continue
            else:
                final_response = response_text.strip().replace("<|im_end|>", "")
                break

    if final_response is None:
        final_response = "Извините, произошла ошибка при обработке запроса."

    _status("Формирую ответ...")

    if use_history:
        _chat_history.append({"role": "user", "content": user_text})
        _chat_history.append({"role": "assistant", "content": final_response})
        if len(_chat_history) > 20:
            _chat_history = _chat_history[-20:]

    return final_response