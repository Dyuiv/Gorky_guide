from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception as e:
    raise RuntimeError(
        "Не установлен пакет google-generativeai. Установите: pip install google-generativeai"
    ) from e


def _ensure_model(model_name: str, api_key: Optional[str] = None):
    """
        Инициализирует клиент Gemini с переданным/окружным ключом и возвращает модель.
        Args:
            model_name (str): Имя модели Gemini.
            api_key (Optional[str]): Явный API-ключ; по умолчанию берётся из окружения.
        Returns:
            Any: Экземпляр genai.GenerativeModel.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден GEMINI_API_KEY/GOOGLE_API_KEY в окружении.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def get_reason(
    user_query: str,
    main_points: List[Dict[str, Any]],
    extra_points: List[Dict[str, Any]],
    *,
    language: str = "ru",
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """
        Запрашивает у Gemini краткий JSON-гид: интро, причины по основным точкам и подсказки по доп. точкам.
        Args:
            user_query (str): Исходный запрос/интересы пользователя.
            main_points (List[Dict]): Отобранные «основные» точки маршрута.
            extra_points (List[Dict]): Дополнительные точки «по пути».
            language (str): Язык ответа.
            model_name (str): Имя модели Gemini.
            temperature (float): Температура генерации.
        Returns:
            Dict[str, Any]: Структура с intro/mains/extras/outro (или отладочная info при не-JSON).
    """
    model = _ensure_model(model_name)

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )

    payload = {
        "language": language,
        "user_query": user_query,
        "main_points": [
            {
                "title": (p.get("title") or "").strip(),
                "description": (p.get("description") or "").strip(),
                "address": (p.get("address") or "").strip(),
            }
            for p in (main_points or [])
        ],
        "extra_points": [
            {
                "title": (p.get("title") or "").strip(),
                "description": (p.get("description") or "").strip(),
                "address": (p.get("address") or "").strip(),
            }
            for p in (extra_points or [])
        ],
    }

    system_instruction = f"""
Ты — гид по городской прогулке. Отвечай на {language}.
Используй ТОЛЬКО факты из предоставленных описаний объектов. Не выдумывай новых фактов.

Сформируй JSON строго по схеме:
{{
  "intro": "1–2 предложения вводного контекста под интересы пользователя",
  "mains": [
    {{"title": "...", "summary": "2–3 коротких предложения для этой основной точки"}}
  ],
  "extras": [
    {{"title": "...", "why_interesting": "1 лаконичное предложение, почему стоит заглянуть по пути"}}
  ],
  "outro": "краткое завершающее предложение"
}}

Требования к стилю:
- Плотно по сути, без воды,  не повторяй одно и то же.
- Если описания у точки почти нет, дай нейтральную причинку посетить (панорама/атмосфера/история района) без домыслов.
- Не используй markdown, эмодзи и кавычки-ёлочки.
- Возвращай ТОЛЬКО валидный JSON без преамбулы и постскриптума.
"""

    resp = model.generate_content(
        [
            system_instruction,
            "Данные для работы (JSON):",
            json.dumps(payload, ensure_ascii=False),
        ],
        generation_config=generation_config,
    )

    txt = resp.text or ""
    try:
        data = json.loads(txt)
        data.setdefault("intro", "")
        data.setdefault("mains", [])
        data.setdefault("extras", [])
        data.setdefault("outro", "")
        data["mains"] = [
            {"title": (m.get("title") or "").strip(),
             "summary": (m.get("summary") or "").strip()}
            for m in data["mains"] if (m.get("title") or "").strip()
        ]
        data["extras"] = [
            {"title": (e.get("title") or "").strip(),
             "why_interesting": (e.get("why_interesting") or "").strip()}
            for e in data["extras"] if (e.get("title") or "").strip()
        ]
        return data
    except Exception as e:
        return {
            "intro": "",
            "mains": [],
            "extras": [],
            "outro": "",
            "raw_text": txt.strip(),
            "note": f"LLM returned non-JSON: {e}",
        }


def render_guide_text(guide: Dict[str, Any], *, max_len: int = 3800) -> str:
    """
       Рендерит JSON-гид в компактный текст для Telegram с секциями «Основные» и «Дополнительные».
       Args:
           guide (Dict[str, Any]): Структура, возвращённая get_reason.
           max_len (int): Ограничение длины итогового текста.
       Returns:
           str: Готовый текст сообщения.
    """
    if not guide:
        return "Не удалось сформировать подсказки по маршруту."

    lines: List[str] = []

    intro = (guide.get("intro") or "").strip()
    if intro:
        lines.append(intro)

    mains = guide.get("mains") or []
    if mains:
        lines.append("\nОсновные точки:")
        for i, m in enumerate(mains, 1):
            title = (m.get("title") or "").strip()
            summ = (m.get("summary") or "").strip()
            lines.append(f"{i}. {title} — {summ}")

    extras = guide.get("extras") or []
    if extras:
        lines.append("\nДополнительные точки (по пути):")
        for i, e in enumerate(extras, 1):
            title = (e.get("title") or "").strip()
            why = (e.get("why_interesting") or "").strip()
            lines.append(f"• {title}: {why}")

    outro = (guide.get("outro") or "").strip()
    if outro:
        lines.append("\n" + outro)

    text = "\n".join(lines).strip()

    if len(text) > max_len:
        text = text[: max_len - 20].rstrip() + "…"

    if not text and guide.get("raw_text"):
        text = guide["raw_text"][: max_len]

    return text
