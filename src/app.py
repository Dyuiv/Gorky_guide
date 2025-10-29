import os
import asyncio
from typing import Optional, Tuple
import re
from datetime import datetime
from gorky_guide import get_reason, render_guide_text

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, ReplyKeyboardRemove,
    InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
)
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from aiohttp import web, ClientSession, ClientTimeout
from pathlib import Path

from dotenv import load_dotenv

from src.route_builder_limit import plan_route_under_budget
from src.utils import geocode_nominatim

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

ROUTES_DIR = os.getenv("ROUTES_DIR", "routes")
Path(ROUTES_DIR).mkdir(parents=True, exist_ok=True)

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var")
if not PUBLIC_BASE_URL or not PUBLIC_BASE_URL.startswith("https://"):
    raise RuntimeError("Set PUBLIC_BASE_URL env var with HTTPS URL, e.g. https://xxx.ngrok.io")

class BuildRoute(StatesGroup):
    interests = State()
    hours = State()
    location_method = State()
    address = State()
    coord_lon = State()
    coord_lat = State()
    geo = State()

dp = Dispatcher()

def _parse_num(s: str) -> float:
    """
        Преобразует строку в число с плавающей точкой, поддерживая запятую как десятичный разделитель.
        Args:
            s (str): Входная строка с числом (напр. "1,5" или "1.5").
        Returns:
            float: Число с плавающей точкой.
    """
    return float((s or "").strip().replace(",", "."))

async def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """
    Асинхронно геокодирует адрес через публичный Nominatim и возвращает координаты.
    Args:
        address (str): Адрес в свободной форме.
    Returns:
        Optional[Tuple[float, float]]: (lat, lon) или None, если не найдено/ошибка.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "WalkingRouteBot/1.0 (contact: your_email@example.com)"}
    timeout = ClientTimeout(total=8)
    try:
        async with ClientSession(timeout=timeout, headers=headers) as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if not data:
                    return None
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                return lat, lon
    except Exception:
        return None

def kb_start() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Создать маршрут", callback_data="start_wizard")]])

def kb_location_choice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📍 Поделиться геопозицией", callback_data="loc_geo")],
        [InlineKeyboardButton(text="🏠 Ввести адрес", callback_data="loc_addr")],
        [InlineKeyboardButton(text="🧭 Ввести координаты (долгота → широта)", callback_data="loc_coord")],
        [InlineKeyboardButton(text="🔁 Начать сначала", callback_data="restart")]
    ])

def kb_restart_only() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔁 Начать сначала", callback_data="restart")]])

async def send_step1(message: Message, state: FSMContext):
    await state.set_state(BuildRoute.interests)
    await message.answer("1/3. Напишите интересы (например: стрит-арт, история, кофейни):", reply_markup=kb_restart_only())

@dp.message(Command("start"))
async def cmd_start(m: Message, state: FSMContext):
    await state.clear()
    await m.answer("Привет! Нажмите, чтобы построить прогулку.", reply_markup=kb_start())

@dp.message(Command("restart"))
@dp.message(Command("new"))
async def cmd_restart(m: Message, state: FSMContext):
    await state.clear()
    await send_step1(m, state)

@dp.callback_query(F.data == "restart")
async def cb_restart(c, state: FSMContext):
    await state.clear()
    await send_step1(c.message, state)
    await c.answer()

@dp.callback_query(F.data == "start_wizard")
async def cb_start_wizard(c, state: FSMContext):
    await send_step1(c.message, state)
    await c.answer()

@dp.message(BuildRoute.interests)
async def ask_hours(m: Message, state: FSMContext):
    await state.update_data(interests=(m.text or "").strip())
    await state.set_state(BuildRoute.hours)
    await m.answer(
        "2/3. Сколько у вас времени на прогулку (в часах)? "
        "Можно дробное число: например 1.5",
        reply_markup=kb_restart_only()
    )

@dp.message(BuildRoute.hours)
@dp.message(BuildRoute.hours)
async def ask_location_method(m: Message, state: FSMContext):
    text = (m.text or "").strip().replace(",", ".")
    try:
        hours = float(text)
        if hours <= 0 or hours>24:
            raise ValueError
    except Exception:
        await m.answer(
            "Введите, пожалуйста, число часов (можно с дробной частью), например: 1.5",
            reply_markup=kb_restart_only()
        )
        return

    await state.update_data(hours=hours)
    await state.set_state(BuildRoute.location_method)
    await m.answer("3/3. Как хотите указать местоположение?", reply_markup=kb_location_choice())

@dp.callback_query(BuildRoute.location_method, F.data == "loc_addr")
async def choose_addr(c, state: FSMContext):
    await state.set_state(BuildRoute.address)
    await c.message.answer("Введите адрес (улица, дом, город):", reply_markup=kb_restart_only())
    await c.answer()

@dp.callback_query(BuildRoute.location_method, F.data == "loc_coord")
async def choose_coords(c, state: FSMContext):
    await state.set_state(BuildRoute.coord_lon)
    await c.message.answer("Ок. Сначала отправьте **долготу** (lon), например: 44.003111", reply_markup=kb_restart_only())
    await c.answer()

@dp.callback_query(BuildRoute.location_method, F.data == "loc_geo")
async def choose_geo(c, state: FSMContext):
    await state.set_state(BuildRoute.geo)
    await c.message.answer("Пришлите геопозицию через «📎 Скрепка → Геопозиция».", reply_markup=kb_restart_only())
    await c.answer()

@dp.message(BuildRoute.address)
async def got_address(m: Message, state: FSMContext):
    address = (m.text or "").strip()
    if not address:
        await state.set_state(BuildRoute.location_method)
        await m.answer(
            "Не похоже на адрес. Выберите способ указания местоположения:",
            reply_markup=kb_location_choice()
        )
        return

    await m.answer("Преобразую адрес в координаты…", reply_markup=ReplyKeyboardRemove())
    coords = geocode_nominatim(address)
    if not coords:
        await state.set_state(BuildRoute.location_method)
        await m.answer(
            "Не удалось найти координаты по этому адресу. Выберите другой способ или попробуйте ещё раз:",
            reply_markup=kb_location_choice()
        )
        return

    lon, lat, display_name = coords[0]
    await m.answer(f"Нашёл: {display_name}\nКоординаты: {lon:.6f}, {lat:.6f} ")
    await finalize_route(m, state, lat=lat, lon=lon)


@dp.message(BuildRoute.coord_lon)
async def got_lon(m: Message, state: FSMContext):
    try:
        lon = _parse_num(m.text)
        if not (-180.0 <= lon <= 180.0):
            raise ValueError
    except Exception:
        await m.answer("Долгота должна быть числом в диапазоне -180..180. Пример: 43.987654", reply_markup=kb_restart_only())
        return
    await state.update_data(lon=lon)
    await state.set_state(BuildRoute.coord_lat)
    await m.answer("Теперь отправьте **широту** (lat), например: 56.328437", reply_markup=kb_restart_only())

@dp.message(BuildRoute.coord_lat)
async def got_lat(m: Message, state: FSMContext):
    try:
        lat = _parse_num(m.text)
        if not (-90.0 <= lat <= 90.0):
            raise ValueError
    except Exception:
        await m.answer("Широта должна быть числом в диапазоне -90..90. Пример: 56.328437", reply_markup=kb_restart_only())
        return

    data = await state.get_data()
    lon = data.get("lon")
    if lon is None:
        await m.answer("Не нашёл сохранённую долготу. Нажмите «Начать сначала» и попробуйте ещё раз.", reply_markup=kb_restart_only())
        return

    await finalize_route(m, state, lat=lat, lon=lon)

@dp.message(BuildRoute.geo, F.location)
async def got_geo(m: Message, state: FSMContext):
    lat = m.location.latitude
    lon = m.location.longitude
    await m.answer("Спасибо, геопозиция получена.")
    await finalize_route(m, state, lat=lat, lon=lon)

@dp.message(BuildRoute.geo)
async def geo_expected(m: Message, state: FSMContext):
    await m.answer("Пожалуйста, пришлите геопозицию через «📎 Скрепка → Геопозиция», либо нажмите «Начать сначала».", reply_markup=kb_restart_only())

async def finalize_route(m: Message, state: FSMContext, *, lat: float, lon: float):
    """
        Финальный шаг: собирает параметры, строит маршрут, генерирует текст-гид и отправляет ссылку на карту.
        Args:
            m (Message): Сообщение для ответов пользователю.
            state (FSMContext): Машина состояний для чтения интересов и времени.
            lat (float): Широта старта.
            lon (float): Долгота старта.
        Returns:
            None
    """
    data = await state.get_data()
    interests = data["interests"]
    hours = data["hours"]

    u = m.from_user
    raw_name = u.username or f"{(u.first_name or '').strip()} {(u.last_name or '').strip()}".strip() or f"id{u.id}"
    safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)[:40] or f"id{u.id}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"route_{safe_user}_{ts}.html"
    outfile = Path(ROUTES_DIR) / filename

    await m.answer("Строю маршрут...")
    try:
        result = await plan_route_under_budget(
            query=interests,
            time_budget_hours=hours,
            user_lat=lat,
            user_lon=lon,
            outfile=str(outfile),
            fetch_k=50,
            alpha=0.8,
            geo_tau_km=1.2,
            roundtrip=False,
            pace_scale=1.67,
        )

    except Exception as e:
        await m.answer(f"Не удалось построить маршрут: {e}", reply_markup=kb_restart_only())
        return

    reason = result.get("reason")
    if reason != "ok_under_budget":
        if reason == "time_budget_too_small":
            est_min = result.get("estimated_minutes")  # может быть float
            hint = ""
            if est_min:
                try:
                    est_min = int(round(float(est_min)))
                    hint = f" (минимальная оценка сейчас ~{est_min} мин)"
                except Exception:
                    pass
            await m.answer(
                "Маршрут не удалось уложить во время." + hint +
                "\nЧто можно сделать:\n"
                "• Увеличить время прогулки\n"
                "• Выбрать старт ближе к Нижнему Новгороду\n"
                "• Уточнить интересы (например: «стрит-арт, набережная»)",
                reply_markup=kb_location_choice()
            )
            await state.set_state(BuildRoute.location_method)
            return

        if reason == "no_candidates":
            await m.answer(
                "Поблизости не нашлось подходящих мест по запросу.\n"
                "Попробуйте переформулировать интересы или начать ближе к центру Нижнего Новгорода.",
                reply_markup=kb_restart_only()
            )
            return

        await m.answer(
            "Не получилось построить маршрут. Попробуйте изменить параметры или начать сначала.",
            reply_markup=kb_restart_only()
        )
        return
    places = result.get("selected_poi", [])
    node_roles = result.get("node_roles",
                            []) or []

    roles_for_places = node_roles[1:1 + len(places)]

    main_points = [poi for poi, role in zip(places, roles_for_places) if role == "main"]
    extra_points = [poi for poi, role in zip(places, roles_for_places) if role != "main"]

    if not main_points and places:
        main_points = places[:min(3, len(places))]
        extra_points = places[min(3, len(places)):]
    guide = get_reason(
        user_query=interests,
        main_points=main_points,
        extra_points=extra_points,
        model_name="gemini-2.5-flash",
        temperature=0.3
    )
    guide_text = render_guide_text(guide)

    if guide_text:
        await m.answer(guide_text)
    else:
        await m.answer("ошибка")

    url = f"{PUBLIC_BASE_URL}/routes/{filename}"
    kb_open = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Открыть маршрут", web_app=WebAppInfo(url=url))],
        [InlineKeyboardButton(text="🔁 Начать сначала", callback_data="restart")]
    ])

    await m.answer("Готово! Открывайте маршрут 👇", reply_markup=kb_open)
    await state.clear()

async def create_web_app():
    app = web.Application()
    routes_dir_path = Path(ROUTES_DIR).resolve()

    async def walking_route(_):
        return web.FileResponse(routes_dir_path / "walking_route.html")

    app.add_routes([
        web.get("/walking_route", walking_route),
        web.static("/routes", str(routes_dir_path), show_index=False),
        web.get("/health", lambda _: web.Response(text="OK")),
    ])
    return app

async def main():
    bot = Bot(BOT_TOKEN)
    web_app = await create_web_app()

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()

    print(f"Web on http://{HOST}:{PORT}")

    try:
        await dp.start_polling(bot)
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
