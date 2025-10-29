# walking_route_local_osrm.py
import time
from typing import List, Tuple, Optional
import requests
import folium
from folium.plugins import PolyLineTextPath


Coord = Tuple[float, float]  # (lon, lat)

OSRM = "http://localhost:5000"
DEFAULT_PROFILE = "foot"
REQUEST_TIMEOUT = 20
RETRY_COUNT = 3
RETRY_BACKOFF = 0.6

def http_get_json(url: str, params: Optional[dict] = None) -> dict:
    sess = requests.Session()
    last_err = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            r = sess.get(url, params=params, timeout=REQUEST_TIMEOUT)
            # мягкая обработка 429/5xx
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt == RETRY_COUNT:
                break
            time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))
    raise RuntimeError(f"GET {url} failed after {RETRY_COUNT} retries: {last_err}")

def osrm_table(points: List[Coord], profile: str = DEFAULT_PROFILE):
    """
        Запрашивает у OSRM матрицы длительностей и дистанций между всеми парами точек.
        Args:
            points (List[Coord]): Список (lon, lat).
            profile (str): Профиль OSRM (напр., 'foot').
        Returns:
            Tuple[List[List[float]], List[List[float]]]: Матрицы durations и distances.
    """
    assert len(points) >= 2
    coords = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in points])
    url = f"{OSRM}/table/v1/{profile}/{coords}"
    params = {"annotations": "duration,distance"}
    data = http_get_json(url, params=params)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table error: {data.get('code')} {data.get('message')}")
    return data["durations"], data["distances"]

def osrm_route(a: Coord, b: Coord, profile: str = DEFAULT_PROFILE):
    """
        Строит один маршрут OSRM между двумя точками и возвращает длительность, дистанцию и геометрию.
        Args:
            a (Coord): Старт (lon, lat).
            b (Coord): Финиш (lon, lat).
            profile (str): Профиль OSRM.
        Returns:
            Tuple[float, float, List[Tuple[float, float]]]: (сек., метры, полилиния lat/lon).
    """
    coords = f"{a[0]:.6f},{a[1]:.6f};{b[0]:.6f},{b[1]:.6f}"
    url = f"{OSRM}/route/v1/{profile}/{coords}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false", "alternatives": "false"}
    data = http_get_json(url, params=params)
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM route error: {data.get('code')} {data.get('message')}")
    route = data["routes"][0]
    line_lonlat = route["geometry"]["coordinates"]
    line_latlon = [(lat, lon) for lon, lat in line_lonlat]
    return route["duration"], route["distance"], line_latlon

def path_cost(path: List[int], M: List[List[float]], roundtrip: bool) -> float:
    """
    Считает суммарную «стоимость» пути по матрице (в секундах/единицах), с опцией возврата.
    Args:
        path (List[int]): Перестановка индексов.
        M (List[List[float]]): Матрица стоимостей.
        roundtrip (bool): Возврат к старту.
    Returns:
        float: Суммарная стоимость.
    """
    c = 0.0
    for i in range(len(path) - 1):
        c += M[path[i]][path[i + 1]] or 0.0
    if roundtrip and len(path) > 1:
        c += M[path[-1]][path[0]] or 0.0
    return c

def nearest_neighbor(M: List[List[float]], start_idx: int = 0) -> List[int]:
    """
        Жадная эвристика ближайшего соседа для построения начального порядка обхода.
        Args:
            M (List[List[float]]): Матрица (времён/дистанций).
            start_idx (int): Индекс стартовой вершины.
        Returns:
            List[int]: Начальный маршрут.
    """
    n = len(M)
    unv = set(range(n))
    path = [start_idx]; unv.remove(start_idx)
    while unv:
        last = path[-1]
        nxt = min(unv, key=lambda j: float("inf") if M[last][j] is None else M[last][j])
        path.append(nxt); unv.remove(nxt)
    return path

def two_opt(path: List[int], M: List[List[float]], roundtrip: bool, max_iter: int = 2000) -> List[int]:
    """
    Улучшает маршрут 2-opt перестановками рёбер, пока находится улучшение или не достигнут лимит итераций.
    Args:
        path (List[int]): Исходная перестановка.
        M (List[List[float]]): Матрица стоимостей.
        roundtrip (bool): Учитывать ли ребро возврата.
        max_iter (int): Ограничение итераций.
    Returns:
        List[int]: Улучшенный порядок обхода.
    """
    best = path[:]
    improved = True; it = 0
    while improved and it < max_iter:
        improved = False; it += 1
        end_i = len(best) - (0 if roundtrip else 1) - 1
        for i in range(1, max(1, end_i)):
            for j in range(i + 1, len(best) - (0 if roundtrip else 0)):
                a, b = best[i - 1], best[i]
                c, d = best[j - 1], best[j % len(best)]
                cur = (M[a][b] or 0) + (M[c][d] or 0)
                alt = (M[a][c] or 0) + (M[b][d] or 0)
                if alt + 1e-9 < cur:
                    best[i:j] = reversed(best[i:j])
                    improved = True
    return best

def build_walking_route(
    points: List[Coord],
    start_idx: int = 0,
    roundtrip: bool = False,
    pace_scale: float = 1.0,
    stop_minutes: float = 0.0,
    map_zoom: int = 13,
    outfile: Optional[str] = "walking_route.html",
    duration_mode: str = "osrm",
    *,
    point_labels: Optional[List[str]] = None,
    node_roles: Optional[List[str]] = None,
    dwell_minutes_per_node: Optional[List[float]] = None,
    show_leg_labels: bool = False,
    show_direction_arrows: bool = True,
    add_legend: bool = True,
    fixed_order: Optional[List[int]] = None,
):
    """
    Строит пеший маршрут: вычисляет порядок, тянет OSRM-линии, рисует карту Folium с маркерами, стрелками и легендой.
    Args:
        points (List[Coord]): Список координат (lon, lat), первый — старт.
        start_idx (int): Индекс точки старта.
        roundtrip (bool): Замыкать ли маршрут на старт.
        pace_scale (float): Масштаб темпа (влияет на скорость/время).
        stop_minutes (float): Общий стоп (не используется при переданных dwell_minutes_per_node).
        map_zoom (int): Начальный зум карты.
        outfile (Optional[str]): Путь сохранения HTML-файла карты.
        duration_mode (str): 'osrm' или 'distance' для расчёта времени.
        point_labels (Optional[List[str]]): Подписи точек.
        node_roles (Optional[List[str]]): Роли узлов ('start'/'main'/'additional') для цвета и стоянок.
        dwell_minutes_per_node (Optional[List[float]]): Минуты остановок по узлам.
        show_leg_labels (bool): Рисовать ли подписи сегментов.
        show_direction_arrows (bool): Рисовать ли стрелки направления.
        add_legend (bool): Добавлять ли блок легенды.
        fixed_order (Optional[List[int]]): Задать фиксированный порядок обхода (начинается со старта).
    Returns:
        Dict[str, Any]: Информация о маршруте (порядок, сегменты, суммарные длительности/дистанции, карта).
    """
    assert len(points) >= 2, "Нужно минимум 2 точки."

    target_kmh = 5.0 / max(pace_scale, 1e-6)
    target_mps = target_kmh * 1000.0 / 3600.0

    if fixed_order is not None:
        assert fixed_order[0] == start_idx, "fixed_order должен начинаться со старта"
        order = fixed_order[:]
    else:
        durations, distances = osrm_table(points, profile=DEFAULT_PROFILE)

        if duration_mode == "distance":
            durations_for_tsp = [
                [(distances[i][j] or 0.0) / max(target_mps, 1e-6) for j in range(len(points))]
                for i in range(len(points))
            ]
        else:
            durations_for_tsp = durations

        base = nearest_neighbor(durations_for_tsp, start_idx=start_idx)
        order = two_opt(base, durations_for_tsp, roundtrip=roundtrip, max_iter=1500)

    leg_infos = []
    polyline = []
    seq = order + ([order[0]] if roundtrip else [])
    total_walk_sec = 0.0
    total_stop_sec = 0.0
    total_m = 0.0

    for i in range(len(seq) - 1):
        a, b = points[seq[i]], points[seq[i + 1]]
        dur_osrm, dist_m, line = osrm_route(a, b, profile=DEFAULT_PROFILE)

        if polyline and line:
            polyline.extend(line[1:])
        else:
            polyline.extend(line)

        if duration_mode == "distance":
            walk_sec = dist_m / max(target_mps, 1e-6)
        else:
            walk_sec = dur_osrm * pace_scale

        total_walk_sec += walk_sec
        total_m += dist_m

        is_last_leg = (i == len(seq) - 2)
        stop_sec = 0.0
        if not is_last_leg and dwell_minutes_per_node:
            idx_to = seq[i + 1]
            dwell_min = max(
                0.0,
                dwell_minutes_per_node[idx_to] if 0 <= idx_to < len(dwell_minutes_per_node) else 0.0
            )
            stop_sec = dwell_min * 60.0
        total_stop_sec += stop_sec

        leg_infos.append({
            "from_idx": seq[i],
            "to_idx": seq[i + 1],
            "distance_km": dist_m / 1000.0,
            "walk_min": walk_sec / 60.0,
            "stop_min_after": stop_sec / 60.0,
            "cum_min_total": (total_walk_sec + total_stop_sec) / 60.0
        })

    clat = sum(lat for _, lat in points) / len(points)
    clon = sum(lon for lon, _ in points) / len(points)
    fmap = folium.Map(
        location=(clat, clon),
        zoom_start=map_zoom,
        control_scale=True,
        attr=None
    )
    fmap.get_root().html.add_child(folium.Element(
        "<style>.leaflet-control-attribution {display: none !important;}</style>"
    ))

    def marker_style(role: str):
        if role == "start":
            return dict(color="#2ecc71", fill=True, fill_color="#2ecc71", radius=12)
        if role == "main":
            return dict(color="#ff0000", fill=True, fill_color="#ff0000", radius=12)
        return dict(color="#8E44AD", fill=True, fill_color="#8E44AD", radius=10)

    for i, idx in enumerate(order):
        lon, lat = points[idx]
        name = (point_labels[idx] if point_labels and 0 <= idx < len(point_labels) else None)
        title = f"{name or 'Точка'} — точка {i+1}"
        role = node_roles[idx] if (node_roles and 0 <= idx < len(node_roles)) else ("start" if idx == 0 else "additional")
        style = marker_style(role)
        folium.CircleMarker(location=(lat, lon), tooltip=title, **style).add_to(fmap)

    line_layer = folium.PolyLine(polyline, weight=5, opacity=0.9)
    line_layer.add_to(fmap)
    if show_direction_arrows:
        PolyLineTextPath(
            line_layer,
            '▶',
            repeat=True,
            offset=7,
            attributes={'font-size': '8px', 'font-weight': 'bold', 'opacity': 0.8}
        ).add_to(fmap)

    if show_leg_labels:
        for leg in leg_infos:
            a = points[leg["from_idx"]]
            b = points[leg["to_idx"]]
            mid = ((a[1] + b[1]) / 2, (a[0] + b[0]) / 2)
            txt = (f'{leg["distance_km"]:.2f} км, '
                   f'ходьба {leg["walk_min"]:.0f} мин'
                   + (f' + стоп {leg["stop_min_after"]:.0f} мин' if leg["stop_min_after"] > 0 else '')
                   + f' (∑ {leg["cum_min_total"]:.0f} мин)')
            folium.Marker(mid, icon=folium.DivIcon(html=f'<div style="font-size:10px">{txt}</div>')).add_to(fmap)

    total_sec = total_walk_sec + total_stop_sec

    if add_legend:
        def label_for(idx: int) -> str:
            return (point_labels[idx] if point_labels and 0 <= idx < len(point_labels) else f"Точка {idx}")

        main_lines, add_lines = [], []
        for i, idx in enumerate(order if not roundtrip else order + [order[0]]):
            role = node_roles[idx] if (node_roles and 0 <= idx < len(node_roles)) else (
                "start" if idx == 0 else "additional")
            if idx == 0:
                continue
            (main_lines if role == "main" else add_lines).append(f'#{i + 1}: {label_for(idx)}')

        dwell_main_min = 0.0
        dwell_add_min = 0.0
        for leg in leg_infos:
            idx_to = leg["to_idx"]
            role_to = node_roles[idx_to] if (node_roles and 0 <= idx_to < len(node_roles)) else "additional"
            m = float(leg["stop_min_after"] or 0.0)  # уже минуты
            if role_to == "main":
                dwell_main_min += m
            else:
                dwell_add_min += m

        dwell_main_min = int(round(dwell_main_min))
        dwell_add_min = int(round(dwell_add_min))
        total_dwell_min = dwell_main_min + dwell_add_min

        leg_lines = [f'#{i + 1}→#{i + 2}: {leg["distance_km"]:.2f} км, {leg["walk_min"]:.0f} мин'
                     + (f' + осмотр {leg["stop_min_after"]:.0f} мин' if leg["stop_min_after"] > 0 else '')
                     for i, leg in enumerate(leg_infos)]

        legend_html = f"""
            <div style="
                position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: rgba(255,255,255,0.95); padding: 10px 12px;
                border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
                max-height: 46vh; overflow:auto; font-size: 12px; line-height: 1.35;
                ">
                <div style="font-weight:600; margin-bottom:6px;">Легенда маршрута</div>
                <div>Всего: {total_m / 1000:.2f} км • ходьба ~{total_walk_sec / 60:.0f} мин • осмотр ~{total_dwell_min} мин</div>
                <div style="margin:4px 0 6px 0; color:#888;">Основные (15 мин): {dwell_main_min} мин • Дополнительные (3 мин): {dwell_add_min} мин</div>
                <hr style="margin:6px 0;">
                <div style="margin-bottom:4px;"><b>Основные точки:</b><br>{"<br>".join(main_lines) or "—"}</div>
                <div style="margin-bottom:4px;"><b>Дополнительные точки:</b><br>{"<br>".join(add_lines) or "—"}</div>
                <div><b>Переходы:</b><br>{"<br>".join(leg_lines)}</div>
            </div>
            """
        fmap.get_root().html.add_child(folium.Element(legend_html))

    if outfile:
        fmap.save(outfile)
        print(f"Карта сохранена: {outfile}")

    print(f"Порядок обхода: {order}{' + возврат' if roundtrip else ''}")
    print(f"Итого: {total_m/1000:.2f} км, ходьба ~{total_walk_sec/3600:.2f} ч, "
          f"осмотры ~{total_stop_sec/3600:.2f} ч, всего ~{(total_sec)/3600:.2f} ч "
          f"(режим: {duration_mode}, pace_scale={pace_scale:.2f}, скорость≈{target_kmh:.2f} км/ч)")

    return {
        "order": order,
        "legs": leg_infos,
        "total_km": total_m / 1000.0,
        "total_walk_hours": total_walk_sec / 3600.0,
        "total_stop_hours": total_stop_sec / 3600.0,
        "total_hours": (total_sec) / 3600.0,
        "map": fmap
    }

