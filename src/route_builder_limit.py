from typing import List, Tuple, Optional, Dict, Any
from math import inf

from src.geo_search import (
    semantic_proximity_rerank,
    semantic_topk,
    _haversine_km,
    _make_key,
    make_stable_keys,
)
from src.route_bulder_foot import (
    osrm_table, build_walking_route,
    nearest_neighbor, two_opt, DEFAULT_PROFILE
)

Coord = Tuple[float, float]  # (lon, lat)

def _sum_time_minutes(order: List[int],
                      durations_sec: List[List[float]],
                      roundtrip: bool,
                      pace_scale: float,
                      *,
                      dwell_minutes_per_node: Optional[List[float]] = None) -> float:
    """
        Суммирует время пешего перемещения и остановок по заданному порядку обхода.
        Args:
            order (List[int]): Перестановка индексов узлов.
            durations_sec (List[List[float]]): Матрица времен (сек) между узлами.
            roundtrip (bool): Возврат к старту в конце.
            pace_scale (float): Масштаб темпа (1.0 — базовый).
            dwell_minutes_per_node (Optional[List[float]]): Минуты «осмотра» на узел (кроме финального).
        Returns:
            float: Общее время в минутах.
    """
    if len(order) < 2:
        return 0.0

    walk_sec = 0.0
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        v = durations_sec[a][b] or 0.0
        walk_sec += v

    if roundtrip:
        walk_sec += durations_sec[order[-1]][order[0]] or 0.0

    walk_min = (walk_sec * pace_scale) / 60.0

    stop_min = 0.0
    if dwell_minutes_per_node:
        for i in range(len(order) - 1):
            node = order[i]
            stop_min += max(0.0, dwell_minutes_per_node[node])

    return walk_min + stop_min

def _promote_required(rows: List[Dict], required_keys: List[str]) -> List[Dict]:
    req_set = set(required_keys)
    def key_of(r: Dict) -> str:
        return _make_key({
            "src_id": r.get("src_id"),
            "title": r.get("title"),
            "location": {"lat": r.get("lat"), "lon": r.get("lon")}
        })

    a, b = [], []
    for r in rows:
        (a if key_of(r) in req_set else b).append(r)
    return a + b

async def plan_route_under_budget(
    *,
    query: str,
    user_lat: float,
    user_lon: float,
    time_budget_hours: float,
    category_ids: Optional[List[int]] = None,
    fetch_k: int = 80,
    alpha: float = 0.65,
    geo_tau_km: float = 1.2,
    roundtrip: bool = False,
    pace_scale: float = 1.0,
    stop_minutes: float = 0.0,
    duration_mode: str = "osrm",
    map_zoom: int = 14,
    outfile: Optional[str] = "/routes/walking_route.html",
    max_pois: Optional[int] = None,
    main_semantic_k: int = 5,
    pin_radius_km: float = 8.0,
) -> Dict[str, Any]:
    """
        Строит маршрут, укладывающийся во временной бюджет: выбирает точки, решает порядок (NN+2-opt) и рендерит карту.
        Args:
            query (str): Интересы/поисковый запрос.
            user_lat (float), user_lon (float): Стартовые координаты.
            time_budget_hours (float): Доступное время (часы).
            category_ids (Optional[List[int]]): Фильтр категорий.
            fetch_k (int): Сколько кандидатов собрать до отбора.
            alpha (float), geo_tau_km (float): Параметры реранжирования семантика/гео.
            roundtrip (bool): Замкнутый маршрут к старту.
            pace_scale (float): Масштаб темпа ходьбы (влияет на скорость).
            stop_minutes (float): Зарезервировано; фактические стоянки задаются по ролям.
            duration_mode (str): 'osrm' или 'distance' для расчёта длительности.
            map_zoom (int): Масштаб карты Folium.
            outfile (Optional[str]): Путь сохранения HTML-карты.
            max_pois (Optional[int]): Жёсткое ограничение числа POI.
            main_semantic_k (int): Сколько «семантических главных» выбирать.
            pin_radius_km (float): Радиус, в котором главные продвигаются в ранге.
        Returns:
            Dict[str, Any]: Итог с выбранными POI, ролями узлов, порядком, оценками и сервисной инфой.
    """
    main_semantic = semantic_topk(
        query=query,
        top_k=main_semantic_k,
        fetch_k=max(fetch_k, 300),
        category_ids=category_ids
    )
    main_keys = make_stable_keys(main_semantic)

    rows = semantic_proximity_rerank(
        query=query,
        user_lat=user_lat,
        user_lon=user_lon,
        top_k=fetch_k,
        fetch_k=fetch_k,
        alpha=alpha,
        geo_tau_km=geo_tau_km,
        category_ids=category_ids,
        hard_drop_km= 30
    )
    if not rows:
        return {"selected_poi": [], "reason": "no_candidates", "main_semantic": main_semantic}

    promoted_keys: List[str] = []
    for ms in main_semantic:
        d = _haversine_km(user_lat, user_lon, ms["lat"], ms["lon"])
        if d is not None and d <= pin_radius_km:
            promoted_keys.append(_make_key({
                "src_id": ms.get("src_id"),
                "title": ms.get("title"),
                "location": {"lat": ms.get("lat"), "lon": ms.get("lon")}
            }))

    if promoted_keys:
        rows = _promote_required(rows, promoted_keys)

    if max_pois is not None:
        rows = rows[:max_pois]

    start: Coord = (user_lon, user_lat)
    pois: List[Coord] = [(r["lon"], r["lat"]) for r in rows]
    all_points: List[Coord] = [start] + pois

    durations_full, distances_full = osrm_table(all_points, profile=DEFAULT_PROFILE)

    best_n = 0
    best_order_local = None
    best_time_min = inf

    for n in range(1, len(all_points)):  # n = число POI
        idxs = list(range(0, n + 1))
        # подматрица (n+1) x (n+1)
        M = [
            [durations_full[i][j] if durations_full[i][j] is not None else inf for j in idxs]
            for i in idxs
        ]

        base = nearest_neighbor(M, start_idx=0)
        order_local = two_opt(base, M, roundtrip=roundtrip, max_iter=1200)

        main_count = min(5, n)
        dwell = [0.0] + [15.0] * main_count + [3.0] * (n - main_count)

        time_min = _sum_time_minutes(order_local, M, roundtrip, pace_scale, dwell_minutes_per_node=dwell)

        if time_min <= time_budget_hours * 60.0:
            best_n = n
            best_order_local = order_local[:]
            best_time_min = time_min
            continue
        else:
            break

    if best_n == 0:
        return {
            "selected_poi": [],
            "reason": "time_budget_too_small",
            "estimated_minutes": best_time_min,
            "main_semantic": main_semantic
        }

    final_points = [start] + pois[:best_n]
    selected_rows = rows[:best_n]

    selected_keys = {
        _make_key({"src_id": r.get("src_id"),
                   "title": r.get("title"),
                   "location": {"lat": r.get("lat"), "lon": r.get("lon")}})
        for r in selected_rows
    }

    point_labels = ["Старт"] + [(r["title"] or f"Место {i+1}") for i, r in enumerate(selected_rows)]

    node_roles = ["start"]
    for r in selected_rows:
        k = _make_key({"src_id": r.get("src_id"),
                       "title": r.get("title"),
                       "location": {"lat": r.get("lat"), "lon": r.get("lon")}})
        node_roles.append("main" if k in main_keys else "additional")

    dwell_minutes_per_node = [0.0] + [15.0 if role == "main" else 3.0 for role in node_roles[1:]]

    result = build_walking_route(
        points=final_points,
        start_idx=0,
        roundtrip=roundtrip,
        pace_scale=pace_scale,
        stop_minutes=0.0,
        map_zoom=map_zoom,
        outfile=outfile,
        duration_mode=duration_mode,
        point_labels=point_labels,
        node_roles=node_roles,
        dwell_minutes_per_node=dwell_minutes_per_node,
        show_leg_labels=False,
        show_direction_arrows=True,
        add_legend=True,
        fixed_order=best_order_local,
    )

    main_semantic_not_in_route = [
        ms for ms in main_semantic
        if _make_key({"src_id": ms.get("src_id"),
                      "title": ms.get("title"),
                      "location": {"lat": ms.get("lat"), "lon": ms.get("lon")}}) not in selected_keys
    ]

    result["selected_poi"] = selected_rows
    result["budget_hours"] = time_budget_hours
    result["estimated_minutes_fast"] = best_time_min
    result["reason"] = "ok_under_budget"
    result["node_roles"] = node_roles
    result["fixed_order"] = best_order_local
    result["main_semantic"] = main_semantic
    result["main_semantic_not_in_route"] = main_semantic_not_in_route
    return result
