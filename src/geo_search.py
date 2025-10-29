from typing import List, Dict, Optional, Iterable
from math import radians, sin, cos, asin, sqrt, exp
from qdrant_client.http import models as qmodels
from cultural_qdrant import Searcher, COLLECTION_NAME

def _haversine_km(lat1, lon1, lat2, lon2):
    """
        Вычисляет вел. окружную дистанцию между двумя точками в километрах (формула гаверсинусов).
        Args:
            lat1 (float), lon1 (float): Первая точка.
            lat2 (float), lon2 (float): Вторая точка.
        Returns:
            float: Расстояние в км.
    """
    R = 6371.0088
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlmb/2)**2
    return 2*R*asin(sqrt(a))

def _make_key(pl: dict) -> str:
    sid = pl.get("src_id")
    if sid is not None:
        return f"id:{sid}"
    t = (pl.get("title") or "").strip()
    loc = pl.get("location") or {}
    return f"title:{t}|lat:{loc.get('lat')}|lon:{loc.get('lon')}"

def make_stable_keys(items: Iterable[Dict]) -> List[str]:
    return [_make_key({"src_id": it.get("src_id"),
                       "title": it.get("title"),
                       "location": {"lat": it.get("lat"), "lon": it.get("lon")}})
            for it in items]

def semantic_proximity_rerank(
    *,
    query: str,
    user_lat: float,
    user_lon: float,
    top_k: int = 10,
    fetch_k: int = 300,
    alpha: float = 0.7,
    geo_tau_km: float = 1.2,
    category_ids: Optional[List[int]] = None,
    min_geo_weight: float = 0.0,
    hard_drop_km: Optional[float] = None
) -> List[Dict]:
    """
        Выполняет семантический поиск в Qdrant и переупорядочивает результаты с учётом близости к пользователю.
        Args:
            query (str): Запрос пользователя.
            user_lat (float), user_lon (float): Координаты старта.
            top_k (int): Сколько вернуть после реранжирования.
            fetch_k (int): Сколько вытянуть из Qdrant до реранжирования.
            alpha (float): Вес семантики vs гео (1.0 — только семантика).
            geo_tau_km (float): Декэй-длина для e^{-d/τ}.
            category_ids (Optional[List[int]]): Ограничение по категориям.
            min_geo_weight (float): Нижняя граница для гео-скоринга.
            hard_drop_km (Optional[float]): Жёсткий срез по дистанции.
        Returns:
            List[Dict]: Топ результатов с полями координат, оценок и итоговым score.
    """
    s = Searcher()

    qvec = s.model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()

    qfilter = None
    if category_ids:
        qfilter = qmodels.Filter(must=[
            qmodels.FieldCondition(
                key="category_id",
                match=qmodels.MatchAny(any=[int(c) for c in category_ids]),
            )
        ])

    hits = s.client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        query_filter=qfilter,
        limit=fetch_k,
        with_payload=True,
        with_vectors=False,
    )

    rows: List[Dict] = []
    for h in hits:
        pl = h.payload or {}
        loc = pl.get("location")
        if not loc:
            continue

        lat, lon = float(loc["lat"]), float(loc["lon"])
        d_km = _haversine_km(user_lat, user_lon, lat, lon)

        if hard_drop_km is not None and d_km > hard_drop_km:
            continue
        cos_score = float(h.score or 0.0)
        sem01 = (cos_score + 1.0) / 2.0
        if sem01 < 0.0: sem01 = 0.0
        if sem01 > 1.0: sem01 = 1.0

        geo = exp(-d_km / max(geo_tau_km, 1e-6))
        if min_geo_weight > 0.0:
            geo = max(min_geo_weight, geo)

        final = alpha * sem01 + (1.0 - alpha) * geo

        rows.append({
            "src_id": pl.get("src_id"),
            "title": pl.get("title"),
            "address": pl.get("address"),
            "description": pl.get("description"),
            "coordinate": pl.get("coordinate"),
            "lat": lat,
            "lon": lon,
            "distance_km": d_km,
            "semantic_score_cos": cos_score,
            "semantic_score01": sem01,
            "geo_score": geo,
            "final_score": final,
            "category": pl.get("category_name"),
        })

    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows[:top_k]

def semantic_topk(
    *,
    query: str,
    top_k: int = 10,
    fetch_k: int = 400,
    category_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """
        Возвращает топ-результаты по чистой семантической близости (без геоперенормировки).
        Args:
            query (str): Запрос пользователя.
            top_k (int): Итоговый размер топа.
            fetch_k (int): Лимит отдачи Qdrant до сортировки.
            category_ids (Optional[List[int]]): Фильтр по категориям.
        Returns:
            List[Dict]: Список результатов, отсортированный по semantic_score01.
    """
    s = Searcher()

    qvec = s.model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()

    qfilter = None
    if category_ids:
        qfilter = qmodels.Filter(must=[
            qmodels.FieldCondition(
                key="category_id",
                match=qmodels.MatchAny(any=[int(c) for c in category_ids]),
            )
        ])

    hits = s.client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        query_filter=qfilter,
        limit=fetch_k,
        with_payload=True,
        with_vectors=False,
    )

    rows: List[Dict] = []
    for h in hits:
        pl = h.payload or {}
        loc = pl.get("location")
        if not loc:
            continue

        lat, lon = float(loc["lat"]), float(loc["lon"])
        cos_score = float(h.score or 0.0)
        sem01 = (cos_score + 1.0) / 2.0
        sem01 = 0.0 if sem01 < 0.0 else (1.0 if sem01 > 1.0 else sem01)

        rows.append({
            "src_id": pl.get("src_id"),
            "title": pl.get("title"),
            "address": pl.get("address"),
            "description": pl.get("description"),
            "coordinate": pl.get("coordinate"),
            "lat": lat,
            "lon": lon,
            "distance_km": None,
            "semantic_score_cos": cos_score,
            "semantic_score01": sem01,
            "geo_score": None,
            "final_score": sem01,
            "category": pl.get("category_name"),
        })

    rows.sort(key=lambda r: r["semantic_score01"], reverse=True)
    return rows[:top_k]
