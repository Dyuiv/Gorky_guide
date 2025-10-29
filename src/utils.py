import requests

def geocode_nominatim(query: str, limit: int = 5, viewbox=None, bounded=True):
    """
    Выполняет поиск адресов/объектов через Nominatim и возвращает список координат с подписью.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "jsonv2", "limit": limit}
    if viewbox:
        params.update({"viewbox": f"{viewbox[0]},{viewbox[1]},{viewbox[2]},{viewbox[3]}",
                       "bounded": 1 if bounded else 0})
    headers = {"User-Agent": "temp"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    return [(float(it["lon"]), float(it["lat"]), it.get("display_name")) for it in data]


