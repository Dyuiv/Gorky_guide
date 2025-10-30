import os
import re
import json
import argparse
import html
import math
import pandas as pd
from typing import Tuple, Optional, List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cultural_objects")

CATEGORY_MAP = {
    1: ("памятник", ["монумент", "скульптура"]),
    2: ("парк", ["сквер", "сад", "природа"]),
    3: ("тактильный макет", ["макет", "инклюзия"]),
    4: ("набережная", ["видовая точка", "панорама", "променад"]),
    5: ("историческая архитектура", ["улица", "здание", "крепость", "кремль", "наследие"]),
    6: ("культурный центр", ["дк", "библиотека", "дом культуры"]),
    7: ("музей", ["галерея", "выставка", "центр современного искусства"]),
    8: ("театр", ["филармония", "сцена", "концертный зал"]),
    9: ("прочее", ["город-партнёр", "внешний объект"]),
    10: ("монументальное искусство", ["мозаика", "панно", "стрит-арт"]),
    11: ("кофейни", ["кофе"]),
    12: ("рестораны", ["кафе","еда"])
}

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"


def strip_html(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(no_tags).split())


def parse_point_wkt(point_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if not isinstance(point_str, str):
        return None
    m = re.search(r"POINT\s*\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)", point_str)
    if not m:
        return None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lat, lon


def text_for_embedding(category_id: Optional[int], title: Optional[str], description: Optional[str]) -> str:
    """
        Формирует строку для эмбеддинга: категория, синонимы, заголовок и описание.
        Args:
            category_id (Optional[int]): Идентификатор категории.
            title (Optional[str]): Название объекта.
            description (Optional[str]): Описание (может быть HTML).
        Returns:
            str: Текстовый инпут для модели эмбеддингов.
    """
    title = title or ""
    description = strip_html(description)

    cat_name = ""
    synonyms = []
    if isinstance(category_id, (int, float)) and not math.isnan(category_id):
        category_id = int(category_id)
        if category_id in CATEGORY_MAP:
            cat_name, synonyms = CATEGORY_MAP[category_id]
        else:
            cat_name = f"категория {category_id}"
    else:
        cat_name = "без категории"

    parts = [cat_name, title]
    if synonyms:
        parts.append(" ".join(synonyms))
    if description:
        parts.append(description)
    text = " | ".join([p for p in parts if p])
    return f"passage: {text}"


def build_payload(row: pd.Series) -> Dict:
    latlon = parse_point_wkt(row.get("coordinate"))
    category_id = row.get("category_id")
    cat_name, _ = CATEGORY_MAP.get(int(category_id) if pd.notna(category_id) else -1, ("прочее", []))

    payload = {
        "src_id": int(row.get("id")) if pd.notna(row.get("id")) else None,
        "title": (row.get("title") or "").strip(),
        "address": (row.get("address") or "").strip(),
        "description": strip_html(row.get("description")),
        "coordinate": row.get("coordinate"),
        "category_id": int(category_id) if pd.notna(category_id) else None,
        "category_name": cat_name,
        "url": None if pd.isna(row.get("url")) else str(row.get("url")),
    }

    if latlon:
        lat, lon = latlon
        payload["location"] = {"lat": lat, "lon": lon}

    return payload

def recreate_and_index(xlsx_path: str, batch_size: int = 128):
    df = pd.read_excel(xlsx_path)

    texts = [
        text_for_embedding(row.get("category_id"), row.get("title"), row.get("description"))
        for _, row in df.iterrows()
    ]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dim = vectors.shape[1]

    client = QdrantClient(
        QDRANT_URL,
        timeout=60,
        prefer_grpc=False,
        check_compatibility=False,
    )

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )

    try:
        client.create_payload_index(COLLECTION_NAME, field_name="category_id",
                                    field_schema=qmodels.PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION_NAME, field_name="title",
                                    field_schema=qmodels.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION_NAME, field_name="address",
                                    field_schema=qmodels.PayloadSchemaType.TEXT)
    except Exception:
        pass

    points: List[qmodels.PointStruct] = []
    for i, (_, row) in enumerate(df.iterrows()):
        vec = vectors[i].tolist()
        payload = build_payload(row)

        pid = payload.get("src_id") if payload.get("src_id") is not None else i + 1

        points.append(
            qmodels.PointStruct(
                id=int(pid),
                vector=vec,
                payload=payload,
            )
        )

        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    info = client.get_collection(COLLECTION_NAME)
    print(f"✅ Индексация завершена. В коллекции {COLLECTION_NAME}: {info.points_count} точек.")



class Searcher:
    """
       Обёртка над моделью эмбеддингов и клиентом Qdrant для семантического поиска.
    """
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = QdrantClient(
            QDRANT_URL,
            timeout=60,
            prefer_grpc=False,
            check_compatibility=False,
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
            Выполняет семантический поиск по коллекции и возвращает top-k результатов.
            Args:
                query (str): Пользовательский запрос.
                top_k (int): Количество результатов.
            Returns:
                List[Dict]: Результаты с метаданными, расстояниями и оценками.
        """
        qvec = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        output = []
        for r in results:
            pl = r.payload or {}
            output.append({
                "title": pl.get("title"),
                "address": pl.get("address"),
                "description": pl.get("description"),
                "coordinate": pl.get("coordinate"),
                "score": r.score,
                "category": pl.get("category_name"),
            })
        return output

def main():
    parser = argparse.ArgumentParser(description="Индексация и семантический поиск по Qdrant.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Залить данные в Qdrant (пересоздаёт коллекцию).")
    p_index.add_argument("--xlsx", required=True, help="Путь к Excel (например, data/cultural_objects_mnn.xlsx)")
    p_index.add_argument("--batch-size", type=int, default=128, help="Размер батча для upsert")

    p_search = sub.add_parser("search", help="Семантический поиск.")
    p_search.add_argument("-q", "--query", required=True, help="Запрос пользователя")
    p_search.add_argument("--top-k", type=int, default=5, help="Сколько результатов вернуть")

    args = parser.parse_args()

    if args.cmd == "index":
        recreate_and_index(args.xlsx, batch_size=args.batch_size)
    elif args.cmd == "search":
        searcher = Searcher()
        hits = searcher.search(args.query, top_k=args.top_k)
        print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
