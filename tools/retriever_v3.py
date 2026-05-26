import json
import logging
import os
import argparse
import asyncio
import re
from collections.abc import Iterable
from dotenv import load_dotenv

from db.connection import get_pool
from db.connection import close_pool
from schemas.models_v3 import Domain, Lang, RagDocumentV3
from tools.domain_router_v3 import infer_domain_boosts_v3, infer_domain_filter_v3
from tools.embedder import get_embedding

load_dotenv()

log = logging.getLogger(__name__)

VECTOR_THRESHOLD = float(os.getenv("RAG_V3_VECTOR_THRESHOLD", "0.45"))
TRGM_THRESHOLD = float(os.getenv("RAG_TRGM_THRESHOLD", "0.08"))
NAME_EXPANSION_ENABLED = os.getenv("RAG_V3_NAME_EXPANSION", "true").lower() == "true"
NAME_EXPANSION_THRESHOLD = float(os.getenv("RAG_V3_NAME_EXPANSION_THRESHOLD", "0.22"))
NAME_EXPANSION_MAX_TERMS = int(os.getenv("RAG_V3_NAME_EXPANSION_MAX_TERMS", "8"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
ANSWER_CHUNK_LIMIT = int(os.getenv("RAG_ANSWER_CHUNK_LIMIT", "12"))
DOMAIN_ROUTE_BOOST_ENABLED = os.getenv("RAG_V3_DOMAIN_ROUTE_BOOST", "true").lower() == "true"
AUTO_DOMAIN_FILTER_ENABLED = os.getenv("RAG_V3_AUTO_DOMAIN_FILTER", "true").lower() == "true"

SEARCHABLE_CHUNK_TYPES = ("identifier", "retrieval", "summary", "relation")
ANSWER_CHUNK_TYPES = ("content", "relation", "guide", "summary", "retrieval")
LEXICAL_OR_STOPWORDS = {
    "아이템",
    "템",
    "퀘스트",
    "보스",
    "맵",
    "지도",
    "제작",
    "만들",
    "만드는",
    "만들어",
    "만들어줘",
    "필요",
    "필요한",
    "요구",
    "어디",
    "어디서",
    "어디에",
    "어디씀",
    "어디써",
    "어떻게",
    "써",
    "씀",
    "해",
    "함",
    "드랍",
    "잡",
    "잡는",
    "잡아",
    "잡기",
    "사살",
    "사살하기",
    "처치",
    "처치하기",
    "제거",
    "제거하기",
    "보상",
    "item",
    "items",
    "quest",
    "quests",
    "boss",
    "map",
    "craft",
    "crafts",
    "needed",
    "required",
    "where",
    "how",
    "정보",
    "알려줘",
    "알려",
    "내용",
}

HANGUL_INITIALS = (
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
)
HANGUL_MEDIALS = (
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
)
HANGUL_FINALS = (
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lk",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "t",
)


def _normalize_query_for_search(query: str) -> str:
    normalized = query.strip()
    replacements = [
        r"\s*(정보\s*)?알려\s*줘요?\??$",
        r"\s*정보\s*$",
        r"\s*내용\s*$",
    ]
    for pattern in replacements:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE).strip()
    return normalized or query


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def _entity_name_matches_query(metadata: dict, query: str) -> bool:
    query_compact = _compact_text(query)
    if not query_compact:
        return False
    names = [
        str(metadata.get("entity_name") or ""),
        str(metadata.get("title") or ""),
    ]
    for name in names:
        name_compact = _compact_text(name)
        if len(name_compact) < 2:
            continue
        if name_compact in query_compact or query_compact in name_compact:
            return True
    return False


def _lexical_or_query(query: str) -> str:
    tokens = re.findall(r"[0-9A-Za-z가-힣ぁ-ゟァ-ヿ一-龯]+", query)
    terms = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        if token.lower() in LEXICAL_OR_STOPWORDS:
            continue
        terms.append(f"{token}:*")
    if not terms:
        for token in tokens:
            token = token.strip()
            if len(token) >= 2:
                terms.append(f"{token}:*")
    return " | ".join(dict.fromkeys(terms))


def _romanize_hangul_token(token: str) -> str:
    parts = []
    for char in token:
        code = ord(char)
        if not 0xAC00 <= code <= 0xD7A3:
            if char.isascii() and char.isalnum():
                parts.append(char.lower())
            continue
        syllable = code - 0xAC00
        initial = syllable // 588
        medial = (syllable % 588) // 28
        final = syllable % 28
        parts.append(
            HANGUL_INITIALS[initial]
            + HANGUL_MEDIALS[medial]
            + HANGUL_FINALS[final]
        )
    return "".join(parts)


def _name_expansion_terms(query: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", query)
    terms = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        token_key = token.lower()
        if token_key in LEXICAL_OR_STOPWORDS:
            continue
        if re.search(r"[가-힣]", token):
            romanized = _romanize_hangul_token(token)
            if len(romanized) >= 3:
                terms.append(romanized)
                # ㄹ is ambiguous in Korean-to-Latin user searches. This helps
                # "살레와" match "salewa" in addition to strict "salrewa".
                terms.append(re.sub(r"lr(?=[aeiou])", "l", romanized))
                terms.append(romanized.replace("r", ""))
                loanword = romanized.replace("eu", "")
                terms.append(loanword)
                z_loanword = loanword.replace("j", "z")
                terms.append(z_loanword)
                terms.append(z_loanword.replace("zlri", "zzly"))
                terms.append(z_loanword.replace("zli", "zzly"))
        elif len(token) >= 3:
            terms.append(token_key)

    cleaned = []
    for term in terms:
        term = re.sub(r"[^0-9a-z]+", " ", term.lower()).strip()
        if len(term) >= 3 and term not in cleaned:
            cleaned.append(term)
        if len(cleaned) >= NAME_EXPANSION_MAX_TERMS:
            break
    return cleaned


def _entity_key(row) -> tuple[str, str]:
    return row["domain"], row["entity_id"]


def _reciprocal_rank_fusion(
    ranked_lists: Iterable[list],
    k: int = 60,
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for rows in ranked_lists:
        deduped_rows = []
        seen = set()
        for row in rows:
            key = _entity_key(row)
            if key in seen:
                continue
            seen.add(key)
            deduped_rows.append(row)

        for rank, row in enumerate(deduped_rows):
            key = _entity_key(row)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)
    return scores


def _max_score_by_entity(rows: list) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        key = _entity_key(row)
        score = round(float(row["score"]), 4)
        scores[key] = max(scores.get(key, 0.0), score)
    return scores


async def search_rag_v3(
    query: str,
    lang: Lang = "ko",
    limit: int = int(os.getenv("RAG_LIMIT", "10")),
    domain: Domain | None = None,
) -> list[RagDocumentV3]:
    search_query = _normalize_query_for_search(query)
    auto_domain: Domain | None = None
    if AUTO_DOMAIN_FILTER_ENABLED and not domain:
        auto_domain = infer_domain_filter_v3(query)
        if auto_domain:
            log.info("[retriever_v3] auto domain query=%s domain=%s", query[:40], auto_domain)
            domain = auto_domain

    embedding = await get_embedding(search_query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    lexical_or_query = _lexical_or_query(search_query)
    name_expansion_terms = _name_expansion_terms(search_query)
    candidate_limit = max(limit * 4, 20)

    pool = await get_pool()
    async with pool.acquire() as conn:
        vector_params = [
            embedding_str,
            lang,
            list(SEARCHABLE_CHUNK_TYPES),
            candidate_limit,
        ]
        lexical_params = [
            search_query,
            lang,
            list(SEARCHABLE_CHUNK_TYPES),
            candidate_limit,
            lexical_or_query,
        ]
        trigram_params = [
            search_query,
            lang,
            list(SEARCHABLE_CHUNK_TYPES),
            candidate_limit,
            TRGM_THRESHOLD,
        ]
        vector_domain_clause = ""
        lexical_domain_clause = ""
        trigram_domain_clause = ""
        if domain:
            vector_domain_clause = "AND domain = $5"
            lexical_domain_clause = "AND domain = $6"
            trigram_domain_clause = "AND domain = $6"
            vector_params.append(domain)
            lexical_params.append(domain)
            trigram_params.append(domain)

        vector_rows = await conn.fetch(
            f"""
            SELECT
                domain, entity_id, chunk_id, chunk_type,
                1 - (embedding <=> $1::vector) AS score
            FROM rag_documents_v3
            WHERE lang = $2
              AND is_active
              AND searchable
              AND chunk_type = ANY($3::text[])
              {vector_domain_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            *vector_params,
        )

        lexical_rows = await conn.fetch(
            f"""
            SELECT
                domain, entity_id, chunk_id, chunk_type,
                (
                    ts_rank_cd(search_vector, plainto_tsquery('simple', $1))
                    + ts_rank_cd(search_vector, to_tsquery('simple', $5))
                ) AS score
            FROM rag_documents_v3
            WHERE lang = $2
              AND is_active
              AND searchable
              AND chunk_type = ANY($3::text[])
              AND (
                  search_vector @@ plainto_tsquery('simple', $1)
                  OR search_vector @@ to_tsquery('simple', $5)
              )
              {lexical_domain_clause}
            ORDER BY score DESC
            LIMIT $4
            """,
            *lexical_params,
        )

        trigram_rows = await conn.fetch(
            f"""
            SELECT
                domain, entity_id, chunk_id, chunk_type,
                similarity(content, $1) AS score
            FROM rag_documents_v3
            WHERE lang = $2
              AND is_active
              AND searchable
              AND chunk_type = ANY($3::text[])
              AND similarity(content, $1) > $5
              {trigram_domain_clause}
            ORDER BY score DESC
            LIMIT $4
            """,
            *trigram_params,
        )

        name_expansion_rows = []
        if NAME_EXPANSION_ENABLED and name_expansion_terms and domain in (None, "item"):
            name_expansion_rows = await conn.fetch(
                """
                WITH terms(term) AS (
                    SELECT * FROM unnest($1::text[])
                ),
                item_scores AS (
                    SELECT
                        i.id AS entity_id,
                        max(
                            greatest(
                                similarity(coalesce(i.name_en, ''), terms.term),
                                similarity(coalesce(i.name_ko, ''), terms.term),
                                similarity(coalesce(i.name_ja, ''), terms.term),
                                similarity(replace(coalesce(i.normalized_name, ''), '-', ' '), terms.term),
                                similarity(coalesce(i.normalized_name, ''), terms.term)
                            )
                        ) AS score
                    FROM items i
                    CROSS JOIN terms
                    GROUP BY i.id
                )
                SELECT
                    'item'::text AS domain,
                    entity_id,
                    'name_expansion'::text AS chunk_id,
                    'identifier'::text AS chunk_type,
                    score
                FROM item_scores
                WHERE score >= $2
                ORDER BY score DESC
                LIMIT $3
                """,
                name_expansion_terms,
                NAME_EXPANSION_THRESHOLD,
                candidate_limit,
            )

        rrf_scores = _reciprocal_rank_fusion(
            [
                list(vector_rows),
                list(lexical_rows),
                list(trigram_rows),
                list(name_expansion_rows),
            ],
            k=RRF_K,
        )
        route_boosts = {}
        if DOMAIN_ROUTE_BOOST_ENABLED and not domain:
            route_boosts = infer_domain_boosts_v3(query)
            if route_boosts:
                log.info("[retriever_v3] domain boosts query=%s boosts=%s", query[:40], route_boosts)
                rrf_scores = {
                    key: score * route_boosts.get(key[0], 1.0)
                    for key, score in rrf_scores.items()
                }

        if not rrf_scores:
            log.info("[retriever_v3] no candidates query=%s", query[:40])
            return []

        vector_score_map = _max_score_by_entity(list(vector_rows))
        lexical_score_map = _max_score_by_entity(list(lexical_rows))
        trigram_score_map = _max_score_by_entity(list(trigram_rows))
        name_expansion_score_map = _max_score_by_entity(list(name_expansion_rows))

        filtered_keys = [
            key
            for key, _score in sorted(
                rrf_scores.items(), key=lambda item: item[1], reverse=True
            )
            if vector_score_map.get(key, 0.0) >= VECTOR_THRESHOLD
            or lexical_score_map.get(key, 0.0) > 0.0
            or trigram_score_map.get(key, 0.0) > 0.0
            or name_expansion_score_map.get(key, 0.0) > 0.0
        ][:limit]

        if not filtered_keys:
            log.info("[retriever_v3] low confidence candidates query=%s", query[:40])
            return []

        domains = [key[0] for key in filtered_keys]
        entity_ids = [key[1] for key in filtered_keys]

        rows = await conn.fetch(
            """
            WITH selected(domain, entity_id, ord) AS (
                SELECT * FROM unnest($1::text[], $2::text[]) WITH ORDINALITY
                    AS t(domain, entity_id, ord)
            )
            SELECT
                d.domain, d.entity_id, d.chunk_id, d.chunk_type, d.lang,
                d.content, d.metadata, selected.ord
            FROM rag_documents_v3 d
            JOIN selected
              ON selected.domain = d.domain
             AND selected.entity_id = d.entity_id
            WHERE d.lang = $3
              AND d.is_active
              AND d.chunk_type = ANY($4::text[])
            ORDER BY
                selected.ord,
                CASE d.chunk_type
                    WHEN 'relation' THEN 1
                    WHEN 'content' THEN 2
                    WHEN 'guide' THEN 3
                    WHEN 'summary' THEN 4
                    WHEN 'retrieval' THEN 5
                    ELSE 9
                END,
                d.chunk_id
            LIMIT $5
            """,
            domains,
            entity_ids,
            lang,
            list(ANSWER_CHUNK_TYPES),
            ANSWER_CHUNK_LIMIT,
        )

    def _row_sort_key(row) -> tuple[int, int, int, str]:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        exact_entity_match = _entity_name_matches_query(metadata, search_query)
        chunk_order = {
            "relation": 1,
            "content": 2,
            "guide": 3,
            "summary": 4,
            "retrieval": 5,
        }.get(row["chunk_type"], 9)
        return (
            0 if exact_entity_match else 1,
            int(row["ord"]),
            chunk_order,
            row["chunk_id"],
        )

    rows = sorted(rows, key=_row_sort_key)

    results: list[RagDocumentV3] = []
    for row in rows:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        key = (row["domain"], row["entity_id"])
        results.append(
            RagDocumentV3(
                domain=row["domain"],
                entity_id=row["entity_id"],
                chunk_id=row["chunk_id"],
                chunk_type=row["chunk_type"],
                lang=row["lang"],
                content=row["content"],
                metadata=metadata,
                similarity=vector_score_map.get(key),
                lexical_score=lexical_score_map.get(key),
                trigram_score=trigram_score_map.get(key),
                fused_score=round(rrf_scores.get(key, 0.0), 6),
            )
        )

    log.info(
        "[retriever_v3] query=%s vector=%s lexical=%s trigram=%s name_expansion=%s terms=%s entities=%s chunks=%s",
        query[:40],
        len(vector_rows),
        len(lexical_rows),
        len(trigram_rows),
        len(name_expansion_rows),
        name_expansion_terms,
        len(filtered_keys),
        len(results),
    )
    return results


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Search V3 RAG documents.")
    parser.add_argument("query")
    parser.add_argument("--lang", default="ko", choices=["ko", "en", "ja"])
    parser.add_argument("--domain", choices=["item", "quest", "map", "boss", "hideout", "trader", "information", "story"])
    parser.add_argument("--limit", type=int, default=int(os.getenv("RAG_LIMIT", "10")))
    args = parser.parse_args()

    try:
        docs = await search_rag_v3(
            query=args.query,
            lang=args.lang,
            limit=args.limit,
            domain=args.domain,
        )
        for i, doc in enumerate(docs, 1):
            name = doc.metadata.get("entity_name", "")
            print(
                f"\n[{i}] {doc.domain}/{doc.entity_id} "
                f"{doc.chunk_type}:{doc.chunk_id} "
                f"vector={doc.similarity} lexical={doc.lexical_score} "
                f"trgm={doc.trigram_score} fused={doc.fused_score}"
            )
            if name:
                print(f"name: {name}")
            print(doc.content[:1200])
    finally:
        await close_pool()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(_main())
