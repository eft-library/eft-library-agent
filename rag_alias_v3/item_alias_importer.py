import argparse
import asyncio
import csv
import json
import logging
import os
import re
from collections import deque
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from db.connection import close_pool, get_pool
from rag_alias_v3.item_alias_generator import (
    DEFAULT_EXCLUDED_PARENT_CATEGORIES,
    GENERIC_ALIASES,
    _filter_aliases,
)
from rag_builders_v3.base import SUPPORTED_LANGS, lang_value
from schemas.models_v3 import Lang

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_LLM_MODEL")

COMMUNITY_STOPWORDS = {
    "ko": {
        "그리고",
        "그러면",
        "보통",
        "또는",
        "혹은",
        "라고",
        "이라고",
        "부른다",
        "부른",
        "초반",
        "후반",
        "퀘스트",
        "퀘스트에",
        "필요하다",
        "필요",
        "자주",
    },
    "en": set(),
    "ja": set(),
}


@dataclass
class ItemAliasCandidate:
    entity_id: str
    alias: str
    lang: Lang
    confidence: float
    note: str


@dataclass
class CrawledPage:
    url: str
    text: str


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _read_url_list(paths: list[Path]) -> list[str]:
    urls = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def _source_note(*, method: str, evidence_count: int, source: str | None) -> str:
    note = f"{method}; evidence_count={evidence_count}"
    if source:
        note += f"; source={source[:180]}"
    return note[:500]


def _append_note(note: str, addition: str) -> str:
    return f"{note}; {addition}"[:500]


def _official_values(row: dict, lang: Lang) -> set[str]:
    return {
        str(value).strip().lower()
        for value in [
            row.get("name_en"),
            row.get("name_ko"),
            row.get("name_ja"),
            row.get("normalized_name"),
            lang_value(row, "name", lang),
        ]
        if value
    }


def _match_keys(row: dict) -> list[str]:
    keys = []
    for value in [
        row.get("name_en"),
        row.get("name_ko"),
        row.get("name_ja"),
        row.get("normalized_name"),
    ]:
        if value:
            text = str(value).strip()
            if len(text) >= 3:
                keys.append(text.lower())
    return list(dict.fromkeys(keys))


def _candidate_tokens(text: str, lang: Lang) -> list[str]:
    if lang == "ko":
        pattern = r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9+._\-/]{1,28}[가-힣A-Za-z0-9]"
    elif lang == "ja":
        pattern = r"[ぁ-ゟァ-ヿ一-龯A-Za-z0-9][ぁ-ゟァ-ヿ一-龯A-Za-z0-9+._\-/]{1,28}[ぁ-ゟァ-ヿ一-龯A-Za-z0-9]"
    else:
        pattern = r"[A-Za-z0-9][A-Za-z0-9+._\-/]{1,28}[A-Za-z0-9]"

    tokens = []
    stopwords = COMMUNITY_STOPWORDS.get(lang, set())
    for match in re.finditer(pattern, text):
        token = _normalize_candidate_token(match.group(0), lang)
        if token and token.lower() not in stopwords:
            tokens.append(token)
    return tokens


def _normalize_candidate_token(token: str, lang: Lang) -> str:
    token = _clean_text(token.strip(" -_/.,:;()[]{}\"'`"))
    if lang != "ko":
        return token

    for _ in range(2):
        normalized = re.sub(
            r"(이라고|라고|이라|은|는|이|가|을|를|에서|에게|으로|로|에|도|만)$",
            "",
            token,
        )
        if normalized == token:
            break
        token = normalized
    return token.strip(" -_/.,:;()[]{}\"'`")


def _split_alias_phrase(phrase: str, lang: Lang) -> list[str]:
    phrase = re.sub(r"\b(?:aka|also known as|called|nickname|nicknamed)\b", " ", phrase, flags=re.IGNORECASE)
    parts = re.split(r",|/|·|ㆍ| 또는 | 혹은 | 그리고 | and | or ", phrase)
    aliases = []
    for part in parts:
        tokens = _candidate_tokens(part, lang)
        if tokens:
            aliases.append(tokens[-1])
    return aliases


def _strict_alias_tokens(context: str, key: str, lang: Lang) -> list[str]:
    escaped = re.escape(key)
    patterns = [
        rf"{escaped}\s*(?:은|는|이|가)?\s*(?:보통|흔히|줄여서|별명은|별칭은|통칭)?\s*([가-힣A-Za-z0-9+._\-/,\s]{{2,60}}?)(?:이라고|라고|이라|라\s*고|로\s*부르|으로\s*부르|라고\s*부르|aka|AKA)",
        rf"([가-힣A-Za-z0-9+._\-/,\s]{{2,60}}?)(?:이라고|라고|이라|라\s*고|로\s*부르|으로\s*부르|라고\s*부르|aka|AKA)\s*(?:하는|부르는)?\s*{escaped}",
        rf"{escaped}\s*[\(\[]\s*([가-힣][가-힣A-Za-z0-9+._\-/,\s]{{1,30}})\s*[\)\]]",
        rf"([가-힣][가-힣A-Za-z0-9+._\-/]{{1,30}})\s*[\(\[]\s*{escaped}\s*[\)\]]",
    ]

    aliases = []
    for pattern in patterns:
        for match in re.finditer(pattern, context, flags=re.IGNORECASE):
            aliases.extend(_split_alias_phrase(match.group(1), lang))
    return aliases


def _extract_from_text(
    *,
    text: str,
    items: list[dict],
    lang: Lang,
    source: str | None,
    context_chars: int,
    min_evidence: int,
    max_aliases_per_item: int,
    loose_cooccurrence: bool,
) -> list[ItemAliasCandidate]:
    lowered = text.lower()
    generic = {value.lower() for value in GENERIC_ALIASES.get(lang, set())}
    candidates: list[ItemAliasCandidate] = []

    for row in items:
        contexts: list[tuple[str, str]] = []
        for key in _match_keys(row):
            start = lowered.find(key)
            if start < 0:
                continue
            context_start = max(0, start - context_chars)
            context_end = min(len(text), start + len(key) + context_chars)
            contexts.append((key, text[context_start:context_end]))

        if not contexts:
            continue

        counts: dict[str, int] = defaultdict(int)
        official_values = _official_values(row, lang)
        for key, context in contexts:
            if loose_cooccurrence:
                tokens = _candidate_tokens(context, lang)
                method = "community text loose co-occurrence"
            else:
                tokens = _strict_alias_tokens(context, key, lang)
                method = "community text alias-pattern"
            for token in tokens:
                token_key = token.lower()
                if token_key in generic:
                    continue
                counts[token] += 1

        aliases = _filter_aliases(
            sorted(counts, key=lambda alias: (-counts[alias], len(alias), alias)),
            lang=lang,
            official_values=official_values,
            max_aliases=max_aliases_per_item,
        )

        for alias in aliases:
            evidence_count = counts.get(alias, 1)
            if evidence_count < min_evidence:
                continue
            candidates.append(
                ItemAliasCandidate(
                    entity_id=row["id"],
                    alias=alias,
                    lang=lang,
                    confidence=min(0.95, 0.45 + evidence_count * 0.1),
                    note=_source_note(
                        method=method,
                        evidence_count=evidence_count,
                        source=source,
                    ),
                )
            )

    return candidates


def _match_item(row: dict[str, str], items_by_id: dict[str, dict], items: list[dict]) -> dict | None:
    entity_id = row.get("entity_id") or row.get("item_id") or row.get("id")
    if entity_id and entity_id in items_by_id:
        return items_by_id[entity_id]

    raw_name = (
        row.get("item_name")
        or row.get("name")
        or row.get("name_en")
        or row.get("name_ko")
        or row.get("normalized_name")
    )
    if not raw_name:
        return None
    wanted = _compact(raw_name)
    for item in items:
        if wanted in {_compact(key) for key in _match_keys(item)}:
            return item
    return None


def _read_csv_candidates(
    *,
    path: Path,
    items: list[dict],
    lang: Lang,
) -> list[ItemAliasCandidate]:
    items_by_id = {item["id"]: item for item in items}
    candidates: list[ItemAliasCandidate] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = _clean_text(row.get("alias") or "")
            if not alias:
                continue
            item = _match_item(row, items_by_id, items)
            if not item:
                log.warning("[alias_import_v3] skipped csv row without matching item alias=%s", alias)
                continue
            row_lang = row.get("lang") or lang
            if row_lang not in SUPPORTED_LANGS:
                log.warning("[alias_import_v3] skipped unsupported lang=%s alias=%s", row_lang, alias)
                continue
            filtered = _filter_aliases(
                [alias],
                lang=row_lang,  # type: ignore[arg-type]
                official_values=_official_values(item, row_lang),  # type: ignore[arg-type]
                max_aliases=1,
            )
            if not filtered:
                continue
            confidence = float(row.get("confidence") or 0.75)
            source_url = row.get("source_url") or row.get("url")
            note = row.get("note") or _source_note(
                method="csv import",
                evidence_count=int(row.get("evidence_count") or 1),
                source=source_url,
            )
            candidates.append(
                ItemAliasCandidate(
                    entity_id=item["id"],
                    alias=filtered[0],
                    lang=row_lang,  # type: ignore[arg-type]
                    confidence=confidence,
                    note=note[:500],
                )
            )
    return candidates


def _json_from_llm_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _verification_prompt(
    *,
    item: dict,
    candidate: ItemAliasCandidate,
) -> list[dict[str, str]]:
    user_content = {
        "task": "Verify an Escape from Tarkov item alias candidate from community text.",
        "decision_rule": (
            "Approve only if the candidate is explicitly used as a name, nickname, "
            "abbreviation, transliteration, or search alias for the item. The candidate "
            "does not need to contain the full official item name if players would use "
            "it to refer to this item in search. For example, approve Korean "
            "transliterations of Latin brand/product names when the evidence says they "
            "refer to the item. Reject words "
            "that are merely nearby context, PC hardware terms, place names, general "
            "game words, jokes, or unrelated nouns."
        ),
        "positive_examples": [
            {
                "item_name_en": "Salewa first aid kit",
                "candidate_alias": "살레와",
                "reason": "Korean transliteration of Salewa used to refer to the item.",
            },
            {
                "item_name_en": "Salewa first aid kit",
                "candidate_alias": "살레와킷",
                "reason": "Korean community shorthand for Salewa kit.",
            },
        ],
        "negative_examples": [
            {
                "item_name_en": "PC CPU",
                "candidate_alias": "엔비디아",
                "reason": "Nearby hardware/vendor term, not an alias for PC CPU.",
            },
            {
                "item_name_en": "PC CPU",
                "candidate_alias": "프레임생성",
                "reason": "Nearby performance term, not an item alias.",
            },
        ],
        "item": {
            "id": item["id"],
            "name_en": item.get("name_en"),
            "name_ko": item.get("name_ko"),
            "name_ja": item.get("name_ja"),
            "normalized_name": item.get("normalized_name"),
            "category": item.get("category"),
            "parent_category": item.get("parent_category"),
        },
        "candidate_alias": candidate.alias,
        "evidence_note": candidate.note,
        "output_schema": {
            "approved": "boolean",
            "confidence": "number between 0 and 1",
            "reason": "short string",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict data-quality reviewer for game search aliases. "
                "Return JSON only. Do not invent aliases."
            ),
        },
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


async def _verify_candidate_with_llm(
    client: httpx.AsyncClient,
    *,
    item: dict,
    candidate: ItemAliasCandidate,
    num_ctx: int,
) -> ItemAliasCandidate | None:
    if not OLLAMA_BASE_URL:
        raise RuntimeError("OLLAMA_BASE_URL is not set")
    if not CHAT_MODEL:
        raise RuntimeError("OLLAMA_CHAT_MODEL or OLLAMA_LLM_MODEL is not set")

    payload = {
        "model": CHAT_MODEL,
        "messages": _verification_prompt(item=item, candidate=candidate),
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_ctx": num_ctx},
    }
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    data = _json_from_llm_text(content)
    approved = bool(data.get("approved"))
    confidence = float(data.get("confidence") or 0.0)
    reason = _clean_text(str(data.get("reason") or ""))[:160]
    log.info(
        "[alias_import_v3] llm verify item=%s alias=%s approved=%s confidence=%.2f reason=%s",
        candidate.entity_id,
        candidate.alias,
        approved,
        confidence,
        reason,
    )
    if not approved:
        return None
    return ItemAliasCandidate(
        entity_id=candidate.entity_id,
        alias=candidate.alias,
        lang=candidate.lang,
        confidence=min(candidate.confidence, confidence or candidate.confidence),
        note=_append_note(candidate.note, f"llm_verified=true; llm_reason={reason}"),
    )


async def verify_candidates_with_llm(
    *,
    candidates: list[ItemAliasCandidate],
    items_by_id: dict[str, dict],
    num_ctx: int,
) -> list[ItemAliasCandidate]:
    if not candidates:
        return []

    verified: list[ItemAliasCandidate] = []
    async with httpx.AsyncClient() as client:
        for candidate in candidates:
            item = items_by_id.get(candidate.entity_id)
            if not item:
                log.warning("[alias_import_v3] llm verify skipped missing item=%s", candidate.entity_id)
                continue
            try:
                verified_candidate = await _verify_candidate_with_llm(
                    client,
                    item=item,
                    candidate=candidate,
                    num_ctx=num_ctx,
                )
            except Exception as exc:
                log.warning(
                    "[alias_import_v3] llm verify failed item=%s alias=%s error=%s",
                    candidate.entity_id,
                    candidate.alias,
                    exc,
                )
                continue
            if verified_candidate:
                verified.append(verified_candidate)

    log.info(
        "[alias_import_v3] llm verified candidates=%s/%s",
        len(verified),
        len(candidates),
    )
    return verified


async def _fetch_url_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return _clean_text(soup.get_text(" "))


def _normalize_url(url: str) -> str:
    normalized, _fragment = urldefrag(url.strip())
    return normalized


def _host(url: str) -> str:
    return urlparse(url).netloc.lower()


def _allowed_crawl_url(url: str, allowed_domains: set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _allowed_crawl_path(url: str, allowed_path_prefixes: tuple[str, ...]) -> bool:
    if not allowed_path_prefixes:
        return True
    path = urlparse(url).path or "/"
    return any(path.startswith(prefix) for prefix in allowed_path_prefixes)


async def _fetch_robots(
    client: httpx.AsyncClient,
    robots_cache: dict[str, RobotFileParser | None],
    url: str,
) -> RobotFileParser | None:
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root in robots_cache:
        return robots_cache[root]

    parser = RobotFileParser()
    parser.set_url(f"{root}/robots.txt")
    try:
        response = await client.get(f"{root}/robots.txt", timeout=10.0)
        if response.status_code >= 400:
            robots_cache[root] = None
            return None
        parser.parse(response.text.splitlines())
    except httpx.HTTPError as exc:
        log.warning("[alias_import_v3] robots lookup failed url=%s error=%s", root, exc)
        robots_cache[root] = None
        return None

    robots_cache[root] = parser
    return parser


async def _robots_allowed(
    client: httpx.AsyncClient,
    robots_cache: dict[str, RobotFileParser | None],
    url: str,
    user_agent: str,
) -> bool:
    parser = await _fetch_robots(client, robots_cache, url)
    return True if parser is None else parser.can_fetch(user_agent, url)


async def _fetch_crawl_page(client: httpx.AsyncClient, url: str) -> tuple[str, list[str]]:
    response = await client.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and "text" not in content_type:
        return "", []

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    links = []
    for link in soup.find_all("a", href=True):
        next_url = _normalize_url(urljoin(str(response.url), link["href"]))
        if next_url:
            links.append(next_url)

    return _clean_text(soup.get_text(" ")), links


async def crawl_pages(
    *,
    seeds: list[str],
    allowed_domains: set[str],
    allowed_path_prefixes: tuple[str, ...],
    max_pages: int,
    delay_seconds: float,
) -> list[CrawledPage]:
    if not seeds or max_pages <= 0:
        return []

    allowed_domains = set(allowed_domains)
    if not allowed_domains:
        allowed_domains = {_host(seed) for seed in seeds if _host(seed)}

    queue = deque(_normalize_url(seed) for seed in seeds if seed.strip())
    seen = set(queue)
    pages: list[CrawledPage] = []
    user_agent = "eft-library-agent alias importer"
    robots_cache: dict[str, RobotFileParser | None] = {}

    async with httpx.AsyncClient(
        headers={"User-Agent": user_agent},
        follow_redirects=True,
    ) as client:
        while queue and len(pages) < max_pages:
            url = queue.popleft()
            if not _allowed_crawl_url(url, allowed_domains):
                log.info("[alias_import_v3] crawl skipped outside allowed domains url=%s", url)
                continue
            if not _allowed_crawl_path(url, allowed_path_prefixes):
                log.info("[alias_import_v3] crawl skipped outside allowed paths url=%s", url)
                continue
            if not await _robots_allowed(client, robots_cache, url, user_agent):
                log.info("[alias_import_v3] crawl skipped by robots url=%s", url)
                continue

            try:
                text, links = await _fetch_crawl_page(client, url)
            except httpx.HTTPError as exc:
                log.warning("[alias_import_v3] crawl fetch failed url=%s error=%s", url, exc)
                continue

            if text:
                pages.append(CrawledPage(url=url, text=text))
                log.info("[alias_import_v3] crawled page=%s/%s url=%s", len(pages), max_pages, url)

            for link in links:
                if len(seen) >= max(max_pages * 20, 1000):
                    break
                if link in seen:
                    continue
                if not _allowed_crawl_url(link, allowed_domains):
                    continue
                if not _allowed_crawl_path(link, allowed_path_prefixes):
                    continue
                seen.add(link)
                queue.append(link)

            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    return pages


async def fetch_items(
    conn,
    *,
    limit: int | None,
    excluded_parent_categories: set[str],
) -> list[dict]:
    where = ""
    args: list[Any] = []
    if excluded_parent_categories:
        args.append(sorted(excluded_parent_categories))
        where = """
        WHERE (parent_category IS NULL OR parent_category <> ALL($1::text[]))
          AND (category IS NULL OR category <> ALL($1::text[]))
        """

    limit_clause = "LIMIT $1" if limit is not None else ""
    if limit is not None:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    rows = await conn.fetch(
        f"""
        SELECT id, parent_category, category, name_en, name_ko, name_ja, normalized_name
        FROM items
        {where}
        ORDER BY update_time DESC, id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def upsert_import_candidates(conn, candidates: list[ItemAliasCandidate]) -> int:
    count = 0
    for candidate in candidates:
        await conn.execute(
            """
            INSERT INTO rag_entity_aliases_v3 (
                domain, entity_id, lang, alias, source, status, confidence, note
            )
            VALUES ('item', $1, $2, $3, 'import', 'pending', $4, $5)
            ON CONFLICT (domain, entity_id, lang, alias)
            DO UPDATE SET
                source = EXCLUDED.source,
                confidence = GREATEST(
                    COALESCE(rag_entity_aliases_v3.confidence, 0),
                    COALESCE(EXCLUDED.confidence, 0)
                ),
                note = EXCLUDED.note,
                updated_at = now()
            WHERE rag_entity_aliases_v3.status = 'pending'
            """,
            candidate.entity_id,
            candidate.lang,
            candidate.alias,
            candidate.confidence,
            candidate.note,
        )
        count += 1
    return count


async def import_item_alias_candidates(
    *,
    lang: Lang,
    csv_paths: list[Path],
    file_paths: list[Path],
    url_list_paths: list[Path],
    urls: list[str],
    crawl_seeds: list[str],
    crawl_allowed_domains: set[str],
    crawl_allowed_path_prefixes: tuple[str, ...],
    crawl_max_pages: int,
    crawl_delay_seconds: float,
    dry_run: bool,
    limit_items: int | None,
    excluded_parent_categories: set[str],
    context_chars: int,
    min_evidence: int,
    max_aliases_per_item: int,
    loose_cooccurrence: bool,
    llm_verify: bool,
    llm_verify_num_ctx: int,
) -> int:
    pool = await get_pool()
    all_candidates: list[ItemAliasCandidate] = []
    async with pool.acquire() as conn:
        items = await fetch_items(
            conn,
            limit=limit_items,
            excluded_parent_categories=excluded_parent_categories,
        )
        log.info(
            "[alias_import_v3] loaded items=%s excluded_parent_categories=%s",
            len(items),
            sorted(excluded_parent_categories),
        )

        urls = [*urls, *_read_url_list(url_list_paths)]

        for path in csv_paths:
            candidates = _read_csv_candidates(path=path, items=items, lang=lang)
            log.info("[alias_import_v3] csv=%s candidates=%s", path, len(candidates))
            all_candidates.extend(candidates)

        for path in file_paths:
            text = path.read_text(encoding="utf-8")
            candidates = _extract_from_text(
                text=text,
                items=items,
                lang=lang,
                source=str(path),
                context_chars=context_chars,
                min_evidence=min_evidence,
                max_aliases_per_item=max_aliases_per_item,
                loose_cooccurrence=loose_cooccurrence,
            )
            log.info("[alias_import_v3] file=%s candidates=%s", path, len(candidates))
            all_candidates.extend(candidates)

        if urls:
            async with httpx.AsyncClient(
                headers={"User-Agent": "eft-library-agent alias importer"}
            ) as client:
                for url in urls:
                    text = await _fetch_url_text(client, url)
                    candidates = _extract_from_text(
                        text=text,
                        items=items,
                        lang=lang,
                        source=url,
                        context_chars=context_chars,
                        min_evidence=min_evidence,
                        max_aliases_per_item=max_aliases_per_item,
                        loose_cooccurrence=loose_cooccurrence,
                    )
                    log.info("[alias_import_v3] url=%s candidates=%s", url, len(candidates))
                    all_candidates.extend(candidates)

        if crawl_seeds:
            pages = await crawl_pages(
                seeds=crawl_seeds,
                allowed_domains=crawl_allowed_domains,
                allowed_path_prefixes=crawl_allowed_path_prefixes,
                max_pages=crawl_max_pages,
                delay_seconds=crawl_delay_seconds,
            )
            for page in pages:
                candidates = _extract_from_text(
                    text=page.text,
                    items=items,
                    lang=lang,
                    source=page.url,
                    context_chars=context_chars,
                    min_evidence=min_evidence,
                    max_aliases_per_item=max_aliases_per_item,
                    loose_cooccurrence=loose_cooccurrence,
                )
                log.info(
                    "[alias_import_v3] crawled url=%s candidates=%s",
                    page.url,
                    len(candidates),
                )
                all_candidates.extend(candidates)

        deduped: dict[tuple[str, Lang, str], ItemAliasCandidate] = {}
        for candidate in all_candidates:
            key = (candidate.entity_id, candidate.lang, candidate.alias)
            previous = deduped.get(key)
            if previous is None or candidate.confidence > previous.confidence:
                deduped[key] = candidate

        candidates = list(deduped.values())
        if llm_verify:
            candidates = await verify_candidates_with_llm(
                candidates=candidates,
                items_by_id={item["id"]: item for item in items},
                num_ctx=llm_verify_num_ctx,
            )

        for candidate in candidates[:50]:
            log.info(
                "[alias_import_v3] candidate item=%s lang=%s alias=%s confidence=%.2f note=%s",
                candidate.entity_id,
                candidate.lang,
                candidate.alias,
                candidate.confidence,
                candidate.note,
            )

        if dry_run:
            log.info("[alias_import_v3] dry-run candidates=%s", len(candidates))
            return len(candidates)

        stored = await upsert_import_candidates(conn, candidates)
        log.info("[alias_import_v3] stored pending candidates=%s", stored)
        return stored


def _parse_lang(raw: str) -> Lang:
    if raw not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang: {raw}")
    return raw  # type: ignore[return-value]


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Import pending item alias candidates from CSV, text files, or public URLs."
    )
    parser.add_argument("--lang", default="ko", choices=["ko", "en", "ja"])
    parser.add_argument("--csv", action="append", default=[], help="CSV with alias and item_id/name columns.")
    parser.add_argument("--file", action="append", default=[], help="Plain text or HTML text file to scan.")
    parser.add_argument("--url", action="append", default=[], help="Public URL to scan.")
    parser.add_argument(
        "--url-list",
        action="append",
        default=[],
        help="Text file containing one public URL per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--crawl-seed",
        action="append",
        default=[],
        help="Seed URL for bounded same-domain crawling.",
    )
    parser.add_argument(
        "--crawl-allowed-domain",
        action="append",
        default=[],
        help="Allowed crawl domain. Defaults to seed hosts when omitted.",
    )
    parser.add_argument(
        "--crawl-allowed-path-prefix",
        action="append",
        default=[],
        help="Allowed URL path prefix for crawling, such as /mgallery/board/.",
    )
    parser.add_argument("--crawl-max-pages", type=int, default=20)
    parser.add_argument("--crawl-delay-seconds", type=float, default=0.5)
    parser.add_argument("--limit-items", type=int, help="Limit item rows loaded from platform DB.")
    parser.add_argument(
        "--include-excluded-parent-categories",
        action="store_true",
        help="Do not apply the default parent_category exclusion list.",
    )
    parser.add_argument("--context-chars", type=int, default=180)
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--max-aliases-per-item", type=int, default=5)
    parser.add_argument(
        "--loose-cooccurrence",
        action="store_true",
        help="Use broad nearby-token extraction. This is noisy and should only be used for exploration.",
    )
    parser.add_argument(
        "--llm-verify",
        action="store_true",
        help="Use Ollama chat model to verify extracted candidates before dry-run output or pending insert.",
    )
    parser.add_argument(
        "--llm-verify-num-ctx",
        type=int,
        default=int(os.getenv("NUM_CTX", "8192")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        await import_item_alias_candidates(
            lang=_parse_lang(args.lang),
            csv_paths=[Path(path) for path in args.csv],
            file_paths=[Path(path) for path in args.file],
            url_list_paths=[Path(path) for path in args.url_list],
            urls=args.url,
            crawl_seeds=args.crawl_seed,
            crawl_allowed_domains={domain.lower() for domain in args.crawl_allowed_domain},
            crawl_allowed_path_prefixes=tuple(args.crawl_allowed_path_prefix),
            crawl_max_pages=args.crawl_max_pages,
            crawl_delay_seconds=args.crawl_delay_seconds,
            dry_run=args.dry_run,
            limit_items=args.limit_items,
            excluded_parent_categories=set()
            if args.include_excluded_parent_categories
            else DEFAULT_EXCLUDED_PARENT_CATEGORIES,
            context_chars=args.context_chars,
            min_evidence=args.min_evidence,
            max_aliases_per_item=args.max_aliases_per_item,
            loose_cooccurrence=args.loose_cooccurrence,
            llm_verify=args.llm_verify,
            llm_verify_num_ctx=args.llm_verify_num_ctx,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
