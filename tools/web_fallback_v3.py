import logging
import os
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

WEB_FALLBACK_ENABLED = os.getenv("WEB_FALLBACK_ENABLED", "false").lower() == "true"
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "").lower()
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", "")
WEB_SEARCH_LIMIT = int(os.getenv("WEB_SEARCH_LIMIT", "5"))

DEFAULT_FALLBACK_DOMAINS = [
    "gall.dcinside.com/mgallery/board",
    "tarkov.dev",
    "escapefromtarkov.fandom.com",
]

WEB_FALLBACK_DOMAINS = [
    domain.strip()
    for domain in os.getenv(
        "WEB_FALLBACK_DOMAINS",
        ",".join(DEFAULT_FALLBACK_DOMAINS),
    ).split(",")
    if domain.strip()
]

COMMUNITY_DOMAINS = ("gall.dcinside.com",)


@dataclass(frozen=True)
class WebFallbackResult:
    title: str
    url: str
    snippet: str
    source: str
    is_community: bool


def _domain_filter_query(query: str, domains: list[str]) -> str:
    site_filters = " OR ".join(f"site:{domain}" for domain in domains)
    return f"{query} ({site_filters})"


def _source_from_url(url: str) -> tuple[str, bool]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return host, any(domain in host for domain in COMMUNITY_DOMAINS)


def _allowed_url(url: str, domains: list[str]) -> bool:
    parsed = urlparse(url)
    host_path = f"{parsed.netloc}{parsed.path}".lower()
    return any(domain.lower() in host_path for domain in domains)


async def search_web_fallback_v3(
    query: str,
    domains: list[str] | None = None,
    limit: int = WEB_SEARCH_LIMIT,
) -> list[WebFallbackResult]:
    if not WEB_FALLBACK_ENABLED:
        log.info("[web_fallback_v3] disabled")
        return []

    domains = domains or WEB_FALLBACK_DOMAINS
    if WEB_SEARCH_PROVIDER == "brave":
        return await _search_brave(query, domains, limit)
    if WEB_SEARCH_PROVIDER == "tavily":
        return await _search_tavily(query, domains, limit)
    if WEB_SEARCH_PROVIDER == "public":
        return await _search_public_sites(query, domains, limit)

    log.warning("[web_fallback_v3] unsupported provider=%s", WEB_SEARCH_PROVIDER)
    return []


async def _search_public_sites(
    query: str,
    domains: list[str],
    limit: int,
) -> list[WebFallbackResult]:
    results: list[WebFallbackResult] = []
    async with httpx.AsyncClient(
        timeout=20.0,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        follow_redirects=True,
    ) as client:
        if any("gall.dcinside.com" in domain for domain in domains):
            results.extend(await _search_dcinside(client, query, limit - len(results)))

    log.info("[web_fallback_v3] public query=%s results=%s", query[:40], len(results))
    return results[:limit]


async def _search_dcinside(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
) -> list[WebFallbackResult]:
    if limit <= 0:
        return []

    url = "https://gall.dcinside.com/mgallery/board/lists/"
    params = {
        "id": "eft",
        "s_type": "search_subject_memo",
        "s_keyword": query,
    }
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("[web_fallback_v3] dcinside search failed: %s", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[WebFallbackResult] = []
    seen: set[str] = set()
    for row in soup.select("tr.ub-content.us-post"):
        title_link = row.select_one("td.gall_tit a[href*='no=']")
        if not title_link:
            continue
        title = title_link.get_text(" ", strip=True)
        href = title_link.get("href") or ""
        article_url = urljoin("https://gall.dcinside.com", href)
        if article_url in seen:
            continue
        seen.add(article_url)
        snippet = await _fetch_dcinside_snippet(client, article_url)
        source, is_community = _source_from_url(article_url)
        results.append(
            WebFallbackResult(
                title=title,
                url=article_url,
                snippet=snippet or title,
                source=source,
                is_community=is_community,
            )
        )
        if len(results) >= limit:
            break

    return results


async def _fetch_dcinside_snippet(
    client: httpx.AsyncClient,
    url: str,
    max_chars: int = 800,
) -> str:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for selector in (".write_div", ".writing_view_box", ".gallview_contents"):
        content = soup.select_one(selector)
        if content:
            text = " ".join(content.get_text(" ", strip=True).split())
            return text[:max_chars]
    return ""


async def _search_brave(
    query: str,
    domains: list[str],
    limit: int,
) -> list[WebFallbackResult]:
    if not WEB_SEARCH_API_KEY:
        log.warning("[web_fallback_v3] missing WEB_SEARCH_API_KEY for brave")
        return []

    params = {
        "q": _domain_filter_query(query, domains),
        "count": min(max(limit, 1), 10),
        "search_lang": "ko",
        "country": "KR",
    }
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": WEB_SEARCH_API_KEY,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    results: list[WebFallbackResult] = []
    for item in data.get("web", {}).get("results", []):
        url = item.get("url") or ""
        if not url or not _allowed_url(url, domains):
            continue
        source, is_community = _source_from_url(url)
        results.append(
            WebFallbackResult(
                title=item.get("title") or url,
                url=url,
                snippet=item.get("description") or "",
                source=source,
                is_community=is_community,
            )
        )
        if len(results) >= limit:
            break

    log.info("[web_fallback_v3] brave query=%s results=%s", query[:40], len(results))
    return results


async def _search_tavily(
    query: str,
    domains: list[str],
    limit: int,
) -> list[WebFallbackResult]:
    if not WEB_SEARCH_API_KEY:
        log.warning("[web_fallback_v3] missing WEB_SEARCH_API_KEY for tavily")
        return []

    payload = {
        "api_key": WEB_SEARCH_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": limit,
        "include_domains": domains,
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post("https://api.tavily.com/search", json=payload)
        resp.raise_for_status()
        data = resp.json()

    results: list[WebFallbackResult] = []
    for item in data.get("results", []):
        url = item.get("url") or ""
        if not url or not _allowed_url(url, domains):
            continue
        source, is_community = _source_from_url(url)
        results.append(
            WebFallbackResult(
                title=item.get("title") or url,
                url=url,
                snippet=item.get("content") or "",
                source=source,
                is_community=is_community,
            )
        )
        if len(results) >= limit:
            break

    log.info("[web_fallback_v3] tavily query=%s results=%s", query[:40], len(results))
    return results
