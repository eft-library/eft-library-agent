import logging
import os
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

WEB_FALLBACK_ENABLED = os.getenv("WEB_FALLBACK_ENABLED", "false").lower() == "true"
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "").lower()
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", "")
WEB_SEARCH_LIMIT = int(os.getenv("WEB_SEARCH_LIMIT", "5"))
WEB_FALLBACK_STEAM_APP_ID = os.getenv("WEB_FALLBACK_STEAM_APP_ID", "3932890")
WEB_FALLBACK_STEAM_COUNTRY = os.getenv("WEB_FALLBACK_STEAM_COUNTRY", "KR")
WEB_FALLBACK_STEAM_LANG = os.getenv("WEB_FALLBACK_STEAM_LANG", "korean")
WEB_PUBLIC_SEARCH_ENABLED = os.getenv("WEB_PUBLIC_SEARCH_ENABLED", "true").lower() == "true"

DEFAULT_FALLBACK_DOMAINS = [
    "store.steampowered.com/app/3932890",
    "www.escapefromtarkov.com/news",
    "escapefromtarkov.com/news",
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
STEAM_STORE_KEYWORDS = (
    "steam",
    "스팀",
    "구매",
    "살수",
    "살 수",
    "가격",
    "얼마",
    "할인",
)
TARKOV_KEYWORDS = (
    "tarkov",
    "타르코프",
    "eft",
    "이스케이프 프롬 타르코프",
)


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
    if _is_steam_store_query(query):
        steam_domains = [*domains, DEFAULT_FALLBACK_DOMAINS[0]]
        steam_result = await _fetch_steam_store_price(steam_domains)
        if steam_result:
            return [steam_result][:limit]

    if WEB_SEARCH_PROVIDER == "brave":
        return await _search_brave(query, domains, limit)
    if WEB_SEARCH_PROVIDER == "tavily":
        return await _search_tavily(query, domains, limit)
    if WEB_SEARCH_PROVIDER == "public":
        return await _search_public_sites(query, domains, limit)

    log.warning("[web_fallback_v3] unsupported provider=%s", WEB_SEARCH_PROVIDER)
    return []


def _is_steam_store_query(query: str) -> bool:
    normalized = query.lower().replace(" ", "")
    has_store_intent = any(
        keyword.replace(" ", "").lower() in normalized
        for keyword in STEAM_STORE_KEYWORDS
    )
    has_tarkov = any(
        keyword.replace(" ", "").lower() in normalized
        for keyword in TARKOV_KEYWORDS
    )
    return has_store_intent and has_tarkov


async def _fetch_steam_store_price(
    domains: list[str],
) -> WebFallbackResult | None:
    steam_url = (
        f"https://store.steampowered.com/app/{WEB_FALLBACK_STEAM_APP_ID}/"
        "Escape_from_Tarkov/"
    )
    if not _allowed_url(steam_url, domains):
        return None

    params = {
        "appids": WEB_FALLBACK_STEAM_APP_ID,
        "cc": WEB_FALLBACK_STEAM_COUNTRY,
        "l": WEB_FALLBACK_STEAM_LANG,
        "filters": "price_overview,basic",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://store.steampowered.com/api/appdetails",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("[web_fallback_v3] steam store lookup failed: %s", exc)
        return None

    app_data = data.get(WEB_FALLBACK_STEAM_APP_ID, {})
    if not app_data.get("success"):
        return None

    details = app_data.get("data") or {}
    price = details.get("price_overview") or {}
    final_price = price.get("final_formatted")
    if not final_price:
        return None

    discount = int(price.get("discount_percent") or 0)
    initial_price = price.get("initial_formatted")
    name = details.get("name") or "Escape from Tarkov"
    discount_text = (
        f" 현재 할인율은 {discount}%이며, 정상가는 {initial_price}입니다."
        if discount and initial_price
        else " 현재 Steam 할인은 적용되어 있지 않습니다."
    )
    snippet = (
        f"Steam Store 공개 API 기준 {WEB_FALLBACK_STEAM_COUNTRY.upper()} 지역의 "
        f"{name} 현재 가격은 {final_price}입니다.{discount_text}"
    )
    source, is_community = _source_from_url(steam_url)
    log.info(
        "[web_fallback_v3] steam store app=%s country=%s price=%s",
        WEB_FALLBACK_STEAM_APP_ID,
        WEB_FALLBACK_STEAM_COUNTRY,
        final_price,
    )
    return WebFallbackResult(
        title=f"{name} - Steam Store",
        url=steam_url,
        snippet=snippet,
        source=source,
        is_community=is_community,
    )


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
        if WEB_PUBLIC_SEARCH_ENABLED:
            if any("escapefromtarkov.fandom.com" in domain for domain in domains):
                results.extend(
                    await _search_fandom_api(
                        client,
                        query,
                        domains,
                        limit - len(results),
                    )
                )
            results.extend(
                await _search_duckduckgo_html(
                    client,
                    query,
                    domains,
                    limit - len(results),
                )
            )
        if any("gall.dcinside.com" in domain for domain in domains):
            results.extend(await _search_dcinside(client, query, limit - len(results)))

    log.info("[web_fallback_v3] public query=%s results=%s", query[:40], len(results))
    return _dedupe_results(results)[:limit]


def _fandom_search_terms(query: str) -> list[str]:
    rewritten = _rewrite_query_for_public_search(query)
    compact = query.lower().replace(" ", "")
    terms = [rewritten, query]
    if "thedoor" in compact or "the door" in query.lower():
        terms.insert(0, "The Door")
    if "미궁" in compact or "labyrinth" in compact:
        terms.insert(0, "The Labyrinth")
        terms.insert(1, "The Labyrinth map")
    if "진행중인이벤트" in compact or "현재이벤트" in compact or "currentevent" in compact:
        terms.insert(0, "Events")
    if "bloody" in compact or "블러드키" in compact or "블러디키" in compact:
        terms.insert(0, "Rusted bloody key")
    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


async def _search_fandom_api(
    client: httpx.AsyncClient,
    query: str,
    domains: list[str],
    limit: int,
) -> list[WebFallbackResult]:
    if limit <= 0:
        return []

    results: list[WebFallbackResult] = []
    for term in _fandom_search_terms(query):
        try:
            resp = await client.get(
                "https://escapefromtarkov.fandom.com/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": max(limit, 3),
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("[web_fallback_v3] fandom search failed: %s", exc)
            continue

        for item in data.get("query", {}).get("search", []):
            title = item.get("title") or ""
            if not title:
                continue
            url = f"https://escapefromtarkov.fandom.com/wiki/{quote(title.replace(' ', '_'), safe=':_()')}"
            if not _allowed_url(url, domains):
                continue
            snippet = await _fetch_fandom_extract(client, title)
            if not snippet:
                snippet = BeautifulSoup(item.get("snippet") or "", "html.parser").get_text(" ", strip=True)
            source, is_community = _source_from_url(url)
            results.append(
                WebFallbackResult(
                    title=f"{title} - The Official Escape from Tarkov Wiki",
                    url=url,
                    snippet=snippet or title,
                    source=source,
                    is_community=is_community,
                )
            )
            results = _dedupe_results(results)
            if len(results) >= limit:
                log.info("[web_fallback_v3] fandom query=%s results=%s", query[:40], len(results))
                return results

    log.info("[web_fallback_v3] fandom query=%s results=%s", query[:40], len(results))
    return results


async def _fetch_fandom_extract(
    client: httpx.AsyncClient,
    title: str,
    max_chars: int = 1200,
) -> str:
    try:
        resp = await client.get(
            "https://escapefromtarkov.fandom.com/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                "titles": title,
                "format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return ""

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = " ".join(str(page.get("extract") or "").split())
        if extract:
            return extract[:max_chars]
    return ""


def _dedupe_results(results: list[WebFallbackResult]) -> list[WebFallbackResult]:
    deduped: list[WebFallbackResult] = []
    seen: set[str] = set()
    for result in results:
        key = result.url.split("#", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _public_search_query(query: str, domains: list[str]) -> str:
    useful_domains = [
        domain
        for domain in domains
        if not domain.startswith("store.steampowered.com")
        and "gall.dcinside.com" not in domain
    ]
    if not useful_domains:
        useful_domains = domains
    site_filters = " OR ".join(f"site:{domain}" for domain in useful_domains)
    return f"Escape from Tarkov {query} ({site_filters})"


def _rewrite_query_for_public_search(query: str) -> str:
    rewritten = query.strip()
    replacements = {
        "블러드키": "Rusted bloody key",
        "블러디키": "Rusted bloody key",
        "bloody 키": "Rusted bloody key",
        "미궁": "Labyrinth",
        "여는 법": "guide",
        "여는법": "guide",
        "구하는법": "how to get",
        "구하는 법": "how to get",
        "위치": "location",
        "지도": "map",
        "진행중인 이벤트": "current event",
        "현재 이벤트": "current event",
        "최신 패치노트": "latest patch notes",
        "패치노트": "patch notes",
        "시세": "price",
    }
    for source, target in replacements.items():
        rewritten = rewritten.replace(source, target)
    return rewritten


def _public_search_queries(query: str, domains: list[str]) -> list[str]:
    rewritten = _rewrite_query_for_public_search(query)
    compact = query.lower().replace(" ", "")
    targeted: list[str] = []
    if "thedoor" in compact or "the door" in query.lower():
        targeted.extend(
            [
                "The Door Escape from Tarkov wiki",
                "The Door Tarkov quest guide",
            ]
        )
    if "미궁" in compact or "labyrinth" in compact:
        targeted.extend(
            [
                "Escape from Tarkov Labyrinth map",
                "Map:The Labyrinth Escape from Tarkov wiki",
            ]
        )
    if "진행중인이벤트" in compact or "현재이벤트" in compact or "currentevent" in compact:
        targeted.extend(
            [
                "Tarkov current event",
                "Escape from Tarkov Events wiki",
                "Escape from Tarkov news",
            ]
        )
    if "bloody" in compact or "블러드키" in compact or "블러디키" in compact:
        targeted.extend(
            [
                "Rusted bloody key Escape from Tarkov wiki",
                "Rusted bloody key location Tarkov",
            ]
        )
    queries = [
        *targeted,
        _public_search_query(query, domains),
        _public_search_query(rewritten, domains),
        f"{rewritten} Escape from Tarkov wiki",
        f"{rewritten} Tarkov",
    ]
    return list(dict.fromkeys(q for q in queries if q.strip()))


def _resolve_duckduckgo_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return target or href
    return href


async def _search_duckduckgo_html(
    client: httpx.AsyncClient,
    query: str,
    domains: list[str],
    limit: int,
) -> list[WebFallbackResult]:
    if limit <= 0:
        return []

    try:
        search_pages = []
        for search_query in _public_search_queries(query, domains):
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": search_query},
            )
            resp.raise_for_status()
            search_pages.append(resp.text)
    except httpx.HTTPError as exc:
        log.warning("[web_fallback_v3] duckduckgo search failed: %s", exc)
        return []

    results: list[WebFallbackResult] = []
    search_domains = [*domains, "www.escapefromtarkov.com/news", "escapefromtarkov.com/news"]
    for html in search_pages:
        soup = BeautifulSoup(html, "html.parser")
        for result_node in soup.select(".result"):
            link = result_node.select_one("a.result__a")
            if not link:
                continue
            url = _resolve_duckduckgo_url(link.get("href") or "")
            if not url or not _allowed_url(url, search_domains):
                continue

            title = link.get_text(" ", strip=True)
            snippet_node = result_node.select_one(".result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            page_snippet = await _fetch_public_page_snippet(client, url)
            source, is_community = _source_from_url(url)
            results.append(
                WebFallbackResult(
                    title=title or url,
                    url=url,
                    snippet=page_snippet or snippet or title or url,
                    source=source,
                    is_community=is_community,
                )
            )
            results = _dedupe_results(results)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    log.info("[web_fallback_v3] duckduckgo query=%s results=%s", query[:40], len(results))
    return results


async def _fetch_public_page_snippet(
    client: httpx.AsyncClient,
    url: str,
    max_chars: int = 1200,
) -> str:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for unwanted in soup.select("script, style, nav, footer, header, noscript"):
        unwanted.decompose()

    selectors = (
        ".mw-parser-output",
        "main",
        "article",
        "#content",
        "body",
    )
    for selector in selectors:
        content = soup.select_one(selector)
        if not content:
            continue
        text = " ".join(content.get_text(" ", strip=True).split())
        if len(text) > 80:
            return text[:max_chars]
    return ""


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
