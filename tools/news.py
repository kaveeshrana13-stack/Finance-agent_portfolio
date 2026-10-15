"""Recent news headlines for an Indian stock: real, dated and linked, from one outlet.

Source: Moneycontrol, found through Google News RSS restricted to its news section
(`<company> site:moneycontrol.com/news`). Moneycontrol's own RSS feeds are stale, and Google News
RSS is an unofficial, best-effort interface, so results can change or be missing.

The `/news` restriction matters: without it, up to a quarter of results were Moneycontrol's
daily-refreshing quote pages ("Tech Mahindra Ltd.", "1372.3", "SENSEX >> ..."), not news.

The tool returns headlines only (no article text) and never invents any: when nothing was
published in the window it says so explicitly. Summaries and judgements belong to a later,
LLM step that must cite these items.
"""
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from tools.resolver import resolve_stock

SOURCE_NAME = "Moneycontrol (via Google News RSS)"
SITE = "moneycontrol.com/news"
DEFAULT_DAYS = 7
MAX_ITEMS = 10
MAX_TERMS = 3
CACHE_TTL_SECONDS = 600
_FEED_URL = "https://news.google.com/rss/search"
_TITLE_SUFFIX = re.compile(r"\s*-\s*Moneycontrol(\.com)?\s*$", re.I)
# Moneycontrol section landing pages that still appear in /news results.
_GENERIC_PAGE = re.compile(r"^(business news,|share/stock market news|stock market news)|latest nse, bse", re.I)
_CORP_SUFFIX = re.compile(r"\b(limited|ltd\.?|corporation|corp\.?|company|co\.?)\b", re.I)

_cache = {}


def clear_cache():
    _cache.clear()


# --------------------------------------------------------------------------- pure helpers

def search_terms(company_name=None, symbol=None):
    """Distinct search terms: a short company name plus the NSE symbol, in that order."""
    terms = []

    def add(t):
        t = re.sub(r"\s+", " ", str(t or "")).strip(" .,-")
        if t and t.lower() not in {x.lower() for x in terms}:
            terms.append(t)

    if company_name:
        add(_CORP_SUFFIX.sub(" ", company_name))
    if symbol and len(str(symbol)) >= 3:      # 1-2 letter symbols (LT) are not usable search terms
        add(symbol)
    return terms[:MAX_TERMS]


def build_query(terms, days=DEFAULT_DAYS):
    """Google News query that requires a search term in the headline (intitle:).

    Without intitle:, Google matches terms anywhere on the page and returns the outlet's latest
    articles loosely (Titan -> Nestle results, Eternal -> a book extract).
    """
    parts = [f'intitle:"{t}"' if " " in t else f"intitle:{t}" for t in terms]
    expr = parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"
    return f"{expr} site:{SITE} when:{days}d"


def build_queries(terms, days=DEFAULT_DAYS):
    """One query per term. A single query joining terms with OR intermittently came back empty
    from Google (the same query returned 10 items, then 0 with identical bytes), while each term
    on its own was reliable."""
    return [build_query([t], days) for t in terms]


def parse_feed(raw):
    """RSS bytes -> [{title, date (aware datetime), link}].

    Skips items with no usable date and Moneycontrol section landing pages.
    """
    items = []
    for it in ET.fromstring(raw).iter("item"):
        title = _TITLE_SUFFIX.sub("", (it.findtext("title") or "").strip())
        try:
            when = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if title and not _GENERIC_PAGE.search(title):
            items.append({"title": title, "date": when, "link": (it.findtext("link") or "").strip()})
    return items


def select_items(items, days=DEFAULT_DAYS, max_items=MAX_ITEMS, now=None):
    """Keep items from the last `days` days, drop exact-duplicate titles, newest first."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["date"], reverse=True):
        key = it["title"].lower()
        if it["date"] < cutoff or it["date"] > now + timedelta(hours=6) or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out[:max_items]


# --------------------------------------------------------------------------- network

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _fetch_feed(query, timeout=15):
    hit = _cache.get(query)
    now = time.time()
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    url = _FEED_URL + "?" + urllib.parse.urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        raw = resp.read()
    _cache[query] = (now, raw)
    return raw


# --------------------------------------------------------------------------- the tool

def get_stock_news(ticker_or_name, company_name=None, days=DEFAULT_DAYS):
    """Recent Moneycontrol headlines for a stock (headlines only, last `days` days).

    Args:
        ticker_or_name: NSE symbol or company name, e.g. 'INFY' or 'Infosys'.
        company_name: optional display name to search with, e.g. the name the user typed.
        days: how many days back to include (default 7).
    Returns an envelope {status, symbol, query, as_of, source, data, warnings}; data has
    count, window_days and items [{title, date, link}] newest first.
    """
    res = resolve_stock(ticker_or_name)
    symbol = res["symbol"] if res["status"] == "resolved" else None
    name = (res["name"] if res["status"] == "resolved" else None) or company_name or str(ticker_or_name)
    terms = search_terms(company_name=name, symbol=symbol)
    # Also search the names the caller used (e.g. "Zomato" for ETERNAL), since headlines use them.
    for extra in (company_name, ticker_or_name):
        for t in search_terms(company_name=extra):
            if len(terms) < MAX_TERMS and t.lower() not in {x.lower() for x in terms}:
                terms.append(t)
    queries = build_queries(terms, days)

    def envelope(status, items, warnings):
        return {
            "status": status, "symbol": symbol, "query": " || ".join(queries),
            "as_of": datetime.now().isoformat(timespec="seconds"),
            "source": SOURCE_NAME,
            "data": {"count": len(items), "window_days": days,
                     "items": [{"title": i["title"], "date": i["date"].date().isoformat(), "link": i["link"]}
                               for i in items]},
            "warnings": warnings,
        }

    collected, failures = [], []
    for q in queries:
        try:
            collected.extend(parse_feed(_fetch_feed(q)))
        except (urllib.error.URLError, TimeoutError, ET.ParseError, OSError) as e:
            failures.append(f"{type(e).__name__}: {e}")

    if failures and len(failures) == len(queries):
        return envelope("error", [], [f"Could not fetch news ({failures[0]})."])

    items = select_items(collected, days=days)
    warnings = []
    if failures:
        warnings.append(f"{len(failures)} of {len(queries)} news queries failed; results may be incomplete.")
    if not items:
        warnings.append(f"No Moneycontrol headlines found in the last {days} days.")
    return envelope("partial" if failures else "ok", items, warnings)
