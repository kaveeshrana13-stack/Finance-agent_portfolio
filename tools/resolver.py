"""Deterministic stock resolver: company name / symbol / ISIN -> official NSE symbol.

Source of truth is NSE's equity list (data/nse_equity_list.csv). Nothing here calls an LLM.
A query is never silently guessed: it is either resolved, ambiguous (candidates returned),
or not found.
"""
import csv
import difflib
import json
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.request
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(
    os.getenv("NSE_EQUITY_LIST")
    or Path(__file__).resolve().parent.parent / "data" / "nse_equity_list.csv"
)
META_FILE = DATA_FILE.with_suffix(".meta.json")  # records when the list was downloaded
NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
MAX_AGE_DAYS = 30  # refresh the list when it is older than this

# Names people actually type that differ from the official company name.
# More than one symbol means the name is genuinely ambiguous (returned as candidates).
ALIASES = {
    "zomato": ["ETERNAL"],           # renamed to Eternal
    "l&t": ["LT"],
    "larsen": ["LT"],
    "tata motors": ["TMPV", "TMCV"],  # demerged: passenger vs commercial vehicles
    "airtel": ["BHARTIARTL"],
    "sbi": ["SBIN"],
    "hul": ["HINDUNILVR"],
    "reliance": ["RELIANCE"],
    "maruti": ["MARUTI"],
    "indian oil": ["IOC"],
    "kotak bank": ["KOTAKBANK"],
}

_ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")
_NAME_NOISE = re.compile(r"\b(limited|ltd|corporation|corp|company|co|incorporated|inc)\b")
_MAX_CANDIDATES = 5


def _norm(text):
    """Lowercase, '&' -> 'and', drop punctuation/apostrophes and corporate suffixes."""
    t = str(text).lower().replace("&", " and ")
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = _NAME_NOISE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _load():
    rows = {}
    with open(DATA_FILE, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().upper() for h in next(reader)]
        i_sym, i_name, i_series, i_isin = (
            header.index(c) for c in ("SYMBOL", "NAME OF COMPANY", "SERIES", "ISIN NUMBER")
        )
        need = max(i_sym, i_name, i_series, i_isin)
        for r in reader:
            if len(r) <= need or not r[i_sym].strip():
                continue
            sym = r[i_sym].strip()
            rows[sym] = {
                "symbol": sym,
                "name": r[i_name].strip(),
                "series": r[i_series].strip(),
                "isin": r[i_isin].strip(),
            }
    by_name, by_isin = {}, {}
    for sym, rec in rows.items():
        by_name.setdefault(_norm(rec["name"]), []).append(sym)
        by_isin[rec["isin"]] = sym
    aliases = {_norm(k): v for k, v in ALIASES.items()}
    return rows, by_name, by_isin, aliases


def _resolved(rec, query, match_type, confidence, note=""):
    if rec["series"] and rec["series"] != "EQ":
        extra = f"Series {rec['series']} (not the regular EQ segment)."
        note = f"{note} {extra}".strip()
    return {
        "status": "resolved",
        "query": query,
        "symbol": rec["symbol"],
        "yahoo_symbol": rec["symbol"] + ".NS",
        "name": rec["name"],
        "isin": rec["isin"],
        "series": rec["series"],
        "match_type": match_type,
        "confidence": confidence,
        "candidates": [],
        "note": note,
    }


def _candidates(symbols, rows):
    out = []
    for s in symbols[:_MAX_CANDIDATES]:
        rec = rows.get(s)
        if rec:
            out.append({"symbol": s, "name": rec["name"]})
    return out


def _ambiguous(query, symbols, rows, match_type, note=""):
    return {
        "status": "ambiguous",
        "query": query,
        "symbol": None,
        "yahoo_symbol": None,
        "name": None,
        "isin": None,
        "series": None,
        "match_type": match_type,
        "confidence": "none",
        "candidates": _candidates(symbols, rows),
        "note": note or "Multiple listed companies match. Use the exact symbol.",
    }


def _not_found(query, note="No match in the NSE equity list."):
    return {
        "status": "not_found",
        "query": query,
        "symbol": None,
        "yahoo_symbol": None,
        "name": None,
        "isin": None,
        "series": None,
        "match_type": None,
        "confidence": "none",
        "candidates": [],
        "note": note,
    }


def _strip_exchange(upper):
    s = re.sub(r"^(NSE|BSE)[:\s]+", "", upper)
    return re.sub(r"\.(NS|BO|NSE|BSE)$", "", s).strip()


def is_stale(downloaded, today=None, max_age_days=MAX_AGE_DAYS):
    """True when the list was downloaded more than max_age_days ago (or the date is unknown)."""
    if downloaded is None:
        return True
    today = today or date.today()
    return (today - downloaded).days > max_age_days


def list_downloaded_on():
    """Date the local list was downloaded, from the sidecar file (None if unknown)."""
    try:
        return datetime.strptime(json.loads(META_FILE.read_text())["downloaded"], "%Y-%m-%d").date()
    except Exception:
        return None


_freshness_checked = False


def _ensure_fresh():
    """Once per process: refresh the list if it is older than MAX_AGE_DAYS.

    Never raises and never blocks for long; on any failure the local copy is used.
    Set NSE_AUTO_REFRESH=0 to disable.
    """
    global _freshness_checked
    if _freshness_checked:
        return
    _freshness_checked = True
    if os.getenv("NSE_AUTO_REFRESH", "1") == "0":
        return
    if is_stale(list_downloaded_on()):
        refresh_nse_list(timeout=8)


def resolve_stock(query):
    """Resolve a company name, NSE symbol or ISIN to an NSE-listed stock.

    Returns a dict with status 'resolved' | 'ambiguous' | 'not_found', plus symbol,
    yahoo_symbol, name, isin, match_type, confidence, candidates and a note.
    Match order: ISIN, exact symbol, alias, exact name, name prefix, fuzzy name.
    """
    q = str(query or "").strip()
    if not q:
        return _not_found(query, "Empty query.")

    _ensure_fresh()
    rows, by_name, by_isin, aliases = _load()
    up = q.upper()

    # 1. ISIN
    if _ISIN_RE.match(up) and up in by_isin:
        return _resolved(rows[by_isin[up]], query, "isin", "high")

    # 2. Exact symbol (with or without exchange suffix)
    sym_q = _strip_exchange(up)
    for cand in (sym_q, sym_q.replace(" ", "")):
        if cand in rows:
            return _resolved(rows[cand], query, "symbol", "high")

    nq = _norm(q)
    if not nq:
        return _not_found(query)

    # 3. Alias table
    if nq in aliases:
        syms = [s for s in aliases[nq] if s in rows]
        if len(syms) == 1:
            return _resolved(rows[syms[0]], query, "alias", "high")
        if syms:
            return _ambiguous(query, syms, rows, "alias")

    # 4. Exact (normalized) company name
    if nq in by_name:
        syms = by_name[nq]
        if len(syms) == 1:
            return _resolved(rows[syms[0]], query, "name", "high")
        return _ambiguous(query, syms, rows, "name")

    # 5. Name starts with the query (token boundary), e.g. "infosys" -> Infosys Limited
    if len(nq) >= 3:
        syms = []
        for name, group in by_name.items():
            if name.startswith(nq + " "):
                syms.extend(group)
        if len(syms) == 1:
            return _resolved(
                rows[syms[0]], query, "name_prefix", "medium",
                note=f"Matched by name prefix to {rows[syms[0]]['name']}.",
            )
        if syms:
            syms.sort(key=lambda s: len(rows[s]["name"]))
            return _ambiguous(query, syms, rows, "name_prefix")

    # 6. Fuzzy name match
    names = list(by_name)
    close = difflib.get_close_matches(nq, names, n=_MAX_CANDIDATES, cutoff=0.8)
    if close:
        scored = [(difflib.SequenceMatcher(None, nq, n).ratio(), n) for n in close]
        scored.sort(reverse=True)
        best, runner = scored[0], (scored[1] if len(scored) > 1 else None)
        unique = len(by_name[best[1]]) == 1
        clear_winner = runner is None or (best[0] >= 0.88 and best[0] - runner[0] >= 0.04)
        if unique and clear_winner:
            sym = by_name[best[1]][0]
            confidence = "medium" if best[0] >= 0.88 else "low"
            return _resolved(
                rows[sym], query, "fuzzy", confidence,
                note=f"Fuzzy match ({best[0]:.2f}) to {rows[sym]['name']}; please confirm.",
            )
        syms = [s for _, n in scored for s in by_name[n]]
        return _ambiguous(query, syms, rows, "fuzzy")

    return _not_found(query)


def format_resolution(result):
    """One-line, human/LLM-readable summary of a resolve_stock() result."""
    if result["status"] == "resolved":
        line = f"{result['query']!r} -> {result['symbol']} ({result['name']}), {result['match_type']} match, {result['confidence']} confidence."
        return f"{line} {result['note']}".strip()
    if result["status"] == "ambiguous":
        opts = "; ".join(f"{c['symbol']} ({c['name']})" for c in result["candidates"])
        return f"{result['query']!r} is ambiguous. Options: {opts}"
    return f"{result['query']!r} not found. {result['note']}"


def nse_list_size():
    return len(_load()[0])


def _ssl_context():
    """TLS context using certifi's CA bundle when available.

    python.org builds of Python on macOS often ship without a usable CA store, which makes
    plain urllib fail with CERTIFICATE_VERIFY_FAILED. Verification stays ON either way.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def refresh_nse_list(timeout=20):
    """Download the latest NSE equity list. Keeps the local copy if anything fails.

    Returns a short status string.
    """
    try:
        req = urllib.request.Request(NSE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            data = resp.read()
        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        if not lines or not lines[0].upper().startswith("SYMBOL") or len(lines) < 1000:
            return "Download did not look like the NSE equity list; kept the local copy."
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=DATA_FILE.parent, suffix=".csv")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, DATA_FILE)
        META_FILE.write_text(json.dumps({"downloaded": date.today().isoformat(), "source": NSE_URL}))
        _load.cache_clear()
        return f"Updated NSE list: {len(lines) - 1} rows."
    except urllib.error.URLError as e:
        return f"Could not refresh ({getattr(e, 'reason', e)}); kept the local copy."
    except Exception as e:
        return f"Could not refresh ({type(e).__name__}: {e}); kept the local copy."
