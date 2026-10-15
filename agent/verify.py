"""Check an answer's figures against what the tools actually returned.

Pure Python (no LLM, no smolagents), so graph nodes and tests can use it directly.

    ledger = [{"tool": "get_stock_snapshot", "args": {...}, "result": {...}}, ...]
    report = verify_answer("TCS ROCE is 54.9%", ledger)

Every number in the answer is classified:
    supported    equals a number a tool returned (allowing for the rounding the answer used),
                 and sits next to a label that fits that number's field
    derived      equals a difference or ratio of two returned numbers of the same field
    mismatched   equals a returned number, but the label next to it names a different field
                 (e.g. "P/E 54.93" when 54.93 is ROCE)
    unsupported  matches nothing a tool returned

Also checked: trend words (improving / declining / stable) against *_trend fields, and
market-regime names against the returned regime.

What it cannot do: tell that a direction is wrong ("fell 4.47%" when the field is -4.47 is fine,
"rose 4.47%" is not caught), verify units it has no label for, or judge advice and opinion.
"""
import itertools
import re

# --------------------------------------------------------------------------- extraction

_NUM = re.compile(
    r"(?P<sign>[-−])?(?P<cur>₹|Rs\.?\s*)?"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<unit>\s*(?:%|percent\b|lakh\s+crore\b|crore\b|cr\b|x\b))?",
    re.I,
)
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_TIME_AFTER = re.compile(r"^\s*(?:-|\s)?(?:day|week|month|year|yr|mo|session|quarter|hour|min)s?\b", re.I)
_COUNT_AFTER = re.compile(
    r"^\s+(?:stocks?|compan(?:y|ies)|headlines?|items?|results?|holdings?|sectors?|steps?|tools?|"
    r"articles?|stories|shares?\b(?!\s+price)|names?|peers?|ones?)\b", re.I)
_TIME_BEFORE = re.compile(r"(?:last|past|over|next|previous|prior|in)\s+$", re.I)


def extract_numbers(text):
    """Numeric claims in `text`: [{raw, value, decimals, unit, start, end}].

    Skips things that are not figures: years, FY/Q labels, time windows ("3-month", "52 weeks"),
    dates, list markers and "top N".
    """
    out = []
    for m in _NUM.finditer(text or ""):
        raw, num = m.group(0).strip(), m.group("num")
        start, end = m.start(), m.end()
        before, after = text[max(0, start - 12):start], text[end:end + 14]
        value_txt = num.replace(",", "")
        decimals = len(value_txt.split(".")[1]) if "." in value_txt else 0
        has_unit = bool(m.group("unit")) or bool(m.group("cur"))

        if not has_unit and decimals == 0 and re.fullmatch(r"(19|20)\d\d", value_txt):
            continue                                                    # a year
        if re.search(r"(?:FY|Q|H)\s*$", before, re.I):
            continue                                                    # FY26, Q2
        if re.search(r"(?:19|20)\d\d\s*-\s*$", before) and not has_unit:
            continue                                                    # second half of "2025-26"
        if not has_unit and (_TIME_AFTER.match(after) or _TIME_BEFORE.search(before)):
            continue                                                    # 3-month, 52 weeks, last 7 days
        if not has_unit and re.match(rf"^\s*{_MONTHS}\b", after, re.I):
            continue                                                    # 15 Oct
        if not has_unit and re.search(rf"{_MONTHS}\.?\s*$", before, re.I):
            continue                                                    # Oct 15
        if re.search(r"(?:^|\n)\s*$", before) and re.match(r"^[.)]\s", after):
            continue                                                    # "1. " list marker
        if re.search(r"\btop\s+$", before, re.I):
            continue                                                    # top 5
        if not has_unit and re.search(r"(?:nifty|sensex|bse|nse|midcap|smallcap|next|s&p|nasdaq|dow|bank\s*nifty)\s*$",
                                      before, re.I):
            continue                                                    # index names: NIFTY 50, BSE 500, Nifty Next 50
        if not has_unit and decimals == 0 and re.match(r"^[mwyd](?:_|\b)", after, re.I):
            continue                                                    # 1m, 3M, 52w, 1y: window shorthand
        if not has_unit and decimals == 0 and _COUNT_AFTER.match(after):
            continue                                                    # "2 stocks", "3 headlines"
        if re.search(r"[A-Za-z]$", text[max(0, start - 1):start]) and not m.group("cur"):
            continue                                                    # digit glued to a word (TMPV2)

        value = float(value_txt)
        unit = (m.group("unit") or "").strip().lower()
        if re.match(r"lakh\s+crore", unit):
            value *= 1e5
            unit = "crore"
        if m.group("sign") and not re.search(r"\w\s*$", text[max(0, start - 1):start]):
            value = -value
        out.append({"raw": raw, "value": value, "decimals": decimals, "unit": unit, "start": start, "end": end})
    return out


# --------------------------------------------------------------------------- facts from the ledger

def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def collect_facts(ledger):
    """Flatten every tool result into numeric and categorical facts, remembering where each came from."""
    numeric, strings = [], []
    for call in ledger:
        tool, result = call.get("tool"), call.get("result")
        who = (result.get("symbol") if isinstance(result, dict) else None) or ""
        for path, val in _walk(result):
            leaf = re.sub(r"\[\d+\]", "", path.split(".")[-1])
            src = f"{tool}.{path}" + (f" ({who})" if who else "")
            if isinstance(val, bool) or val is None:
                continue
            if isinstance(val, (int, float)):
                numeric.append({"value": float(val), "key": leaf, "path": path, "source": src})
            elif isinstance(val, str):
                strings.append({"value": val, "key": leaf, "source": src})
                if leaf in ("title", "message", "note", "meaning", "text"):            # figures quoted in headlines etc.
                    for n in extract_numbers(val):
                        numeric.append({"value": n["value"], "key": "headline_text", "path": path, "source": src})
    return numeric, strings


# --------------------------------------------------------------------------- matching

# label words that may precede/follow a figure -> substrings of the field names that can hold it
_LABELS = [
    (r"p/?e\b|price[- ]to[- ]earnings|earnings multiple", ("pe_ratio",)),
    (r"p/?b\b|price[- ]to[- ]book", ("pb_ratio",)),
    (r"peg\b", ("peg_ratio",)),
    (r"roce\b|return on capital", ("roce",)),
    (r"roe\b|return on equity", ("roe",)),
    (r"debt|d/e\b|leverage|gearing", ("debt_to_equity", "de_ratio")),
    (r"beta\b", ("beta",)),
    (r"dividend", ("dividend",)),
    (r"volatil", ("volatility",)),
    (r"drawdown", ("drawdown",)),
    (r"market cap|market capitali[sz]ation", ("market_cap",)),
    (r"free cash|fcf\b", ("fcf",)),
    (r"margin", ("margin",)),
    (r"revenue|sales|growth|cagr", ("growth", "cagr", "revenue")),
    (r"total value|portfolio value|current value|invested|total investment|cost", ("total_value", "value", "cost", "investment")),
    (r"overall return|portfolio return|p&l|profit|loss\b", ("return", "pct")),
    (r"allocation|weight|concentrat|largest|top holding|top 5|top five|exposure|of the portfolio|of portfolio",
     ("weight", "allocation", "pct", "largest", "top")),
    (r"vs\.?\s+(?:its\s+)?sector|versus|relative|outperform\w*|underperform\w*|ahead of|behind|percentage points?",
     ("vs_", "relative", "sector", "pct", "nifty")),
    (r"52[- ]?w(?:ee)?k\s+high|year[- ]high|all[- ]time high|\bhigh\b", ("52w_high",)),
    (r"52[- ]?w(?:ee)?k\s+low|year[- ]low|all[- ]time low|\blow\b", ("52w_low",)),
    (r"share price|stock price|trad(?:es|ing|ed) at|closed at|quoted at|priced at|price\b|quote", ("current_price", "price")),
    (r"return|change|gain|fell|fall|rose|rise[ns]?\b|up\b|down\b|declin|decreas|increas|mov(?:ed|es?)\b|drop|jump|surg|slid|slump|climb",
     ("change_pct", "returns", "1w", "1m", "3m", "nifty", "return")),
]
_LABEL_RES = [(re.compile(p, re.I), keys) for p, keys in _LABELS]


def _label_for(text, start, end):
    """The field family a figure belongs to, as (keys, inherited), or (None, False).

    People write "label ... figure", so the nearest label BEFORE the figure in the same sentence wins.
    If another figure sits between the label and this one ("P/E 15.12 and TCS's is 16.47"), the label
    is carried over and flagged `inherited`, which makes a mismatch advisory instead of an error.
    A label after the figure ("54.93% ROCE") is used only when nothing precedes it.
    """
    text = text.replace("_", " ")          # field-style names ("pe_ratio") must still match their labels
    sent_start = max((text.rfind(ch, 0, start) for ch in ".!?\n"), default=-1) + 1
    window = text[sent_start:start]
    best = None
    for rx, keys in _LABEL_RES:
        for m in rx.finditer(window):
            if best is None or m.end() > best[0]:
                best = (m.end(), keys)
    if best:
        between = window[best[0]:]
        inherited = bool(re.search(r"\d", between))
        return best[1], inherited
    after = text[end:end + 18]
    cut = re.search(r"[,;)]|\s(?:and|but|while|whereas|which)\s", after)
    after = after[:cut.start()] if cut else after          # a label in the next clause is not this figure's
    fwd = None
    for rx, keys in _LABEL_RES:
        m = rx.search(after)
        if m and (fwd is None or m.start() < fwd[0]):
            fwd = (m.start(), keys)
    return (fwd[1], False) if fwd else (None, False)


_CURRENCY_KEYS = ("price", "high", "low", "eps", "target", "headline_text")
_CRORE_KEYS = ("fcf", "market_cap", "revenue", "income", "_cr", "headline_text")
_UP = re.compile(r"\b(?:rose|risen|rises?|rising|gain(?:ed|s)?|up|increas\w*|higher|jump\w*|climb\w*|advanc\w*|surg\w*|positive)\b", re.I)
_DOWN = re.compile(r"\b(?:fell|falls?|falling|declin\w*|down|decreas\w*|lower|drop\w*|slump\w*|slid\w*|negative|loss(?:es)?)\b", re.I)


def _direction_conflict(text, start, fact):
    """True when the words just before a figure say the opposite direction to the field's sign."""
    if fact["value"] == 0 or not any(k in fact["key"] for k in ("change", "returns", "1w", "1m", "3m", "nifty")):
        return False
    near = text[max(0, start - 28):start]
    up, down = bool(_UP.search(near)), bool(_DOWN.search(near))
    if up == down:
        return False
    return (up and fact["value"] < 0) or (down and fact["value"] > 0)


def _close(stated, fact, decimals):
    s, f = abs(stated), abs(fact)
    return abs(round(f, decimals) - s) < 1e-9 or abs(s - f) <= 0.5 * 10 ** (-decimals) + 1e-9


_DERIVABLE = ("pct", "yield", "roce", "roe", "beta", "ratio", "_pe", "pe_", "change", "growth", "margin", "volatility")


def _derived(numeric):
    """Differences between two returned numbers of the same rate-like field (ROCE of two stocks).

    Deliberately narrow. Ratios and differences of prices or amounts are NOT derived: with a few dozen
    facts in a ledger, some pair always lands near any small number, which would launder invented
    figures as "derived" (4455.8 / 1450.0 = 3.07 once matched an invented "3").
    """
    by_key = {}
    for f in numeric:
        if f["key"] != "headline_text" and any(d in f["key"] for d in _DERIVABLE):
            by_key.setdefault(f["key"], []).append(f)
    for key, group in by_key.items():
        for a, b in itertools.combinations(group, 2):
            yield abs(a["value"] - b["value"]), f"difference of {a['source']} and {b['source']}", key


_TREND_WORDS = {
    "improving": r"\bimprov\w*|\brising\b|\bstrengthen\w*",
    "declining": r"\bdeclin\w*|\bdeteriorat\w*|\bworsen\w*|\bfalling\b|\bweaken\w*",
    "stable": r"\bstable\b|\bsteady\b|\bflat\b",
}
_TREND_KEYS = [(r"roce|return on capital", "roce_trend"), (r"margin", "margin_trend"), (r"fcf|free cash", "fcf_trend")]
_REGIMES = ["Selective Risk-On", "Risk-On", "Range-bound", "Risk-Off", "Mixed"]


def _check_trends(answer, strings):
    findings = []
    for word, rx in _TREND_WORDS.items():
        for m in re.finditer(rx, answer, re.I):
            window = answer[max(0, m.start() - 45):m.end() + 20]
            key = next((k for p, k in _TREND_KEYS if re.search(p, window, re.I)), None)
            pool = [s for s in strings if (s["key"] == key if key else s["key"].endswith("_trend"))]
            if not pool:
                if key:                                      # a trend was claimed but never fetched
                    findings.append({"kind": "trend", "claim": f"{key.split('_')[0]} {word}", "status": "unsupported",
                                     "context": window.strip(), "source": None})
                continue
            ok = [s for s in pool if s["value"] == word]
            findings.append({"kind": "trend", "claim": f"{(key or 'trend').split('_')[0]} {word}",
                             "status": "supported" if ok else "unsupported", "context": window.strip(),
                             "source": ok[0]["source"] if ok else f"tool says {sorted({s['value'] for s in pool})}"})
    return findings


def _check_regime(answer, strings):
    seen = {s["value"] for s in strings if s["key"] == "regime"}
    findings = []
    for name in _REGIMES:
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", answer, re.I):
            if name == "Risk-On" and re.search(r"selective\s+risk-on", answer, re.I):
                continue
            findings.append({"kind": "regime", "claim": name, "status": "supported" if name in seen else "unsupported",
                             "context": name, "source": "get_market_regime.regime" if name in seen else f"tool says {sorted(seen) or 'nothing'}"})
    return findings


# --------------------------------------------------------------------------- the verifier

def verify_answer(answer, ledger):
    """Classify every figure and qualitative claim in `answer` against the tool results in `ledger`."""
    answer = answer or ""
    numeric, strings = collect_facts(ledger)
    derived = list(_derived(numeric))
    claims = []

    for n in extract_numbers(answer):
        label, inherited = _label_for(answer, n["start"], n["end"])
        ctx = answer[max(0, n["start"] - 30):n["end"] + 12].replace("\n", " ").strip()
        hits = [f for f in numeric if _close(n["value"], f["value"], n["decimals"])]
        entry = {"kind": "number", "claim": n["raw"], "value": n["value"], "context": ctx, "source": None}

        # a currency symbol or "crore" says what kind of field this must be, whatever the words say
        unit_keys = None
        if "crore" in n["unit"]:
            unit_keys = _CRORE_KEYS
        elif re.search(r"\u20b9|rs", n["raw"], re.I):
            unit_keys = _CURRENCY_KEYS
        allowed = tuple(label or ()) + tuple(unit_keys or ())
        if n["unit"] in ("%", "percent"):
            # a percentage is never a price level or a crore amount: "the price fell 9.22%" is a change
            allowed = tuple(k for k in allowed if k not in ("current_price", "price", "high", "low", "target", "eps"))
            allowed = allowed or ("change_pct", "returns", "1w", "1m", "3m", "nifty", "pct", "growth", "margin",
                                  "roce", "roe", "yield", "volatility", "drawdown", "cagr")

        # a bare integer with nothing to anchor it ("about 16") may only match a whole-number field
        bare_int = n["decimals"] == 0 and not n["unit"] and not allowed
        if bare_int:
            hits = [f for f in hits if abs(f["value"] - round(f["value"])) < 1e-9]

        if allowed:
            # a figure quoted from a headline or document has no field name to match, so any label fits
            fitting = [f for f in hits if f["key"] == "headline_text" or any(k in f["key"] for k in allowed)]
            if fitting:
                wrong_way = next((f for f in fitting if _direction_conflict(answer, n["start"], f)), None)
                if wrong_way and not any(not _direction_conflict(answer, n["start"], f) for f in fitting):
                    entry.update(status="mismatched",
                                 source=f"direction is opposite: {wrong_way['source']} = {wrong_way['value']:+g}")
                else:
                    entry.update(status="supported", source=fitting[0]["source"])
            elif hits and inherited and not unit_keys:
                entry.update(status="supported", source=hits[0]["source"],
                             note="label inferred from earlier in the sentence")
            elif hits:
                entry.update(status="mismatched",
                             source=f"matches {hits[0]['source']}, but the label names {allowed[0]}")
            else:
                dv = next((d for d in derived if n["decimals"] >= 1 and _close(n["value"], d[0], n["decimals"])
                           and any(k in d[2] for k in allowed)), None)
                entry.update(status="derived", source=dv[1]) if dv else entry.update(status="unsupported")
        else:
            if hits:
                wrong_way = next((f for f in hits if _direction_conflict(answer, n["start"], f)), None)
                if wrong_way and not any(not _direction_conflict(answer, n["start"], f) for f in hits):
                    entry.update(status="mismatched",
                                 source=f"direction is opposite: {wrong_way['source']} = {wrong_way['value']:+g}")
                else:
                    entry.update(status="supported", source=hits[0]["source"], note="no label next to the figure")
            else:
                entry.update(status="unsupported")        # no label and no match: nothing to derive from
        claims.append(entry)

    claims += _check_trends(answer, strings)
    claims += _check_regime(answer, strings)

    bad = [c for c in claims if c["status"] in ("unsupported", "mismatched")]
    good = [c for c in claims if c["status"] in ("supported", "derived")]
    if not claims:
        verdict = "no_claims"
    elif not bad:
        verdict = "verified"
    elif good:
        verdict = "partially_verified"
    else:
        verdict = "unverified"
    return {"verdict": verdict, "claims": claims, "problems": bad,
            "n_supported": len([c for c in good if c["status"] == "supported"]),
            "n_derived": len([c for c in good if c["status"] == "derived"]),
            "n_problems": len(bad), "tool_calls_seen": len(ledger)}


def annotate(answer, report):
    """The answer with every problem figure followed by [unverified] (or [label mismatch])."""
    spans = []
    for c in report["problems"]:
        if c["kind"] != "number":
            continue
        for n in extract_numbers(answer):
            if n["raw"] == c["claim"] and abs(n["value"] - c["value"]) < 1e-9:
                spans.append((n["end"], "[label mismatch]" if c["status"] == "mismatched" else "[unverified]"))
                break
    out = answer
    for end, tag in sorted(set(spans), reverse=True):
        out = out[:end] + " " + tag + out[end:]
    return out


def summarize_report(report):
    """One short line for logs and the UI."""
    if report["verdict"] == "no_claims":
        return "No numeric claims to verify."
    s = f"{report['n_supported']} supported, {report['n_derived']} derived, {report['n_problems']} unsupported/mismatched"
    return f"{report['verdict']}: {s}"
