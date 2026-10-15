"""Week-over-week memory: remember each run and say exactly what changed since the last one.

The comparison is plain code, so "ITC moved from HOLD to REDUCE" is a fact rather than something a
model recalls. The model is only asked to explain the changes it is given.
"""
import json
from datetime import date, datetime
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "memory" / "runs.json"
MAX_RUNS = 104          # about two years of weekly runs


class Memory:
    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_PATH)

    # ------------------------------------------------------------------ storage
    def load(self):
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def previous(self, today=None):
        """The most recent saved run from before `today`, or None on a first run."""
        today = (today or date.today()).isoformat()
        older = [r for r in self.load() if r.get("date", "") < today]
        return max(older, key=lambda r: r["date"]) if older else None

    def save(self, record, today=None):
        """Store a run. A second run on the same day replaces the first, so it never diffs against itself."""
        record = {**record, "date": (today or date.today()).isoformat(),
                  "saved_at": datetime.now().isoformat(timespec="seconds")}
        runs = [r for r in self.load() if r.get("date") != record["date"]] + [record]
        runs = sorted(runs, key=lambda r: r["date"])[-MAX_RUNS:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(runs, indent=1, ensure_ascii=False))
        return record


# --------------------------------------------------------------------------- building a record

def build_record(state):
    """The part of a finished run worth remembering, from the pipeline state."""
    stocks = {}
    for n in state.get("notes", []):
        s = state.get("facts", {}).get(n["symbol"], {})
        snap = s.get("snapshot", {})
        stocks[n["symbol"]] = {
            "action": n["action"],
            "flags": sorted({f["rule"] for f in state.get("flags", {}).get(n["symbol"], [])}),
            "current_price": snap.get("current_price"),
            "roce_pct": snap.get("roce_pct") if isinstance(snap.get("roce_pct"), (int, float)) else None,
            "pe_ratio": snap.get("pe_ratio"),
            "weight_pct": s.get("weight_pct"),
        }
    p = state.get("portfolio", {})
    c = p.get("concentration", {})
    return {
        "regime": state.get("regime", {}).get("regime"),
        "portfolio": {"total_value": p.get("total_value"), "overall_return_pct": p.get("overall_return_pct"),
                      "largest_sector": c.get("largest_sector"), "largest_sector_pct": c.get("largest_sector_pct")},
        "stocks": stocks,
    }


# --------------------------------------------------------------------------- what changed

def diff(previous, current, price_move_min_pct=3.0):
    """Changes between two records as [{type, symbol, detail, ...}]. Deterministic."""
    if not previous:
        return []
    out = []
    if previous.get("regime") != current.get("regime"):
        out.append({"type": "regime_changed", "symbol": None,
                    "detail": f"Market regime moved from {previous.get('regime')} to {current.get('regime')}."})
    ps, cs = previous.get("stocks", {}), current.get("stocks", {})
    for sym in sorted(set(ps) | set(cs)):
        if sym not in ps:
            out.append({"type": "new_holding", "symbol": sym, "detail": f"{sym} is new since the last report."})
            continue
        if sym not in cs:
            out.append({"type": "removed_holding", "symbol": sym, "detail": f"{sym} is no longer in the portfolio."})
            continue
        a, b = ps[sym], cs[sym]
        if a.get("action") != b.get("action"):
            out.append({"type": "action_changed", "symbol": sym, "from": a.get("action"), "to": b.get("action"),
                        "detail": f"{sym} changed from {a.get('action')} to {b.get('action')}."})
        for rule in sorted(set(b.get("flags", [])) - set(a.get("flags", []))):
            out.append({"type": "flag_added", "symbol": sym, "rule": rule,
                        "detail": f"{sym} newly flagged: {rule.replace('_', ' ').lower()}."})
        for rule in sorted(set(a.get("flags", [])) - set(b.get("flags", []))):
            out.append({"type": "flag_cleared", "symbol": sym, "rule": rule,
                        "detail": f"{sym} no longer flagged: {rule.replace('_', ' ').lower()}."})
        p0, p1 = a.get("current_price"), b.get("current_price")
        if isinstance(p0, (int, float)) and isinstance(p1, (int, float)) and p0:
            move = round((p1 / p0 - 1) * 100, 2)
            if abs(move) >= price_move_min_pct:
                out.append({"type": "price_move", "symbol": sym, "pct": move,
                            "detail": f"{sym}'s price moved {move:+.2f}% since the last report."})
    return out


def summarize_changes(changes, since):
    """Plain-text list for the report and the prompt. Empty string when nothing changed."""
    if not changes:
        return ""
    lines = [f"Changes since the report of {since}:"] + [f"- {c['detail']}" for c in changes]
    return "\n".join(lines)
