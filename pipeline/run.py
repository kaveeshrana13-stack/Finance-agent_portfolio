"""Generate the weekly report.

    python -m pipeline.run --holdings sample_holdings.csv
    python -m pipeline.run --holdings "TCS:10:3000,Titan:5:3500,HDFC Bank:20:1500"
    python -m pipeline.run --holdings sample_holdings.csv --no-llm      # data only, makes no model calls
    python -m pipeline.run --holdings sample_holdings.csv --stocks 2    # research agent investigates up to 2
    python -m pipeline.run --holdings sample_holdings.csv --date 2026-10-09   # back-date, to demo "what changed"

Holdings are name:quantity:average_price, or a CSV with a name, quantity and price column.
"""
import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_holdings(spec):
    """'TCS:10:3000,Titan:5' or a path to a CSV -> [{name, quantity, avg_price}]."""
    path = Path(spec)
    if path.suffix.lower() == ".csv" and path.exists():
        import pandas as pd
        from tools.portfolio import smart_detect_csv_columns
        raw = pd.read_csv(path)
        cols = smart_detect_csv_columns(raw.columns.tolist())
        out = []
        for _, row in raw.iterrows():
            name = str(row[cols["stock"]]).strip()
            if not name or name.lower() == "nan":
                continue
            qty = pd.to_numeric(row[cols["quantity"]], errors="coerce") if cols["quantity"] else None
            price = pd.to_numeric(row[cols["price"]], errors="coerce") if cols["price"] else None
            out.append({"name": name, "quantity": None if pd.isna(qty) else float(qty),
                        "avg_price": None if price is None or pd.isna(price) else float(price)})
        return out
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        bits = [b.strip() for b in part.split(":")]
        num = lambda i: float(bits[i]) if len(bits) > i and re.fullmatch(r"-?\d+(\.\d+)?", bits[i]) else None
        out.append({"name": bits[0], "quantity": num(1), "avg_price": num(2)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdings", required=True)
    ap.add_argument("--no-llm", action="store_true", help="data-only report; no model calls")
    ap.add_argument("--knowledge", default=str(ROOT / "knowledge"), help="folder of .txt/.md/.pdf documents")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--stocks", type=int, default=2, help="how many flagged stocks the research agent may investigate")
    ap.add_argument("--calls", type=int, default=2, help="tool calls per investigated stock")
    ap.add_argument("--date", default=None, help="report date YYYY-MM-DD (default today)")
    ap.add_argument("--no-memory", action="store_true", help="do not read or write week-over-week memory")
    a = ap.parse_args()

    from pipeline.graph import build_graph
    from pipeline.llm import FallbackLLM
    from pipeline.memory import Memory
    from pipeline.rag import DocumentIndex

    holdings = parse_holdings(a.holdings)
    if not holdings:
        sys.exit("no holdings found")
    kb = DocumentIndex.build(a.knowledge) if Path(a.knowledge).exists() else None
    llm = None if a.no_llm else FallbackLLM()
    day = date.fromisoformat(a.date) if a.date else date.today()

    started = time.time()
    graph = build_graph(llm=llm, kb=kb, memory=None if a.no_memory else Memory(), today=day,
                        investigate=a.stocks > 0, max_stocks=a.stocks, max_tool_calls=a.calls)
    state = graph.invoke({"holdings": holdings})
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{day.isoformat()}.md"
    path.write_text(state["report_md"], encoding="utf-8")

    print(state["report_md"].split("## Verification")[-1].strip())
    if llm:
        print("\nmodel attempts:")
        for t in llm.attempts:
            print(f"  {t['provider']}/{t['model']}: {'ok' if t['ok'] else 'FAILED ' + str(t['error'])[:110]} ({t['seconds']}s)")
    print(f"\nreport: {path}   ({time.time() - started:.0f}s, {len(kb) if kb else 0} document passages indexed)")


if __name__ == "__main__":
    main()
