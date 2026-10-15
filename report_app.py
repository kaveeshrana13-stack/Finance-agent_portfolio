"""AI Portfolio Research Agent: a verified weekly report, a research agent and RAG over company filings.

Run (from this folder):

    ~/smolagents-env/bin/streamlit run report_app.py

Everything numeric comes from tools. The model writes decisions and a summary, a research agent decides
what to look into, and every figure it states is checked against the tool results.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="AI Portfolio Research Agent", page_icon="📈", layout="wide")

DEFAULT_HOLDINGS = pd.DataFrame({
    "Stock": ["TCS", "Titan", "HDFC Bank", "ITC", "Reliance"],
    "Quantity": [10, 5, 20, 50, 15],
    "Avg buy price": [3000.0, 3500.0, 1500.0, 400.0, 1250.0],
})
ACTION_COLOUR = {"HOLD": "🟦", "ADD": "🟩", "REDUCE": "🟧", "EXIT": "🟥", "REVIEW": "⬜"}


# --------------------------------------------------------------------------- cached resources

@st.cache_resource(show_spinner="Indexing documents with LlamaIndex (first run downloads a small local embedding model)…")
def get_index():
    from pipeline.rag import DocumentIndex
    return DocumentIndex.build(ROOT / "knowledge")


def company_folders():
    base = ROOT / "knowledge"
    return sorted(p.name.upper() for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")) if base.exists() else []


# --------------------------------------------------------------------------- the run

def run_report(holdings, use_ai, n_stocks, n_calls, report_date, save_memory):
    from pipeline.graph import build_graph
    from pipeline.llm import FallbackLLM
    from pipeline.memory import Memory

    llm = FallbackLLM() if use_ai else None
    graph = build_graph(llm=llm, kb=get_index(), memory=Memory() if save_memory else None, today=report_date,
                        investigate=bool(use_ai and n_stocks > 0), max_stocks=n_stocks, max_tool_calls=n_calls)
    state, trace, started = {"holdings": holdings}, [], time.time()
    with st.status("Running the pipeline…", expanded=True) as status:
        for update in graph.stream({"holdings": holdings}, stream_mode="updates"):
            for node, changes in update.items():
                changes = changes or {}          # a node that changes nothing is reported as None
                state.update(changes)
                trace.append({"step": len(trace) + 1, "node": node, "adds": ", ".join(changes) or "-",
                              "seconds": round(time.time() - started, 1)})
                st.write(f"✅ **{node}**  →  {', '.join(changes) or 'nothing new'}")
        status.update(label=f"Done in {time.time() - started:.0f}s", state="complete", expanded=False)
    return {"state": state, "trace": trace, "attempts": list(getattr(llm, "attempts", [])),
            "seconds": round(time.time() - started, 1), "use_ai": use_ai}


def holdings_from(df):
    out = []
    for _, r in df.iterrows():
        name = str(r.get("Stock", "")).strip()
        if name and name.lower() != "nan":
            q, p = pd.to_numeric(r.get("Quantity"), errors="coerce"), pd.to_numeric(r.get("Avg buy price"), errors="coerce")
            out.append({"name": name, "quantity": None if pd.isna(q) else float(q),
                        "avg_price": None if pd.isna(p) else float(p)})
    return out


# --------------------------------------------------------------------------- sidebar

st.title("📈 AI Portfolio Research Agent")
st.caption("Code fetches and calculates every number. A research agent investigates what the rules flag. "
           "The model writes, and every figure it writes is verified against the tool results.")

with st.sidebar:
    st.header("Portfolio")
    upload = st.file_uploader("Upload a CSV (name, quantity, price)", type=["csv"])
    base = DEFAULT_HOLDINGS
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            from tools.portfolio import smart_detect_csv_columns
            cols = smart_detect_csv_columns(raw.columns.tolist())
            base = pd.DataFrame({"Stock": raw[cols["stock"]],
                                 "Quantity": raw[cols["quantity"]] if cols["quantity"] else None,
                                 "Avg buy price": raw[cols["price"]] if cols["price"] else None})
        except Exception as e:
            st.warning(f"Could not read that file ({type(e).__name__}); using the sample portfolio.")
    holdings_df = st.data_editor(base, num_rows="dynamic", use_container_width=True, hide_index=True)

    st.header("Run")
    use_ai = st.toggle("Use the AI model", value=True,
                       help="Off = data, rules and flags only. Makes no model requests.")
    n_stocks = st.slider("Stocks the research agent may investigate", 0, 3, 2, disabled=not use_ai)
    n_calls = st.slider("Tool calls per stock", 1, 3, 2, disabled=not use_ai)
    report_date = st.date_input("Report date", value=date.today(),
                                help="Run once with last week's date and once with today's to see 'what changed'.")
    save_memory = st.checkbox("Remember this run (for what-changed)", value=True)
    max_req = 1 + n_stocks * (n_calls + 1) + 2 if use_ai and n_stocks else (2 if use_ai else 0)
    st.caption(f"Model requests this run: up to about {max_req} (+ retries). Free Gemini allows 20 per day per model; "
               f"the app falls back to a second Gemini model, then a local one.")
    go = st.button("▶ Run weekly report", type="primary", use_container_width=True)
    st.divider()
    idx = None
    try:
        idx = get_index()
        st.caption(f"📚 {idx.n_docs} document pages / {idx.n_chunks} passages indexed · {idx.backend}")
    except Exception as e:                                          # keep the page usable without documents
        st.caption(f"📚 Document index unavailable ({type(e).__name__})")

if go:
    holdings = holdings_from(holdings_df)
    if not holdings:
        st.error("Add at least one stock.")
    else:
        try:
            st.session_state["run"] = run_report(holdings, use_ai, n_stocks, n_calls, report_date, save_memory)
        except Exception as e:
            st.error(f"The run failed: {type(e).__name__}: {e}")

run = st.session_state.get("run")
tab_report, tab_agent, tab_flags, tab_docs, tab_trace, tab_verify = st.tabs(
    ["📄 Report", "🕵️ Research agent", "⚑ Rules & flags", "📚 Ask the documents", "🔗 Pipeline trace", "✅ Verification"])

# --------------------------------------------------------------------------- report

with tab_report:
    if not run:
        st.info("Choose holdings in the sidebar and press **Run weekly report**. "
                "Turn the AI off for an instant data-only report.")
    else:
        s = run["state"]
        p, c = s.get("portfolio", {}), s.get("portfolio", {}).get("concentration", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Market regime", s.get("regime", {}).get("regime", "n/a"))
        m2.metric("Portfolio value", f"₹{p.get('total_value', 0):,.0f}")
        ret = p.get("overall_return_pct")
        m3.metric("Return (known cost)", "n/a" if ret is None else f"{ret:+.2f}%")
        m4.metric("Largest sector", f"{c.get('largest_sector', 'n/a')} {c.get('largest_sector_pct', 0):.0f}%")
        n_conc = sum(1 for fl in s.get("flags", {}).values() for f in fl if f["kind"] == "concern")
        m5.metric("Concerns flagged", n_conc)

        if s.get("previous"):
            st.subheader(f"What changed since {s['previous']['date']}")
            for ch in s.get("changes", []) or [{"detail": "No material changes."}]:
                st.write(f"- {ch['detail']}")
        if s.get("summary_text"):
            st.subheader("Summary")
            st.write(s["summary_text"])

        st.subheader("Decisions")
        for n in s.get("notes", []):
            badge = "✓ verified" if n.get("verified") else "⚠ contains unverified figures"
            with st.container(border=True):
                st.markdown(f"### {ACTION_COLOUR.get(n['action'], '')} {n['symbol']} — {n['action']}  ·  _{badge}_")
                st.write(n["rationale"])
                if n.get("trigger"):
                    st.caption(f"Would change if: {n['trigger']}")
                if n.get("guardrail"):
                    st.warning(f"Guardrail: {n['guardrail']}")
                fl = s.get("flags", {}).get(n["symbol"], [])
                if fl:
                    st.caption("Flags: " + " · ".join(f"{'⚠' if x['kind'] == 'concern' else '✓'} {x['title']}" for x in fl))
        if not s.get("notes"):
            st.caption("No model-written decisions in this run (AI was off or unavailable).")

        st.subheader("Holdings")
        rows = []
        for r in s.get("resolved", []):
            f = s.get("facts", {}).get(r["symbol"])
            if f:
                sn = f["snapshot"]
                num = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
                financial = isinstance(sn.get("roce_pct"), str)
                rows.append({"Stock": r["symbol"], "Weight %": num(f.get("weight_pct")), "Price ₹": num(sn.get("current_price")),
                             "1M %": num(sn.get("1m_change_pct")), "vs sector pp": num(f.get("vs_sector_1m")),
                             "P/E": num(sn.get("pe_ratio")), "ROCE %": num(sn.get("roce_pct")),
                             "ROCE trend": "n/a (financial)" if financial else (sn.get("roce_trend") or ""),
                             "D/E": num(sn.get("debt_to_equity")), "Sector benchmark": f.get("benchmark_name") or ""})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for w in s.get("warnings", []):
            st.warning(w)
        st.download_button("⬇ Download report.md", s.get("report_md", ""), file_name=f"report_{report_date}.md")

# --------------------------------------------------------------------------- research agent

with tab_agent:
    if not run:
        st.info("Run a report with the AI on to see what the research agent decided to investigate.")
    else:
        s = run["state"]
        t = s.get("triage") or {}
        if not s.get("investigations"):
            st.info("The research agent did not run (AI off, 0 stocks allowed, or nothing was flagged).")
        else:
            st.markdown(f"**Step 1 — triage.** The model chose which flagged stocks to look into "
                        f"(method: _{t.get('how')}_): " + ", ".join(f"**{p['symbol']}**" for p in t.get("picks", [])))
            for i in s["investigations"]:
                with st.container(border=True):
                    st.markdown(f"### {i['symbol']} — {i['company']}")
                    st.markdown(f"**Question the agent set itself:** {i['question']}")
                    st.caption(f"{i['tool_calls']} tool call(s), {i['requests']} model request(s), stopped: {i['stopped']}")
                    for st_ in i["steps"]:
                        args = ", ".join(f"{k}={v!r}" for k, v in st_["args"].items())
                        line = f"**{st_['n']}. {st_['action']}**({args})"
                        if st_["why"]:
                            line += f" — _{st_['why']}_"
                        st.markdown(line)
                        for eid in st_["evidence"]:
                            e = next((x for x in i["evidence"] if x["id"] == eid), None)
                            if e:
                                where = f" · {e['source']}, p.{e['page']}" if e.get("source") else ""
                                st.info(f"**[{eid}]**{where}\n\n{e['text'][:500]}")
                    st.markdown("**Findings** (each is checked: evidence exists, supports the claim, figures trace to a tool)")
                    for f in i["findings"]:
                        cite = f" `[{', '.join(f['evidence'])}]`" if f["evidence"] else " `[no evidence]`"
                        if f["verified"]:
                            st.success(f"✓ {f['claim']}{cite}")
                        else:
                            st.error(f"⚠ {f['claim']}{cite}\n\n" + "; ".join(f["issues"]))

# --------------------------------------------------------------------------- rules

with tab_flags:
    from pipeline.rules import THRESHOLDS, _PROV
    st.markdown("Rules **flag** things worth a closer look; they do not give advice. There is no authoritative "
                "standard for these thresholds, so each rule shows where its number comes from.")
    if run:
        s = run["state"]
        rows = [{"Stock": sym, "Kind": f["kind"], "Severity": f["severity"], "Flag": f["title"], "Detail": f["detail"],
                 "Basis": f["provenance"]} for sym, fl in s.get("flags", {}).items() for f in fl]
        rows += [{"Stock": f"(portfolio) {f['symbol']}", "Kind": f["kind"], "Severity": f["severity"], "Flag": f["title"],
                  "Detail": f["detail"], "Basis": f["provenance"]} for f in s.get("portfolio_flags", [])]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.success("No flags raised.")
    with st.expander("Thresholds in use (edit them in pipeline/rules.py)"):
        st.table(pd.DataFrame({"Threshold": list(THRESHOLDS), "Value": list(THRESHOLDS.values())}))

# --------------------------------------------------------------------------- ask the documents

with tab_docs:
    st.markdown("Ask a question about a company's filings. The answer must come **only** from retrieved passages, "
                "with citations that are checked, or the system says it is not in the documents.")
    folders = company_folders()
    if not idx or not idx.n_chunks:
        st.info("No documents indexed yet. Put PDFs in `knowledge/<SYMBOL>/` (for example `knowledge/tcs/`).")
    else:
        sym = st.selectbox("Company", folders or [None])
        examples = ["What total contract value did TCS book in FY26 and how many mega deals?",
                    "What was TCS's operating margin for FY26?",
                    "What is TCS's annualised AI revenue?",
                    "What was TCS's employee attrition rate?",
                    "Is TCS planning a share buyback?"]
        pick = st.selectbox("Try an example (the last two are NOT in the document, to test abstention)", [""] + examples)
        question = st.text_input("Your question", value=pick)
        c1, c2 = st.columns(2)
        show_only = c1.button("🔎 Show retrieved passages only (no AI)")
        ask = c2.button("💬 Answer with citations", type="primary")
        if (show_only or ask) and question.strip():
            hits = idx.search(question, k=3, symbol=sym)
            if show_only or not use_ai:
                st.markdown("**Retrieved passages** (hybrid: meaning + keywords)")
            if hits and (show_only or not ask):
                for n, h in enumerate(hits, 1):
                    st.info(f"**[D{n}]** {h['source']} · page {h['page']} · found by {h['found_by']} · score {h['score']}\n\n{h['text']}")
            if ask:
                try:
                    from pipeline.llm import FallbackLLM
                    from pipeline.qa import answer_question
                    with st.spinner("Answering from the passages…"):
                        r = answer_question(question, idx, FallbackLLM(), symbol=sym)
                    if r["abstained"]:
                        st.warning(f"**Not in the documents.** The model abstained instead of guessing.  \n_({r.get('model')})_")
                    else:
                        (st.success if r["grounded"] else st.error)(
                            ("✓ grounded" if r["grounded"] else "⚠ not fully grounded") + f" · {r['model']}\n\n{r['answer']}")
                        for c in r["citations"]:
                            st.caption(f"{'✓' if c['ok'] else '⚠'} citation {c['cited']} — word overlap with the cited passage "
                                       f"{c['support']:.0%}" + (f" · missing {c['missing']}" if c["missing"] else ""))
                        if r["verification"] and r["verification"]["problems"]:
                            st.caption("Figures not found in the passages: " + ", ".join(
                                x["claim"] for x in r["verification"]["problems"]))
                    for n, h in enumerate(r["passages"], 1):
                        with st.expander(f"[D{n}] {h['source']} · page {h['page']} · {h['found_by']}"):
                            st.write(h["text"])
                except Exception as e:
                    st.error(f"Could not answer: {type(e).__name__}: {str(e)[:200]}")

# --------------------------------------------------------------------------- trace

with tab_trace:
    st.markdown("The report is a **LangGraph** graph. Each row is one node reading and updating the shared state.")
    if run:
        st.dataframe(pd.DataFrame(run["trace"]), use_container_width=True, hide_index=True)
    else:
        st.info("Run a report to see the node-by-node trace.")
    with st.expander("Graph structure"):
        st.code("""resolve → market → stock_data → portfolio → rules → memory_load → investigate
        → write_notes ⇄ check_notes → guardrails → compare → write_summary ⇄ check_summary
        → save_memory → assemble""", language="text")

# --------------------------------------------------------------------------- verification

with tab_verify:
    if not run:
        st.info("Run a report to see the verification summary.")
    else:
        s = run["state"]
        figs = s.get("figures", {})
        cols = st.columns(max(1, len(figs)) or 1)
        for col, (name, v) in zip(cols, figs.items()):
            col.metric(f"{name}: traced / checked", f"{v.get('supported', 0)} / {v.get('checked', 0)}",
                       delta=f"{v.get('problems', 0)} problems", delta_color="inverse")
        st.markdown("**Model requests made**")
        att = run.get("attempts", [])
        if att:
            st.dataframe(pd.DataFrame(att), use_container_width=True, hide_index=True)
        else:
            st.caption("None: this was a data-only run.")
        st.markdown("**What the check does and does not prove**")
        st.markdown("- Every figure in model-written text must equal a number a tool returned (at the precision used), "
                    "sit next to a label that fits it, and have the right direction.\n"
                    "- Citations must point to evidence that exists and shares content with the claim.\n"
                    "- It proves the text is **grounded in the data**, not that the data or the advice is correct.")
