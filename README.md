📊 AI Portfolio Manager (India)
An AI-powered weekly portfolio decision engine for Indian equity investors.
Built to think like a fund manager, not a chatbot.
This system analyzes your stock portfolio using market regime, fundamentals, sector-specific metrics, and news, and delivers clear weekly actions — not generic advice.
🚀 What This Is (and What It Isn’t)
✅ What it does
Evaluates your entire portfolio holistically
Adapts decisions to the current market regime
Combines fundamentals + momentum + news
Gives clear actions: Hold / Add / Reduce / Exit
Guides weekly capital deployment
Focuses on Indian markets & policies
❌ What it is NOT
Not a trading bot
Not a signal generator
Not diversification-for-the-sake-of-it
Not generic “finance gyaan”
This is a decision-support system, not blind advice.
🧠 Core Philosophy
Diversification is a risk constraint, not a goal
Concentration is acceptable if earned
Every negative call must explain what breaks future returns
Every recommendation is conditional
Less information, more decisions
🧩 Key Capabilities
1️⃣ Market Regime Detection
Classifies the market as:
Bull
Range-bound
Risk-off
Uses:
Index trend
Sector leadership
Macro cues (rates, liquidity, govt capex, INR, oil)
This determines how aggressive or defensive the portfolio should be.
2️⃣ Portfolio Health Score
A quick diagnostic:
Strong
Neutral
Needs Attention
Based on:
Concentration risk
Momentum breadth
Fundamental quality
3️⃣ Stock-Level Analysis (Sector-Aware)
Each stock is evaluated using:
Core Metrics (all stocks)
ROCE trend
Free cash flow
Revenue & EPS growth
Margin trend
Balance sheet strength
Sector-Specific Metrics (mandatory)
Examples:
Banking/NBFC: Credit growth, GNPA, NIM, CASA
IT: Deal wins, margin guidance, attrition
Infra/Capex: Order book, execution visibility
Aviation: Load factor, ATF sensitivity, yield
FMCG: Volume growth, pricing power
Stocks are classified as:
Sector leader
Sector average
Structural laggard
4️⃣ News & Policy Intelligence
A dedicated section answering:
“What changed this week?”
Includes:
Company news
Sector/regulatory changes
Government policy (GST, defence, infra, capex)
Each update is mapped to:
Strengthens thesis
Weakens thesis
No material impact
5️⃣ Weekly Action Plan (Most Important)
For each stock or sector:
HOLD / ADD / REDUCE / EXIT
Clear reasoning
Conditional triggers
Example:
“HOLD Indigo — aviation demand strong, sector momentum intact.
Reduce only if ATF prices rise >10% or yields weaken for two quarters.”
No reduction purely for diversification unless justified.
6️⃣ Capital Deployment Engine
Based on:
Market regime
Portfolio risk
Valuations & momentum
Decides:
Deploy now / partially / wait
% of monthly capital to deploy this week
Sectors to prioritize
Sectors to avoid
Clear, decisive, no ambiguity.
📥 Inputs Supported
Manual text portfolio input
CSV upload
Image (portfolio screenshot / statement)
Monthly investment amount
📤 Outputs You Get
Market regime summary
Portfolio health score
Concise stock-wise actions
Key risks & upside drivers
Weekly capital allocation plan
Designed to be read in under 5 minutes.
🇮🇳 Built for Indian Investors
INR-based
Indian stocks & sectors
Indian government policy awareness
Indian market structure
⚠️ Disclaimer
This tool is for educational and decision-support purposes only.
It does not guarantee returns and does not replace independent judgment.

---

---

## 🤖 Verified research agent (v3)

The original app asked an LLM to do everything, including fetching and calculating. This version moves
every fact into plain, tested code, uses rules to *flag* what deserves attention, lets a bounded research
agent investigate the flags (including the company's real filings), and uses the model only to *write*.
Everything the model writes is checked against what the tools returned.

```
holdings → resolve → market → stock_data → portfolio → RULES → MEMORY → INVESTIGATE (agent)
                                                         flags   last      model chooses what to
                                                                 week      look into, cites evidence
         → write_notes ⇄ check_notes → GUARDRAILS → compare → write_summary ⇄ check_summary → save → report
              (model)    (verifier,     (never bolder   (what      (model)       (verifier)
                         retry once)    than a rule)    changed)
```

| Part | What it is | Uses a model? |
|---|---|---|
| Tools (`tools/`) | Prices, ROCE, news, portfolio maths, NSE symbol resolution | No |
| Rules (`pipeline/rules.py`) | Flags concerns/strengths; every threshold lists where it comes from | No |
| Memory (`pipeline/memory.py`) | "What changed since last week", computed not recalled | No |
| Research agent (`pipeline/investigator.py`) | The model picks which flagged stocks to look into and which tool to use next; hard budget | **Yes: this is the agent** |
| RAG (`pipeline/rag.py`) | LlamaIndex over filings and transcripts, local embeddings, hybrid retrieval | Embeddings only |
| Verifier (`agent/verify.py`) | Every figure must trace to a tool result; citations must exist and support the claim | No |

**What is and is not an "agent" here.** The research step is an agent: the model decides which stock to
investigate and which tool to call next. The rest of the report is a fixed workflow: code decides the order.

### Run it

```bash
pip install -r agent/requirements.txt                       # separate from the original app's requirements.txt
streamlit run report_app.py                                  # the demo UI
python -m pipeline.run --holdings sample_holdings.csv        # same thing from the command line
python -m pipeline.run --holdings sample_holdings.csv --no-llm       # data, rules and flags only; no model
python -m pipeline.run --holdings sample_holdings.csv --date 2026-10-09   # back-date, then run again for "what changed"
python -m pipeline.rag_eval --answers                        # document search + answer evaluation
python -m agent.run "Compare TCS and Infosys on P/E and ROCE"    # free-form questions
python -m unittest discover -s tests -t .                    # 242 offline tests, no network or model
```

Put filings and transcripts in `knowledge/<SYMBOL>/` (the included example is TCS's Q4 FY26 earnings-call
transcript from NSE's archive). Keys live in `.env` (`GOOGLE_API_KEY`), which is git-ignored.

### Results (and how far to trust them)

**Document search and answers** on the TCS transcript: 14 questions whose answers I read out of the document,
10 answerable and 4 whose facts are *absent* (attrition, buyback, target price, a different company's margin).

| | Result |
|---|---|
| Retrieval, answering passage in top 3: meaning-only / hybrid | 6/10 / 9/10 |
| Retrieval, top 5: meaning-only / hybrid | 8/10 / 10/10 |
| Answers with documents | 10/10 correct and cited; 4/4 absent facts refused; 0 invented |
| Same questions without documents | 0/10 (the model refused: the call postdates its knowledge) |

**Read these honestly:** the first run scored 6/10 because my scorer was unfair (it required exact transcript
wording); I fixed the scorer and the retrieval depth *after* seeing the results, so 10/10 is tuned to these
14 questions and is not a held-out test. "0/10 without documents" shows the knowledge is not in the model;
it is not a hallucination test.

**Bugs found by running real data (each is now a regression test):**

| Problem | Evidence | Fix |
|---|---|---|
| Overall return wrong when a buy price is missing | **+76.5%** shown, **+17.6%** correct | like-for-like return + a data note |
| Dividend yield under 1% multiplied by 100 | TITAN **34.0%** (real 0.34%) | unit-correct yield |
| Debt/equity reported ×100; beta not against India | TITAN **195.0** (real 1.95); ITC beta **-0.06** | computed from statements / vs NIFTY |
| Ticker and sector guessed by an LLM | `BAJAJ-AUTO`, `L&T`, Zomato→Eternal, Tata Motors demerger | deterministic NSE resolver; ambiguity returned, never guessed |
| "News" invented (a search query was built and never used) | no search tool attached | real Moneycontrol headlines |
| Results published 6 days ago labelled "this week" | `abs(days)` in the original code | past results are "recently reported" |
| Position limits flagged every holding in a 5-stock portfolio | all 5 "very large" | limits relative to an equal split |
| Research agent investigated stocks it had no documents for | live run: RELIANCE and TITAN, ignoring TCS | agent is told which companies have documents |

### Verification

`agent/verify.py` is plain code. For each figure in model-written text it checks that a tool returned that
number at the precision used, that the label next to it fits the field (a real ROCE called "P/E" is
flagged), and that the direction is right. It was hardened by attacking it; every false alarm and every miss
found that way is a test case. It proves text is **grounded in the data**, not that the data or the advice
is correct.

### Choices worth knowing

- **Rules flag; they do not advise.** Markets have too many factors for a rulebook, so rules decide what is
  worth investigating and which recommendations are blocked (never ADD against a high-severity flag). There
  is no authoritative standard for any threshold; each rule's source is shown (screener presets, adviser rules
  of thumb, or "this project's own heuristic") and all are editable in one dict.
- **Free tier only.** Gemini 2.5 Flash allows 20 requests/day on a free key, so the model chain falls through
  to a second Gemini model and then local Ollama. A report with the agent is about 6 to 10 requests.
- **Tools never call a model.** The data layer is deterministic and unit-tested.

### Known limitations

- **No held-out evaluation yet.** The document evaluation is tuned on its own questions, and there is no
  backtest of whether the flags or decisions are any good. The advice quality is whatever the model writes;
  in practice a small model returns mostly HOLD with modest reasoning. The strength here is grounding and
  traceability, not alpha.
- Data comes from Yahoo Finance via `yfinance` and Google News RSS, both unofficial. Statement-derived values
  were spot-checked against reported results for two companies only; reported figures include one-offs.
- The verifier checks **figures and a few categories, not meaning**, and will "verify" a number a tool got wrong.
- Document search is only as good as the documents provided; one company has any. Scanned PDFs are not read.
- Banks/NBFCs: ROCE, EBIT margin, FCF and D/E are marked not applicable; GNPA/NIM/CASA are unavailable.
- Sector benchmarks for several sectors are ETF stand-ins or hand-picked equal-weight baskets.
- Market regime is a two-threshold rule on NIFTY returns (a heuristic).
- The research agent is bounded (default 2 stocks, 2 tool calls each) and its triage and findings reflect the
  quality of the model behind it.
