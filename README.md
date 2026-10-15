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

## 🤖 Agentic pipeline (v2)

The original app asked an LLM to do everything, including fetching and calculating. v2 moves every fact
into plain, tested code and uses the model only to *write*, then checks what it wrote.

```
holdings ─► resolve ─► market ─► stock_data ─► portfolio ─► documents ─► write_notes ◄─┐
            (NSE list)  regime    price, ROCE,   totals,      (BM25 over     (model)     │ retry once,
                                  news, sector   weights,     knowledge/)       │        │ with the
                                  benchmark      concentration                  ▼        │ problems listed
                                                                          check_notes ───┘
                                                              ─► write_summary ─► check_summary ─► report.md
                                                                    (model)         (verifier)
```

**The model is called at two nodes only**, so a report costs 2 requests (4 with retries). That is what
makes the free Gemini tier (20 requests/day/model) workable. Every figure the model writes is traced to a
tool result; anything that cannot be is tagged `[unverified]` in the report.

### Run it

```bash
pip install -r agent/requirements.txt          # separate from the Streamlit app's requirements.txt
python -m pipeline.run --holdings sample_holdings.csv            # Gemini -> Gemini lite -> local Ollama
python -m pipeline.run --holdings sample_holdings.csv --no-llm   # data only, zero model calls
python -m pipeline.run --holdings "TCS:10:3000,Titan:5:3500"
python -m agent.run "Compare TCS and Infosys on P/E and ROCE"    # free-form questions (smolagents)
python -m unittest discover -s tests -t .                         # 142 offline tests
```

Put annual reports or notes in `knowledge/` (`.pdf`, `.txt`, `.md`); the report cites them as `[D1]`.
Keys live in `.env` (`GOOGLE_API_KEY`), which is git-ignored.

### What it does differently from v1 (each one found by testing, each now a regression test)

| Problem in v1 | Evidence | Fix |
|---|---|---|
| Overall return wrong when a buy price is missing | 3 equal holdings, one without cost: **+76.5%** shown, **+17.6%** correct | like-for-like return + a data-quality note |
| Dividend yield under 1% multiplied by 100 | TITAN **34.0%** (real 0.34%), RELIANCE 51.0% | unit-correct yield, plausibility bound |
| Debt/equity reported ×100 | TITAN **195.0** (real 1.95) | computed from the balance sheet |
| Beta not measured against India | ITC **-0.06**, TCS 0.18 (vs NIFTY: 0.74, 0.83) | beta computed vs NIFTY 50, 1y daily |
| Ticker/sector guessed by an LLM | `BAJAJ-AUTO`, `L&T`, Zomato→Eternal, Tata Motors demerger all wrong | deterministic resolver on the NSE equity list; ambiguity returned, never guessed |
| "News" invented (search query built, never used) | no search tool was attached | real Moneycontrol headlines, dated and linked |
| 5 of 9 sector indices silently empty | single-row histories on Yahoo | ETF stand-ins / baskets, labelled; "none" stated explicitly |
| Fundamentals the prompt demanded were never fetched | ROCE, FCF, margins, growth absent | computed from annual statements; banks marked not applicable |

### Verification

`agent/verify.py` is plain code (no LLM). For each figure in a model-written text it checks that a tool
returned that number at the precision used, that the label next to it fits the field (a real ROCE called
"P/E" is flagged), and that the direction is right (a fall called a rise is flagged). Trend words and
regime names are checked against the returned values. It was hardened by attacking it: every false alarm
and every miss found that way is a test case.

### Choices worth knowing

- **smolagents** for free-form questions, **LangGraph** for the fixed weekly report, a small **BM25**
  retriever for documents. Each was chosen for the shape of its job; LlamaIndex was not needed here.
- **Free-tier only.** Gemini 2.5 Flash is limited to 20 requests/day on a free key, so the chain falls
  through to a second Gemini model and then a local Ollama model. A model that returns a daily-quota
  error is skipped for the rest of the run.
- **Tools never call a model.** The data layer (`tools/`) is deterministic and unit-tested.

### Known limitations

- Data comes from Yahoo Finance via `yfinance` (unofficial) and Google News RSS (unofficial). Either can
  change or throttle. Statement-derived values were spot-checked against reported results for two
  companies (TCS, Titan); TCS net profit could not be fully reconciled across sources, and reported
  figures include one-offs.
- The verifier checks **figures and a few categories, not meaning**. It cannot tell whether a
  recommendation is wise, and it will "verify" a number a tool got wrong.
- Banks/NBFCs: ROCE, EBIT margin, FCF and D/E are marked not applicable; GNPA/NIM/CASA are not available.
- Sector benchmarks for several sectors are ETF stand-ins or hand-picked equal-weight baskets, not
  official indices. The basket lists are maintained by hand.
- Market regime is a two-threshold rule on NIFTY's 1m/3m returns (a heuristic, not validated).
- The local fallback model (7B) is slow and variable; one single attempt took 15 minutes.
- Not yet integrated into the Streamlit UI; run from the command line.
