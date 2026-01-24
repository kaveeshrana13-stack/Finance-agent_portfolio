import os
import re
import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Try to import yfinance
YFINANCE_AVAILABLE = False
try:
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# Global setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Finance Agent & Portfolio", layout="wide")

# Session state
if "profile" not in st.session_state:
    st.session_state.profile = {"identity": None, "income_range": None, "risk": None, "goal": None}
if "chat" not in st.session_state:
    st.session_state.chat = []
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["stock_name", "ticker", "quantity", "avg_buy_price", "sector"])
if "confirmed_portfolio" not in st.session_state:
    st.session_state.confirmed_portfolio = False
if "stock_metrics" not in st.session_state:
    st.session_state.stock_metrics = {}
# ISSUE 4 FIX: Add monthly budget to session state
if "monthly_budget" not in st.session_state:
    st.session_state.monthly_budget = None

# Profile helpers
def infer_profile_with_ai(user_msg):
    extractor_model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""Extract finance profile from: "{user_msg}"
Return JSON: {{"identity": "student/salaried/self-employed/retired or null", "income_range": "0-20k/20k-50k/50k-1L/1L+ or null", "risk": "low/medium/high or null", "goal": "save/invest/reduce_debt/learn or null"}}
Only JSON, no markdown."""
    
    try:
        resp = extractor_model.generate_content(prompt)
        text = resp.text.strip().strip("`").replace("json", "").strip()
        data = json.loads(text)
        prof = st.session_state.profile
        for key in ['identity', 'income_range', 'risk', 'goal']:
            if data.get(key) and not prof[key]:
                prof[key] = data[key]
    except Exception:
        pass

def get_profile_summary():
    p = st.session_state.profile
    parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in p.items() if v]
    # ISSUE 4 FIX: Include monthly budget in profile summary
    if st.session_state.monthly_budget:
        parts.append(f"Monthly Investment Budget: ₹{st.session_state.monthly_budget:,}")
    return " | ".join(parts) if parts else "No profile info"

SYSTEM_PROMPT = """You are a Personal Finance Agent for India. Help users make better financial decisions safely.
Current profile: {PROFILE_SUMMARY}"""

# ISSUE 5 FIX: Add news fetching function with web search capability
def fetch_stock_news(stock_name, ticker, sector):
    """Fetch recent news about stock, sector, or related government policies using AI with search"""
    
    # ISSUE 2 FIX: Use web search to get actual latest news
    try:
        # Use Gemini's built-in search capability or construct a search query
        from datetime import datetime, timedelta
        
        # Calculate date range for recent news
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        
        search_query = f"{stock_name} {ticker} news India stock market {today.strftime('%B %Y')}"
        
        analyzer = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""You are a financial news analyst. Search the web for the MOST RECENT news about:

Stock: {stock_name} ({ticker})
Sector: {sector}
Current Date: {today.strftime('%B %d, %Y')}

IMPORTANT: Only report news from the last 7 days. Ignore any news older than {week_ago.strftime('%B %d, %Y')}.

Search for:
1. Company-specific news (earnings reports, announcements, deals, management changes)
2. Sector policy changes or government regulations
3. Major industry developments affecting this stock

Provide a summary (max 150 words) with:
- Date of each news item (must be within last 7 days)
- 2-3 key headlines with dates
- Impact assessment (positive/negative/neutral)
- Credible sources (Moneycontrol, Economic Times, Business Standard, Reuters, Livemint)

If no news found from the last 7 days, state: "No significant news developments in the past week. Stock appears stable with no major announcements."

Format: Use bullet points with dates for each news item.
"""
        
        response = analyzer.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"News search unavailable. Please check Moneycontrol.com or Google News for latest updates on {stock_name}. You can search: '{stock_name} news latest'"

# Market Regime Classification
def classify_market_regime(nifty_returns, sector_breadth_data=None):
    """Classify current market regime based on NIFTY performance and breadth"""
    
    if not nifty_returns:
        return "Unknown", "Insufficient data to classify market regime"
    
    nifty_1m = nifty_returns.get('1m', 0)
    nifty_3m = nifty_returns.get('3m', 0)
    
    # Regime classification logic
    if nifty_1m > 3 and nifty_3m > 8:
        regime = "Risk-On"
        description = "Broad market participation with strong upward momentum. Favorable for holding cyclical and growth sectors."
    elif nifty_1m > 1 and nifty_3m > 3:
        regime = "Selective Risk-On"
        description = "Market showing positive momentum but participation is sector-specific. Focus on leading sectors."
    elif abs(nifty_1m) < 2 and abs(nifty_3m) < 5:
        regime = "Range-bound"
        description = "Market lacks clear direction. Favor quality over momentum, reduce position sizing."
    elif nifty_1m < -3 or nifty_3m < -8:
        regime = "Risk-Off"
        description = "Defensive positioning warranted. Prioritize quality, reduce cyclical exposure, raise cash."
    else:
        regime = "Mixed"
        description = "Market showing mixed signals. Be selective, favor sectors with policy tailwinds."
    
    # ISSUE 2 FIX: Add detailed explanation of how regime is calculated
    explanation = f"\n\n**How we calculate this regime:** Based on NIFTY 50 returns over 1 Month ({nifty_1m:+.2f}%) and 3 Months ({nifty_3m:+.2f}%). "
    explanation += f"**Risk-On** requires 1M > +3% AND 3M > +8%. "
    explanation += f"**Range-bound** means ±2% (1-month) and ±5% (3-month). "
    explanation += f"**Risk-Off** triggers when 1M < -3% OR 3M < -8%."
    
    return regime, description + explanation

# Sector Leadership Classification
def classify_sector_leadership(sector_return, nifty_return, timeframe="1M"):
    """Classify sector as Leading/Improving/Weakening/Lagging vs market"""
    
    if sector_return is None or nifty_return is None:
        return "Unknown"
    
    relative_performance = sector_return - nifty_return
    
    if sector_return > 3 and relative_performance > 2:
        return "Leading"
    elif sector_return > 0 and relative_performance > 0:
        return "Improving"
    elif sector_return < 0 and relative_performance < -2:
        return "Lagging"
    elif sector_return < -3 and relative_performance < -5:
        return "Severely Lagging"
    else:
        return "Neutral"

# Indian Market Policy Context (Narrative Intelligence)
POLICY_NARRATIVES = {
    "IT": {
        "tailwinds": ["Digital transformation spend", "AI/Cloud adoption", "GCC expansion"],
        "headwinds": ["US recession fears", "Client budget cuts", "Wage inflation"],
        "current_stance": "Mixed - growth visibility unclear due to global macro"
    },
    "Banking": {
        "tailwinds": ["Credit growth 15%+", "NIM expansion", "Asset quality improving"],
        "headwinds": ["RBI hawkish stance", "Deposit competition", "Unsecured loan stress"],
        "current_stance": "Positive - structural credit cycle intact despite rate concerns"
    },
    "Pharma": {
        "tailwinds": ["US generics recovery", "CDMO opportunities", "Domestic demand strong"],
        "headwinds": ["USFDA scrutiny", "Pricing pressure", "API cost volatility"],
        "current_stance": "Improving - regulatory overhang easing"
    },
    "Auto": {
        "tailwinds": ["Rural demand pickup", "EV policy support", "Capacity additions"],
        "headwinds": ["Commodity cost pressure", "Inventory buildup", "High interest rates"],
        "current_stance": "Positive - festive demand strong, 2W leading"
    },
    "Infrastructure": {
        "tailwinds": ["Govt capex 10L+ Cr", "PM Gati Shakti", "Order book visibility 3+ years"],
        "headwinds": ["Execution delays", "Working capital intensity", "Land acquisition"],
        "current_stance": "Structural Bull - Budget 2024 focus continues"
    },
    "FMCG": {
        "tailwinds": ["Rural recovery", "Premiumization trend", "Pricing power"],
        "headwinds": ["Volume growth weak", "Modern trade pressure", "Competition"],
        "current_stance": "Stabilizing - waiting for volume growth confirmation"
    },
    "Energy": {
        "tailwinds": ["Refining margins improving", "Gas demand up", "Renewable push"],
        "headwinds": ["Crude volatility", "Subsidy burden", "Energy transition costs"],
        "current_stance": "Neutral - watchful on crude prices"
    },
    "Metal": {
        "tailwinds": ["China stimulus hopes", "India infra demand", "Capacity discipline"],
        "headwinds": ["Global slowdown", "Steel dumping", "Power costs"],
        "current_stance": "Cyclical bottom - early signs of recovery"
    },
    "Aviation": {
        "tailwinds": ["Passenger growth 10%+", "Capacity discipline", "Fuel costs stable"],
        "headwinds": ["Competition intensity", "Airport charges", "Regulatory uncertainty"],
        "current_stance": "Expansion phase - demand recovery supported by capacity rationalization"
    },
    "Defence": {
        "tailwinds": ["Indigenization push", "Defence budget up 13%", "Export orders growing"],
        "headwinds": ["Long gestation orders", "Technology gaps", "Execution risk"],
        "current_stance": "Structural theme - Atmanirbhar Bharat focus"
    },
    "Chemicals": {
        "tailwinds": ["GST rationalization", "China+1 opportunity", "Specialty growth"],
        "headwinds": ["Inventory destocking", "Chinese competition", "Agro weakness"],
        "current_stance": "Bottom formation - GST cuts improve earnings visibility"
    },
    "Financial Services": {
        "tailwinds": ["Credit growth", "Digital banking adoption", "Market share gains"],
        "headwinds": ["Regulatory scrutiny", "Competition from fintech", "Asset quality monitoring"],
        "current_stance": "Positive - diversified growth opportunities"
    }
}

def get_sector_narrative(sector):
    """Get policy and narrative context for a sector"""
    # Map sector variations to standard names
    sector_mapping = {
        "Financial Services": "Financial Services",
        "Oil & Gas": "Energy",
        "Cement": "Infrastructure"
    }
    
    normalized_sector = sector_mapping.get(sector, sector)
    narrative = POLICY_NARRATIVES.get(normalized_sector, {
        "tailwinds": ["Sector-specific opportunities"],
        "headwinds": ["Sector-specific challenges"],
        "current_stance": "Neutral - monitor for changes"
    })
    
    return narrative

# Ticker helpers
def clean_ticker_symbol(symbol):
    """Clean and normalize ticker symbol"""
    if not symbol:
        return None
    symbol = str(symbol).strip().upper()
    symbol = symbol.replace('.NS', '').replace('.BO', '')
    symbol = re.sub(r'[^A-Z0-9&]', '', symbol)
    return symbol if symbol else None

# Fetch NIFTY 50 and sector index returns for benchmarking
def fetch_index_returns(index_name="NIFTY 50"):
    """Fetch weekly monthly and quarterly returns for an index"""
    yahoo_mapping = {
        "NIFTY 50": "^NSEI",
        "NIFTY IT": "^CNXIT",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY PHARMA": "^CNXPHARMA",
        "NIFTY FMCG": "^CNXFMCG",
        "NIFTY AUTO": "^CNXAUTO",
        "NIFTY ENERGY": "^CNXENERGY",
        "NIFTY METAL": "^CNXMETAL",
        "NIFTY INFRASTRUCTURE": "^CNXINFRA"
    }
    
    ticker = yahoo_mapping.get(index_name, "^NSEI")
    
    try:
        if not YFINANCE_AVAILABLE:
            return None
        
        index = yf.Ticker(ticker)
        hist = index.history(period="3mo")
        
        if hist.empty or len(hist) < 2:
            return None
        
        hist = hist.dropna(subset=['Close'])
        
        if len(hist) < 2:
            return None
        
        current = float(hist['Close'].iloc[-1])
        
        returns = {}
        
        # One week return
        if len(hist) >= 6:
            week_ago = float(hist['Close'].iloc[-6])
            oneweek = round(((current - week_ago) / week_ago) * 100, 2)
            returns['1w'] = oneweek
        
        # One month return
        if len(hist) >= 21:
            month_ago = float(hist['Close'].iloc[-21])
            onemonth = round(((current - month_ago) / month_ago) * 100, 2)
            returns['1m'] = onemonth
        
        # Three month return
        start = float(hist['Close'].iloc[0])
        threemonth = round(((current - start) / start) * 100, 2)
        returns['3m'] = threemonth
        
        return returns
        
    except Exception:
        return None

# Sector to index mapping
SECTOR_TO_INDEX = {
    "IT": "NIFTY IT",
    "Banking": "NIFTY BANK",
    "Pharma": "NIFTY PHARMA",
    "FMCG": "NIFTY FMCG",
    "Auto": "NIFTY AUTO",
    "Energy": "NIFTY ENERGY",
    "Metal": "NIFTY METAL",
    "Infrastructure": "NIFTY INFRASTRUCTURE",
    "Financial Services": "NIFTY BANK",
    "Oil & Gas": "NIFTY ENERGY",
    "Cement": "NIFTY INFRASTRUCTURE"
}
# PART 2 - Stock Metrics Fetching and AI Analysis Functions
# This continues from Part 1

# Fetch stock metrics using yfinance
def fetch_stock_metrics_yf(ticker):
    """Fetch fundamental metrics using yfinance"""
    if not YFINANCE_AVAILABLE or not ticker:
        return None
    
    clean_symbol = clean_ticker_symbol(ticker)
    if not clean_symbol:
        return None
    
    # Try NSE first, then BSE
    for exchange in ['.NS', '.BO']:
        try:
            symbol = f"{clean_symbol}{exchange}"
            stock = yf.Ticker(symbol)
            
            # Get info
            info = stock.info
            
            if not info or 'currentPrice' not in info:
                continue
            
            metrics = {}
            
            # Price metrics
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            metrics['current_price'] = round(float(current_price), 2) if current_price else None
            metrics['day_high'] = round(float(info.get('dayHigh', 0)), 2) if info.get('dayHigh') else None
            metrics['day_low'] = round(float(info.get('dayLow', 0)), 2) if info.get('dayLow') else None
            metrics['52w_high'] = round(float(info.get('fiftyTwoWeekHigh', 0)), 2) if info.get('fiftyTwoWeekHigh') else None
            metrics['52w_low'] = round(float(info.get('fiftyTwoWeekLow', 0)), 2) if info.get('fiftyTwoWeekLow') else None
            
            # MOMENTUM AND TREND - Fetch historical data for weekly monthly quarterly changes
            try:
                hist_3m = stock.history(period="3mo")
                if not hist_3m.empty and len(hist_3m) > 1:
                    current = float(hist_3m['Close'].iloc[-1])
                    
                    # Weekly change (5 trading days)
                    if len(hist_3m) >= 6:
                        week_ago = float(hist_3m['Close'].iloc[-6])
                        weekly_change = round(((current - week_ago) / week_ago) * 100, 2)
                        metrics['1w_change_pct'] = weekly_change
                    
                    # Monthly change (20 trading days)
                    if len(hist_3m) >= 21:
                        month_ago = float(hist_3m['Close'].iloc[-21])
                        monthly_change = round(((current - month_ago) / month_ago) * 100, 2)
                        metrics['1m_change_pct'] = monthly_change
                    
                    # Quarterly change
                    start_price = float(hist_3m['Close'].iloc[0])
                    quarterly_change = round(((current - start_price) / start_price) * 100, 2)
                    metrics['3m_change_pct'] = quarterly_change
                    
                    # Trend classification
                    m1 = metrics.get('1m_change_pct', 0) or 0
                    m3 = metrics.get('3m_change_pct', 0) or 0
                    
                    if m1 > 5 and m3 > 10:
                        metrics['trend_label'] = "Strong Uptrend"
                    elif m1 < -5 and m3 < -10:
                        metrics['trend_label'] = "Downtrend"
                    elif abs(m1) < 3 and abs(m3) < 5:
                        metrics['trend_label'] = "Sideways"
                    elif m1 > 0 and m3 > 0:
                        metrics['trend_label'] = "Weak Uptrend"
                    else:
                        metrics['trend_label'] = "Weak Downtrend"
                    
                    # Volatility calculation (standard deviation of returns)
                    returns = hist_3m['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
                    
                    if volatility < 20:
                        metrics['volatility_bucket'] = "Low"
                    elif volatility < 40:
                        metrics['volatility_bucket'] = "Medium"
                    else:
                        metrics['volatility_bucket'] = "High"
                    
                    metrics['volatility_pct'] = round(volatility, 2)
                    
            except Exception:
                pass
            
            # Calculate % from 52W high/low
            if metrics['current_price'] and metrics['52w_high']:
                metrics['pct_from_52w_high'] = round(((metrics['current_price'] - metrics['52w_high']) / metrics['52w_high']) * 100, 2)
                metrics['drawdown_from_high'] = abs(metrics['pct_from_52w_high'])
            else:
                metrics['pct_from_52w_high'] = None
                metrics['drawdown_from_high'] = None
                
            if metrics['current_price'] and metrics['52w_low']:
                metrics['pct_from_52w_low'] = round(((metrics['current_price'] - metrics['52w_low']) / metrics['52w_low']) * 100, 2)
            else:
                metrics['pct_from_52w_low'] = None
            
            # Valuation metrics
            pe = info.get('trailingPE')
            metrics['pe_ratio'] = round(float(pe), 2) if pe else None
            
            pb = info.get('priceToBook')
            metrics['pb_ratio'] = round(float(pb), 2) if pb else None
            
            peg = info.get('pegRatio')
            metrics['peg_ratio'] = round(float(peg), 2) if peg else None
            
            market_cap = info.get('marketCap')
            if market_cap:
                metrics['market_cap_cr'] = round(float(market_cap) / 10000000, 2)
            else:
                metrics['market_cap_cr'] = None
            
            # Fix dividend yield calculation
            div_yield = info.get('dividendYield')
            if div_yield and div_yield > 0:
                if div_yield < 1:
                    metrics['dividend_yield'] = round(float(div_yield) * 100, 2)
                else:
                    metrics['dividend_yield'] = round(float(div_yield), 2)
            else:
                div_rate = info.get('dividendRate')
                current_price_val = metrics.get('current_price')
                if div_rate and current_price_val and current_price_val > 0:
                    metrics['dividend_yield'] = round((float(div_rate) / float(current_price_val)) * 100, 2)
                else:
                    metrics['dividend_yield'] = None
            
            # Quality & Risk metrics
            roe = info.get('returnOnEquity')
            metrics['roe'] = round(float(roe) * 100, 2) if roe else None
            
            debt = info.get('debtToEquity')
            metrics['debt_to_equity'] = round(float(debt), 2) if debt else None
            
            # Beta (risk measure)
            beta = info.get('beta')
            metrics['beta'] = round(float(beta), 2) if beta else None
            
            # Earnings date proximity
            earnings_dates = info.get('earningsTimestamp')
            if earnings_dates:
                from datetime import datetime, timedelta
                earnings_dt = datetime.fromtimestamp(earnings_dates)
                days_to_earnings = (earnings_dt - datetime.now()).days
                
                if abs(days_to_earnings) <= 7:
                    metrics['earnings_proximity'] = "This week"
                elif abs(days_to_earnings) <= 14:
                    metrics['earnings_proximity'] = "Within 2 weeks"
                elif days_to_earnings < 0 and abs(days_to_earnings) <= 30:
                    metrics['earnings_proximity'] = "Recently reported"
                else:
                    metrics['earnings_proximity'] = "Not soon"
            else:
                metrics['earnings_proximity'] = "Unknown"
            
            # Dividend date
            ex_div_date = info.get('exDividendDate')
            if ex_div_date:
                metrics['recent_dividend'] = True
            else:
                metrics['recent_dividend'] = False
            
            metrics['exchange'] = exchange
            return metrics
            
        except Exception:
            continue
    
    return None

# AI Analysis Helper
def analyze_stock_with_ai(stock_name, ticker, sector, metrics, user_profile, portfolio_context=None, benchmarks=None, market_regime=None, sector_narrative=None, monthly_budget=None):
    """Use AI to provide ACTIONABLE stock analysis with market context and narrative intelligence"""
    analyzer = genai.GenerativeModel("gemini-2.5-flash")
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items() if v is not None])
    
    # Add portfolio context if available
    portfolio_info = ""
    if portfolio_context:
        portfolio_info = f"""
Portfolio Context:
- Total portfolio value: Rs {portfolio_context.get('total_value', 0):,.0f}
- This stock weight: {portfolio_context.get('stock_weight_pct', 0):.1f}%
- Sector ({sector}) weight: {portfolio_context.get('sector_weight_pct', 0):.1f}%
- Number of stocks: {portfolio_context.get('num_stocks', 0)}
"""
    
    # Add benchmark context
    benchmark_info = ""
    sector_leadership = "Unknown"
    if benchmarks:
        stock_1m = benchmarks.get('stock_1m', 0) or 0
        nifty_1m = benchmarks.get('nifty_1m', 0) or 0
        sector_leadership = classify_sector_leadership(benchmarks.get('sector_1m'), nifty_1m, "1M")
        
        benchmark_info = f"""
Benchmark Performance:
- Stock: 1W: {benchmarks.get('stock_1w', 'N/A')}% | 1M: {benchmarks.get('stock_1m', 'N/A')}% | 3M: {benchmarks.get('stock_3m', 'N/A')}%
- Sector: 1W: {benchmarks.get('sector_1w', 'N/A')}% | 1M: {benchmarks.get('sector_1m', 'N/A')}% | 3M: {benchmarks.get('sector_3m', 'N/A')}%
- NIFTY 50: 1W: {benchmarks.get('nifty_1w', 'N/A')}% | 1M: {benchmarks.get('nifty_1m', 'N/A')}% | 3M: {benchmarks.get('nifty_3m', 'N/A')}%

Relative Performance:
- vs Sector (1M): {benchmarks.get('vs_sector_1m', 'N/A')}% - Sector is {sector_leadership}
- vs NIFTY (1M): {benchmarks.get('vs_nifty_1m', 'N/A')}%
"""
    
    # Add market regime context
    regime_info = ""
    if market_regime:
        regime_info = f"""
Market Regime: {market_regime['regime']}
Context: {market_regime['description']}
"""
    
    # Add sector narrative
    narrative_info = ""
    if sector_narrative:
        narrative_info = f"""
Sector Policy Context ({sector}):
Tailwinds: {', '.join(sector_narrative.get('tailwinds', []))}
Headwinds: {', '.join(sector_narrative.get('headwinds', []))}
Current Stance: {sector_narrative.get('current_stance', 'Unknown')}
"""
    
    # ISSUE 4 FIX: Add monthly budget context
    budget_info = ""
    if monthly_budget:
        budget_info = f"\nMonthly Investment Budget: ₹{monthly_budget:,}"
    
    # ISSUE 3 FIX: Add clarity note about stock-level vs portfolio-level decisions
    prompt = f"""You are an AI Portfolio Manager for Indian equity investors.
Your role is to give WEEKLY, ACTIONABLE portfolio decisions — not generic advice.
You must behave like a disciplined fund manager, not a textbook tutor.

IMPORTANT NOTE: This is a STOCK-LEVEL TACTICAL decision. Portfolio-level strategic rebalancing decisions 
are made separately and may differ from this stock-level recommendation. Both can be correct simultaneously 
- you might say "HOLD this stock" while portfolio-level analysis says "reduce sector weight" because the 
stock is strong but the sector is overweight.

CORE PRINCIPLES:
- Diversification is a risk control, not an automatic objective.
- Concentration is acceptable if supported by fundamentals, momentum, and market regime.
- Every recommendation must explain WHAT could change future returns.
- Advice must be concise, conditional, and actionable.

USER PORTFOLIO DATA:
{portfolio_info}
{regime_info}
{benchmark_info}
{narrative_info}

Stock: {stock_name} ({ticker})
Sector: {sector}
User Profile: {user_profile}{budget_info}

Key Metrics:
{metrics_text}

**ANALYSIS FRAMEWORK (MANDATORY):**

1. MARKET CONTEXT
- Current regime: {market_regime.get('regime', 'Unknown') if market_regime else 'Unknown'}
- Sector leadership: {sector_leadership}
- This stock vs sector: {benchmarks.get('vs_sector_1m', 'N/A') if benchmarks else 'N/A'}%
- This stock vs NIFTY: {benchmarks.get('vs_nifty_1m', 'N/A') if benchmarks else 'N/A'}%

2. SECTOR-SPECIFIC FUNDAMENTALS
Evaluate using sector-specific metrics:
{f"Policy context: {sector_narrative.get('current_stance', 'N/A')}" if sector_narrative else ""}

3. STOCK QUALITY ASSESSMENT
- Is this a sector leader or laggard?
- Fundamentals vs momentum alignment
- Any red flags in debt, ROE, or cash flow?

4. NEWS & NARRATIVE INTELLIGENCE
Policy/macro factors affecting this stock:
{f"Tailwinds: {', '.join(sector_narrative.get('tailwinds', []))}" if sector_narrative else ""}
{f"Headwinds: {', '.join(sector_narrative.get('headwinds', []))}" if sector_narrative else ""}

**OUTPUT FORMAT (MANDATORY):**

## WEEKLY ACTION
**Decision: [HOLD / ADD / REDUCE / EXIT]**

Reasoning (2-3 sentences max):
- Fundamentals + momentum alignment
- Market regime fit
- Sector leadership context

## CONDITIONAL TRIGGERS
What would change this decision:
- If [condition] → [Action]
- If [condition] → [Action]

Example: "Reduce if ATF rises over 10% or quarterly PAT falls below X"

## RISK FLAGS (Max 2)
- [Critical risk 1]
- [Critical risk 2]

## CONFIDENCE
[X%] - [One sentence why]

**OUTPUT STYLE RULES:**
- Be concise (total response under 150 words)
- No generic disclaimers
- No "consult advisor" deflections
- Focus on decisions, not explanations
- Take responsibility for the call

**CRITICAL DECISION RULES:**
1. In Risk-On + Sector Leading → NEVER reduce unless fundamentals break
2. Policy tailwinds override short-term technical weakness
3. Concentration over 20% needs strong justification but is ACCEPTABLE if thesis intact
4. Exit only if: thesis broken / better opportunity / regime shift against sector

Remember: You are giving a WEEKLY PORTFOLIO COMMAND for this stock. Be decisive."""

    try:
        response = analyzer.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "AI analysis unavailable. Please check metrics manually."

# Portfolio-level AI analysis
def analyze_portfolio_with_ai(portfolio_df, user_profile, monthly_budget=None):
    """Comprehensive ACTIONABLE portfolio analysis with clear decisions"""
    analyzer = genai.GenerativeModel("gemini-2.5-flash")
    
    # Calculate portfolio metrics
    total_value = portfolio_df['current_value'].sum() if 'current_value' in portfolio_df.columns else 0
    total_investment = portfolio_df['total_investment'].sum() if 'total_investment' in portfolio_df.columns else 0
    total_pl_pct = ((total_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
    
    # Sector concentration
    sector_allocation = portfolio_df.groupby('sector')['current_value'].sum() / total_value * 100 if total_value > 0 else pd.Series()
    sector_summary = "\n".join([f"- {sector}: {pct:.1f}%" for sector, pct in sector_allocation.items()])
    
    # Top holdings
    portfolio_df_sorted = portfolio_df.sort_values('current_value', ascending=False)
    top_5 = portfolio_df_sorted.head(5)
    top_holdings_summary = "\n".join([
        f"- {row['stock_name']}: ₹{row['current_value']:,.0f} ({row['current_value']/total_value*100:.1f}%)"
        for _, row in top_5.iterrows()
    ]) if total_value > 0 else "N/A"
    
    # Risk metrics
    avg_beta = portfolio_df['beta'].mean() if 'beta' in portfolio_df.columns else None
    avg_beta_str = f"{avg_beta:.2f}" if avg_beta and not pd.isna(avg_beta) else "N/A"
    
    high_volatility_count = len(portfolio_df[portfolio_df['volatility_bucket'] == 'High']) if 'volatility_bucket' in portfolio_df.columns else 0
    
    # Momentum analysis
    uptrend_count = len(portfolio_df[portfolio_df['trend_label'].str.contains('Uptrend', na=False)]) if 'trend_label' in portfolio_df.columns else 0
    downtrend_count = len(portfolio_df[portfolio_df['trend_label'].str.contains('Downtrend', na=False)]) if 'trend_label' in portfolio_df.columns else 0
    
    # Concentration risks
    max_stock_weight = portfolio_df['portfolio_weight_pct'].max() if 'portfolio_weight_pct' in portfolio_df.columns else 0
    max_sector_weight = sector_allocation.max() if not sector_allocation.empty else 0
    
    # ISSUE 4 FIX: Add monthly budget context
    budget_context = ""
    if monthly_budget:
        budget_context = f"\nMonthly Investment Budget Available: ₹{monthly_budget:,}"
    
    # ISSUE 3 FIX: Add clarity note
    prompt = f"""You are an AI Portfolio Manager for Indian equity investors.
This is your WEEKLY PORTFOLIO COMMAND CENTER report.

IMPORTANT NOTE: This is PORTFOLIO-LEVEL STRATEGIC guidance. Individual stock-level tactical decisions 
are provided separately and may differ from your sector-level recommendations. For example, you might 
recommend "reduce Financial Services sector exposure" while a specific stock like SBI gets a "HOLD" or 
"ADD" because it's a strong performer within an overweight sector. Both recommendations are valid for 
different purposes.

USER PROFILE: {user_profile}{budget_context}

PORTFOLIO DATA:
- Total Value: Rs {total_value:,.0f}
- Total Investment: Rs {total_investment:,.0f}
- Overall Return: {total_pl_pct:+.2f} percent
- Number of Stocks: {len(portfolio_df)}
- Largest Stock: {max_stock_weight:.1f} percent
- Largest Sector: {max_sector_weight:.1f} percent

SECTOR ALLOCATION:
{sector_summary}

TOP 5 HOLDINGS:
{top_holdings_summary}

PORTFOLIO CHARACTERISTICS:
- Avg Beta: {avg_beta_str}
- High Volatility Stocks: {high_volatility_count} out of {len(portfolio_df)}
- Uptrend: {uptrend_count} | Downtrend: {downtrend_count}

MANDATORY OUTPUT FORMAT:

1. MARKET REGIME ASSESSMENT
Classify: Bull / Range-bound / Risk-off
Signals (2-3): Index trend, sector rotation, breadth

2. PORTFOLIO HEALTH SCORE
Rating: Strong / Neutral / Needs Attention
Justification (1 line): Concentration + momentum + fundamentals summary

3. CRITICAL PORTFOLIO ISSUES (Top 3)
1. Issue + Impact
2. Issue + Impact
3. Issue + Impact

4. PORTFOLIO DIVERSIFICATION RECOMMENDATIONS
Asset Class Allocation:
Evaluate current equity concentration and recommend diversification into:

**Gold ETFs** (e.g., Gold BeES, SBI Gold ETF):
- Recommended allocation: X% of portfolio
- Reason: Hedge against inflation/market volatility
- Specific action: Buy/Hold/Reduce

**India Equity ETFs** (e.g., Nifty 50 ETF, Nifty Next 50):
- Recommended allocation: X% of portfolio  
- Reason: Broader market exposure beyond individual stocks
- Specific action: Buy/Hold/Reduce

**US Equity ETFs** (e.g., Motilal Oswal S&P 500, Mirae Asset NYSE FANG+):
- Recommended allocation: X% of portfolio
- Reason: Geographic diversification, dollar exposure
- Specific action: Buy/Hold/Reduce

Current Portfolio Composition:
- Individual Indian stocks: X%
- Gold/Commodities: X%
- International exposure: X%

Diversification Score: [Good/Needs Improvement/Over-diversified]

Sector-Level Adjustments within Equities:
- Sector: Overweight/Neutral/Underweight - Why

5. CAPITAL DEPLOYMENT DECISION{' (Monthly Budget: ₹' + str(monthly_budget) + ')' if monthly_budget else ''}
{f'''Monthly Capital Available: Rs {monthly_budget:,}

Decision: Deploy Now / Deploy Partially / Wait
- Deploy X percent of monthly capital THIS WEEK (specific amount in rupees)
- Priority sectors: List with reasons
- Avoid sectors: List with reasons
- Specific stocks to add: List with one-line reason each''' if monthly_budget else 'Set monthly budget to get deployment recommendations'}

Reasoning (2 sentences)

6. RISK MANAGEMENT
Position Limits:
- Max single stock: X percent (Current max: {max_stock_weight:.1f} percent)
- Max single sector: X percent (Current max: {max_sector_weight:.1f} percent)

Actions if limits breached:
- Specific action for over-concentrated positions

7. TRIGGERS FOR NEXT WEEK
Portfolio Review Triggered If:
- Market moves plus/minus X percent
- Sector develops specific condition
- News: specific event type

OUTPUT STYLE RULES:
1. Total response under 300 words
2. No disclaimers about consulting advisor
3. Be decisive - you ARE the advisor
4. Assume investor wants you to take responsibility
5. Focus on DECISIONS not explanations
6. Every recommendation must have CONDITIONAL TRIGGERS

CRITICAL RULES:
- Concentration over 40 percent in one sector = urgent action needed
- Concentration over 20 percent in one stock = acceptable ONLY if:
  * Sector is Leading in current regime
  * Fundamentals + momentum aligned
  * Policy tailwinds present
- Never recommend diversify for diversification sake
- Always explain WHAT could change the thesis

Remember: This is a WEEKLY COMMAND CENTER report. Be concise, conditional, and actionable.
"""

    try:
        response = analyzer.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Portfolio analysis unavailable. Please check your data."
    # PART 3 - Portfolio Helper Functions and Complete UI Code
# This continues from Part 2

# Metric explanations dictionary
METRIC_EXPLANATIONS = {
    'current_price': {
        'name': 'Current Price',
        'explanation': 'The latest trading price of the stock. Compare this with your average buy price to see your gains/losses.',
        'good_range': None
    },
    'pe_ratio': {
        'name': 'PE Ratio (Price-to-Earnings)',
        'explanation': 'Shows how much investors are willing to pay for ₹1 of earnings. Lower might mean undervalued, but varies by sector. IT companies often have higher PE than Banks.',
        'sector_context': {
            'IT': 'Normal range: 20-30. High growth expectations.',
            'Banking': 'Normal range: 10-20. Lower due to regulatory nature.',
            'Pharma': 'Normal range: 25-40. R&D intensive sector.',
            'FMCG': 'Normal range: 40-60. Stable, predictable earnings.',
            'Auto': 'Normal range: 15-25. Cyclical industry.'
        }
    },
    'pb_ratio': {
        'name': 'PB Ratio (Price-to-Book)',
        'explanation': 'Compares stock price to company\'s book value (assets minus liabilities). Less than 1 means trading below asset value, but this is normal for some sectors.',
        'sector_context': {
            'IT': 'Often >5. Asset-light, value in intangibles.',
            'Banking': 'Normal: 1-3. Asset-heavy business.',
            'Pharma': 'Often >3. Brand value matters.',
            'FMCG': 'Often >5. Brand strength.',
            'Auto': 'Normal: 2-4. Manufacturing assets.'
        }
    },
    'roe': {
        'name': 'ROE (Return on Equity)',
        'explanation': 'Shows how efficiently company uses shareholder money to generate profits. Higher is generally better, but compare within the same sector.',
        'sector_context': {
            'IT': 'Strong: >25%. Asset-light model.',
            'Banking': 'Good: >15%. Regulated industry.',
            'Pharma': 'Good: >20%. High margin business.',
            'FMCG': 'Strong: >25%. Stable margins.',
            'Auto': 'Good: >15%. Capital intensive.'
        }
    },
    'debt_to_equity': {
        'name': 'Debt-to-Equity Ratio',
        'explanation': 'Shows how much debt company has compared to shareholder equity. Infrastructure and capital-intensive sectors naturally have higher debt.',
        'sector_context': {
            'IT': 'Often <0.5. Low capital needs.',
            'Banking': 'Not applicable. Debt is their business model.',
            'Pharma': 'Good: <1. Moderate capital needs.',
            'FMCG': 'Good: <1. Stable cash flows.',
            'Infrastructure': 'Can be 2-3. Capital intensive.'
        }
    }
}

def get_metric_help(metric_key, sector=None):
    """Get contextual help text for a metric"""
    if metric_key not in METRIC_EXPLANATIONS:
        return None
    
    info = METRIC_EXPLANATIONS[metric_key]
    help_text = f"**{info['name']}**\n\n{info['explanation']}"
    
    if sector and 'sector_context' in info:
        sector_info = info['sector_context'].get(sector)
        if sector_info:
            help_text += f"\n\n**For {sector} sector:** {sector_info}"
        else:
            help_text += f"\n\n**Sector context:** Compare with other {sector} companies."
    
    if info.get('good_range'):
        help_text += f"\n\n**General guideline:** {info['good_range']}"
    
    return help_text

# Portfolio helpers
def empty_portfolio_df():
    return pd.DataFrame(columns=["stock_name", "ticker", "quantity", "avg_buy_price", "sector"])

def get_stock_info(stock_name):
    """Get ticker and sector using AI"""
    enricher = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f'Identify NSE ticker and sector for "{stock_name}". Return JSON: {{"ticker": "TCS", "sector": "IT"}}. No markdown.'
    
    try:
        resp = enricher.generate_content(prompt)
        text = resp.text.strip().strip("`").replace("json", "").strip()
        data = json.loads(text)
        ticker = clean_ticker_symbol(data.get("ticker", stock_name.upper()))
        sector = data.get("sector", "Unknown")
        return ticker, sector
    except Exception:
        return clean_ticker_symbol(stock_name.upper()), "Unknown"

def smart_detect_csv_columns(cols):
    stock_kw = ['stock', 'name', 'symbol', 'ticker', 'company']
    qty_kw = ['qty', 'quantity', 'shares', 'units', 'nos']
    price_kw = ['price', 'avg', 'average', 'buy', 'cost']
    
    norm = {c.lower().strip().replace('_', '').replace(' ', ''): c for c in cols}
    detected = {'stock': None, 'quantity': None, 'price': None}
    
    for n, o in norm.items():
        if not detected['stock'] and any(k in n for k in stock_kw) and not any(k in n for k in qty_kw):
            detected['stock'] = o
        if not detected['quantity'] and any(k in n for k in qty_kw):
            detected['quantity'] = o
        if not detected['price'] and any(k in n for k in price_kw) and 'current' not in n:
            detected['price'] = o
    
    if not detected['stock'] and cols: detected['stock'] = cols[0]
    if not detected['quantity'] and len(cols) > 1: detected['quantity'] = cols[1]
    if not detected['price'] and len(cols) > 2: detected['price'] = cols[2]
    
    return detected

def parse_csv_file(file):
    try:
        raw = pd.read_csv(file)
        if raw.empty: 
            return empty_portfolio_df(), "CSV is empty"
        
        st.info(f"📋 Columns: {', '.join(raw.columns)}")
        detected = smart_detect_csv_columns(raw.columns.tolist())
        
        if not detected['stock'] or not detected['quantity']:
            return empty_portfolio_df(), "Could not detect Stock/Quantity columns"
        
        st.success(f"✅ Stock→{detected['stock']} | Qty→{detected['quantity']} | Price→{detected['price'] or 'N/A'}")
        
        df = empty_portfolio_df()
        
        for _, row in raw.iterrows():
            name = str(row[detected['stock']]).strip() if pd.notna(row[detected['stock']]) else ""
            if not name: 
                continue
            
            qty = int(float(row[detected['quantity']])) if pd.notna(row[detected['quantity']]) else None
            price = float(row[detected['price']]) if detected['price'] and pd.notna(row.get(detected['price'])) else None
            
            ticker, sector = get_stock_info(name)
            
            new_row = {
                "stock_name": name,
                "ticker": ticker,
                "quantity": qty,
                "avg_buy_price": price,
                "sector": sector
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        return df, ""
    except Exception as e:
        return empty_portfolio_df(), f"Error: {e}"

# Analyze portfolio
def analyze_portfolio(portfolio_df):
    """Fetch fundamental metrics for all stocks with benchmark comparisons"""
    if portfolio_df.empty:
        return None
    
    # Fetch NIFTY 50 benchmark once
    st.info("Fetching NIFTY 50 benchmark data...")
    nifty_returns = fetch_index_returns("NIFTY 50")
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    successful = 0
    failed_stocks = []
    
    # First pass: get all metrics
    for idx, row in portfolio_df.iterrows():
        status.text(f"Fetching metrics for {row['stock_name']} ({row['ticker']})... ({idx+1}/{len(portfolio_df)})")
        progress.progress((idx+1)/len(portfolio_df))
        
        metrics = fetch_stock_metrics_yf(row['ticker']) if YFINANCE_AVAILABLE else None
        
        if metrics:
            successful += 1
            result = {
                "stock_name": row['stock_name'],
                "ticker": row['ticker'],
                "sector": row['sector'],
                "quantity": row['quantity'],
                "avg_buy_price": row['avg_buy_price']
            }
            result.update(metrics)
            
            # Fetch sector benchmark
            sector_index = SECTOR_TO_INDEX.get(row['sector'], "NIFTY 50")
            sector_returns = fetch_index_returns(sector_index)
            
            # Add benchmark data
            if nifty_returns:
                result['nifty_1w'] = nifty_returns.get('1w')
                result['nifty_1m'] = nifty_returns.get('1m')
                result['nifty_3m'] = nifty_returns.get('3m')
            
            if sector_returns:
                result['sector_1w'] = sector_returns.get('1w')
                result['sector_1m'] = sector_returns.get('1m')
                result['sector_3m'] = sector_returns.get('3m')
                result['sector_index_name'] = sector_index
            
            # Calculate relative performance
            stock_1w = result.get('1w_change_pct')
            stock_1m = result.get('1m_change_pct')
            stock_3m = result.get('3m_change_pct')
            
            if stock_1m and result.get('sector_1m'):
                result['vs_sector_1m'] = round(stock_1m - result['sector_1m'], 2)
            
            if stock_1m and result.get('nifty_1m'):
                result['vs_nifty_1m'] = round(stock_1m - result['nifty_1m'], 2)
            
            # Relative performance verdict
            if result.get('vs_sector_1m') is not None:
                if result['vs_sector_1m'] > 2:
                    result['sector_verdict'] = "Outperforming"
                elif result['vs_sector_1m'] < -2:
                    result['sector_verdict'] = "Underperforming"
                else:
                    result['sector_verdict'] = "In-line"
            
            if result.get('vs_nifty_1m') is not None:
                if result['vs_nifty_1m'] > 2:
                    result['market_verdict'] = "Outperforming"
                elif result['vs_nifty_1m'] < -2:
                    result['market_verdict'] = "Underperforming"
                else:
                    result['market_verdict'] = "In-line"
            
            # Calculate unrealized P&L
            if metrics['current_price'] and row['avg_buy_price']:
                result['total_investment'] = row['avg_buy_price'] * row['quantity']
                result['current_value'] = metrics['current_price'] * row['quantity']
                result['unrealized_pl'] = result['current_value'] - result['total_investment']
                result['unrealized_pl_pct'] = (result['unrealized_pl'] / result['total_investment']) * 100
            
            results.append(result)
            st.session_state.stock_metrics[row['ticker']] = metrics
        else:
            failed_stocks.append(row['stock_name'])
        
        time.sleep(0.5)
    
    progress.empty()
    status.empty()
    
    if failed_stocks:
        st.warning(f"Could not fetch data for: {', '.join(failed_stocks)}")
        st.info("You can add metrics manually below")
    
    if successful:
        st.success(f"Successfully fetched data for {successful} stocks with benchmark comparisons")
    
    if not results:
        st.error("No data available. Please add metrics manually.")
        return None
    
    result_df = pd.DataFrame(results)
    
    # Calculate portfolio-level context for each stock
    total_portfolio_value = result_df['current_value'].sum() if 'current_value' in result_df.columns else 0
    
    for idx in result_df.index:
        stock_value = result_df.loc[idx, 'current_value'] if 'current_value' in result_df.columns else 0
        sector = result_df.loc[idx, 'sector']
        
        # Stock weight
        if total_portfolio_value > 0:
            result_df.loc[idx, 'portfolio_weight_pct'] = (stock_value / total_portfolio_value) * 100
        
        # Sector weight
        sector_total = result_df[result_df['sector'] == sector]['current_value'].sum() if 'current_value' in result_df.columns else 0
        if total_portfolio_value > 0:
            result_df.loc[idx, 'sector_weight_pct'] = (sector_total / total_portfolio_value) * 100
    
    return result_df

# ========================================
# UI CODE STARTS HERE
# ========================================

tab_agent, tab_portfolio = st.tabs(["💬 Finance Agent", "📊 Portfolio Analysis"])

with tab_agent:
    st.subheader("Personal Finance Assistant (India)")
    user_input = st.chat_input("Ask about your finances...")
    
    if user_input:
        st.session_state.chat.append(("user", user_input))
        infer_profile_with_ai(user_input)
        
        profile_summary = get_profile_summary()
        full_system = SYSTEM_PROMPT.format(PROFILE_SUMMARY=profile_summary)
        conversation = full_system + "\n\n" + "\n".join([f"{r}: {m}" for r, m in st.session_state.chat])
        
        resp = chat_model.generate_content(conversation)
        st.session_state.chat.append(("assistant", resp.text))
    
    with st.expander("👤 Your Profile"):
        p = st.session_state.profile
        cols = st.columns(4)
        for i, (k, v) in enumerate(p.items()):
            cols[i].metric(k.replace('_', ' ').title(), v or "❓")
    
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

with tab_portfolio:
    st.subheader("Portfolio Fundamental Analysis")
    
    # ISSUE 4 FIX: Add monthly budget input at the top
    st.markdown("### 💰 Monthly Investment Planning")
    col1, col2 = st.columns([3, 1])
    with col1:
        monthly_budget_input = st.number_input(
            "How much can you invest this month? (₹)",
            min_value=0,
            step=1000,
            value=st.session_state.monthly_budget if st.session_state.monthly_budget else 0,
            help="AI will provide specific deployment recommendations based on your budget"
        )
        if monthly_budget_input > 0:
            st.session_state.monthly_budget = int(monthly_budget_input)
            st.success(f"✅ Monthly budget set: ₹{st.session_state.monthly_budget:,}")
        elif st.session_state.monthly_budget:
            st.info(f"Current budget: ₹{st.session_state.monthly_budget:,}")
    
    st.markdown("---")
    
    if not YFINANCE_AVAILABLE:
        st.error("⚠️ yfinance library has compatibility issues with Python 3.9")
        st.info("💡 Solutions: (1) Use manual entry, OR (2) Upgrade to Python 3.10+")
        st.code("pip install --upgrade yfinance", language="bash")
    else:
        st.info("📊 Analyze your portfolio with Price, Valuation, Quality & Risk metrics")
    
    if 'parsed_df_temp' not in st.session_state:
        st.session_state.parsed_df_temp = None
    
    # ISSUE 1 FIX: Manual entry tab with option to add to existing portfolio
    tab_manual, tab_csv = st.tabs(["📝 Manual Entry", "📄 CSV Upload"])
    
    with tab_manual:
        if 'stock_entries' not in st.session_state:
            st.session_state.stock_entries = []
        
        # Add individual stock form
        with st.form("add_stock"):
            st.markdown("#### Add Stock to Portfolio")
            cols = st.columns(3)
            name = cols[0].text_input("Stock Name *", help="e.g., TCS, Reliance, HDFC Bank")
            qty = cols[1].number_input("Shares *", min_value=1, step=1, value=1)
            price = cols[2].number_input("Avg Buy Price (₹) *", min_value=0.0, step=0.01, value=100.0)
            
            submitted = st.form_submit_button("➕ Add Stock")
            if submitted:
                if name and qty and price:
                    st.session_state.stock_entries.append({'name': name, 'quantity': qty, 'price': price})
                    st.success(f"✅ Added {name} to list")
                    st.rerun()
                else:
                    st.error("Please fill all fields")
        
        # Show current entries
        if st.session_state.stock_entries:
            st.markdown("#### Stocks to Add:")
            for idx, e in enumerate(st.session_state.stock_entries):
                col1, col2 = st.columns([5, 1])
                col1.write(f"**{e['name']}** - {e['quantity']} shares @ ₹{e['price']}")
                if col2.button("🗑️", key=f"del_{idx}"):
                    st.session_state.stock_entries.pop(idx)
                    st.rerun()
            
            col1, col2 = st.columns(2)
            # ISSUE 1 FIX: Two buttons - one to create new, one to add to existing
            if not st.session_state.portfolio.empty:
                if col1.button("➕ Add to Existing Portfolio", type="primary"):
                    for e in st.session_state.stock_entries:
                        ticker, sector = get_stock_info(e['name'])
                        row = {
                            "stock_name": e['name'],
                            "ticker": ticker,
                            "quantity": int(e['quantity']),
                            "avg_buy_price": float(e['price']),
                            "sector": sector
                        }
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([row])], ignore_index=True)
                    st.session_state.stock_entries = []
                    st.success("✅ Stocks added to portfolio!")
                    st.rerun()
            
            if col2.button("🔄 Create New Portfolio", type="secondary"):
                df = empty_portfolio_df()
                for e in st.session_state.stock_entries:
                    ticker, sector = get_stock_info(e['name'])
                    row = {
                        "stock_name": e['name'],
                        "ticker": ticker,
                        "quantity": int(e['quantity']),
                        "avg_buy_price": float(e['price']),
                        "sector": sector
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                st.session_state.portfolio = df
                st.session_state.stock_entries = []
                st.success("✅ New portfolio created!")
                st.rerun()
    
    with tab_csv:
        file = st.file_uploader("Upload CSV (Stock Name, Quantity, Avg Price)", type=["csv"])
        if st.button("📤 Parse CSV") and file:
            st.session_state.parsed_df_temp, err = parse_csv_file(file)
            if err: 
                st.error(err)
            else:
                st.rerun()
    
    st.markdown("---")
    st.subheader("📋 Portfolio Preview")
    
    if st.session_state.parsed_df_temp is not None:
        st.session_state.portfolio = st.session_state.parsed_df_temp
        st.session_state.confirmed_portfolio = False
        st.session_state.parsed_df_temp = None
    
    if not st.session_state.portfolio.empty:
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        with st.expander("✏️ Edit Portfolio"):
            st.markdown("### Manage Stocks")
            
            for idx, row in st.session_state.portfolio.iterrows():
                col1, col2 = st.columns([5, 1])
                
                ticker = row.get('ticker', 'N/A')
                quantity = row.get('quantity', 0)
                avg_price = row.get('avg_buy_price', 0)
                
                price_str = f"₹{avg_price:.2f}" if avg_price else "N/A"
                
                col1.write(f"**{row['stock_name']}** ({ticker}) - {quantity} shares @ {price_str}")
                
                if col2.button("🗑️", key=f"delete_{idx}"):
                    st.session_state.portfolio = st.session_state.portfolio.drop(idx).reset_index(drop=True)
                    st.success(f"Deleted {row['stock_name']}")
                    st.rerun()
        
        if not st.session_state.confirmed_portfolio:
            if st.button("✅ Confirm Portfolio", type="primary"):
                st.session_state.confirmed_portfolio = True
                st.success("✓ Portfolio confirmed!")
                st.rerun()
        else:
            st.markdown("---")
            st.subheader("📈 Fundamental Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🔄 Fetch Stock Metrics", type="primary"):
                    with st.spinner("Fetching fundamental data..."):
                        analysis_df = analyze_portfolio(st.session_state.portfolio.copy())
                        if analysis_df is not None:
                            st.session_state.analysis_df = analysis_df
                            st.success("✓ Analysis complete!")
                            st.rerun()
            
            if 'analysis_df' in st.session_state and st.session_state.analysis_df is not None:
                df = st.session_state.analysis_df
                
                # Calculate market regime
                nifty_returns_dict = None
                if 'nifty_1m' in df.columns and not df['nifty_1m'].isna().all():
                    nifty_returns_dict = {
                        '1w': df['nifty_1w'].iloc[0] if 'nifty_1w' in df.columns else None,
                        '1m': df['nifty_1m'].iloc[0] if 'nifty_1m' in df.columns else None,
                        '3m': df['nifty_3m'].iloc[0] if 'nifty_3m' in df.columns else None
                    }
                
                regime, regime_desc = classify_market_regime(nifty_returns_dict)
                
                # ISSUE 2: Display Market Regime with explanation
                st.markdown("### 🌐 Market Regime Analysis")
                if regime == "Risk-On":
                    st.success(f"**{regime}**")
                elif regime == "Risk-Off":
                    st.error(f"**{regime}**")
                elif regime == "Range-bound":
                    st.warning(f"**{regime}**")
                else:
                    st.info(f"**{regime}**")
                
                st.markdown(regime_desc)
                st.markdown("---")
                
                # PORTFOLIO-LEVEL AI ANALYSIS
                st.markdown("### 🤖 AI Portfolio Analysis (Strategic Level)")
                # ISSUE 3: Add note about strategic vs tactical
                st.info("💡 **Note:** This provides PORTFOLIO-LEVEL strategic guidance (sector allocation, rebalancing). Individual stock recommendations below may differ - both are valid for different purposes!")
                
                with st.spinner("Analyzing your complete portfolio..."):
                    profile_summary = get_profile_summary()
                    # ISSUE 4: Pass monthly budget to portfolio AI
                    portfolio_analysis = analyze_portfolio_with_ai(df, profile_summary, st.session_state.monthly_budget)
                    st.markdown(portfolio_analysis)
                
                st.markdown("---")
                
                st.markdown("### 💼 Portfolio Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                total_inv = df['total_investment'].sum() if 'total_investment' in df.columns else 0
                total_curr = df['current_value'].sum() if 'current_value' in df.columns else 0
                total_pl = df['unrealized_pl'].sum() if 'unrealized_pl' in df.columns else 0
                total_pl_pct = (total_pl / total_inv * 100) if total_inv > 0 else 0
                
                col1.metric("Total Investment", f"₹{total_inv:,.0f}")
                col2.metric("Current Value", f"₹{total_curr:,.0f}")
                col3.metric("Unrealized P&L", f"₹{total_pl:,.0f}", delta=f"{total_pl_pct:+.2f}%")
                col4.metric("Stocks", len(df))
                
                st.markdown("---")
                
                # ISSUE 3 FIX: Add AI chat interface for clarifications and counter-recommendations
                st.markdown("### 💬 Ask AI for Clarification or Discuss Recommendations")
                st.info("💡 Got questions about the portfolio analysis? Want to discuss a counter-recommendation? Ask here!")
                
                # Initialize chat history for portfolio discussion
                if 'portfolio_chat' not in st.session_state:
                    st.session_state.portfolio_chat = []
                
                # Chat input
                user_question = st.text_input(
                    "Your question or counter-recommendation:",
                    placeholder="e.g., 'Why should I reduce SRIRAM finance when fundamentals look good?' or 'What if I want to hold more tech stocks?'",
                    key="portfolio_chat_input"
                )
                
                if st.button("Ask AI", type="primary") and user_question:
                    # Add user question to chat
                    st.session_state.portfolio_chat.append(("user", user_question))
                    
                    # Build context for AI
                    portfolio_summary = f"""
Portfolio Summary:
- Total Value: ₹{total_curr:,.0f}
- Total P&L: {total_pl_pct:+.2f}%
- Number of stocks: {len(df)}
- Market Regime: {regime}

Recent AI Recommendations Summary:
{portfolio_analysis[:500]}...
"""
                    
                    # Create clarification prompt
                    clarification_prompt = f"""You are an AI Portfolio Manager having a discussion with an investor about their portfolio.

Context:
{portfolio_summary}

User Profile: {get_profile_summary()}

Conversation History:
{chr(10).join([f"{role}: {msg}" for role, msg in st.session_state.portfolio_chat[-4:]])}

User's Question/Counter-Recommendation:
{user_question}

Instructions:
1. If the user is asking for clarification, explain your reasoning clearly and concisely
2. If the user is proposing a counter-recommendation, evaluate it objectively:
   - Acknowledge valid points in their argument
   - Explain potential risks or benefits they might not have considered
   - Provide data-driven reasoning
   - Be open to changing your recommendation if their points are strong
3. Be conversational but professional
4. Keep response under 200 words
5. No disclaimers - engage directly with their question

Respond:"""
                    
                    try:
                        discussion_model = genai.GenerativeModel("gemini-2.5-flash")
                        response = discussion_model.generate_content(clarification_prompt)
                        ai_response = response.text.strip()
                        st.session_state.portfolio_chat.append(("assistant", ai_response))
                    except Exception:
                        ai_response = "I'm having trouble processing that question. Could you rephrase it?"
                        st.session_state.portfolio_chat.append(("assistant", ai_response))
                    
                    st.rerun()
                
                # Display chat history
                if st.session_state.portfolio_chat:
                    st.markdown("#### Discussion History:")
                    for role, msg in st.session_state.portfolio_chat:
                        if role == "user":
                            st.markdown(f"**You:** {msg}")
                        else:
                            st.markdown(f"**AI Portfolio Manager:** {msg}")
                        st.markdown("---")
                    
                    if st.button("Clear Discussion"):
                        st.session_state.portfolio_chat = []
                        st.rerun()
                
                st.markdown("---")
                
                st.markdown("### 📊 Stock-wise Analysis (Tactical Level)")
                st.info("💡 These are STOCK-LEVEL tactical decisions for each holding.")
                
                if 'unrealized_pl_pct' in df.columns:
                    df_sorted = df.sort_values('unrealized_pl_pct', ascending=False)
                else:
                    df_sorted = df
                
                for _, row in df_sorted.iterrows():
                    pl_pct = row.get('unrealized_pl_pct', 0)
                    if pl_pct > 10:
                        emoji = "🟢"
                    elif pl_pct > 0:
                        emoji = "🟡"
                    elif pl_pct < -10:
                        emoji = "🔴"
                    else:
                        emoji = "🟠"
                    
                    with st.expander(f"{emoji} **{row['stock_name']}** ({row['ticker']}) - Sector: {row['sector']}", expanded=False):
                        
                        # Build portfolio context for AI
                        portfolio_context = {
                            'total_value': total_curr,
                            'stock_weight_pct': row.get('portfolio_weight_pct', 0),
                            'sector_weight_pct': row.get('sector_weight_pct', 0),
                            'num_stocks': len(df)
                        }
                        
                        # Build benchmark context for AI
                        benchmarks = {
                            'stock_1w': row.get('1w_change_pct'),
                            'stock_1m': row.get('1m_change_pct'),
                            'stock_3m': row.get('3m_change_pct'),
                            'sector_1w': row.get('sector_1w'),
                            'sector_1m': row.get('sector_1m'),
                            'sector_3m': row.get('sector_3m'),
                            'nifty_1w': row.get('nifty_1w'),
                            'nifty_1m': row.get('nifty_1m'),
                            'nifty_3m': row.get('nifty_3m'),
                            'vs_sector_1m': row.get('vs_sector_1m'),
                            'vs_nifty_1m': row.get('vs_nifty_1m')
                        }
                        
                        # Show benchmark comparison first
                        st.markdown("### 📊 Benchmark Comparison")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.markdown("**1-Week Returns**")
                        col1.metric("Stock", f"{row.get('1w_change_pct', 'N/A'):+.2f}%" if row.get('1w_change_pct') else "N/A")
                        col1.metric("Sector", f"{row.get('sector_1w', 'N/A'):+.2f}%" if row.get('sector_1w') else "N/A")
                        col1.metric("NIFTY 50", f"{row.get('nifty_1w', 'N/A'):+.2f}%" if row.get('nifty_1w') else "N/A")
                        
                        col2.markdown("**1-Month Returns**")
                        col2.metric("Stock", f"{row.get('1m_change_pct', 'N/A'):+.2f}%" if row.get('1m_change_pct') else "N/A")
                        col2.metric("Sector", f"{row.get('sector_1m', 'N/A'):+.2f}%" if row.get('sector_1m') else "N/A")
                        col2.metric("NIFTY 50", f"{row.get('nifty_1m', 'N/A'):+.2f}%" if row.get('nifty_1m') else "N/A")
                        
                        col3.markdown("**3-Month Returns**")
                        col3.metric("Stock", f"{row.get('3m_change_pct', 'N/A'):+.2f}%" if row.get('3m_change_pct') else "N/A")
                        col3.metric("Sector", f"{row.get('sector_3m', 'N/A'):+.2f}%" if row.get('sector_3m') else "N/A")
                        col3.metric("NIFTY 50", f"{row.get('nifty_3m', 'N/A'):+.2f}%" if row.get('nifty_3m') else "N/A")
                        
                        # Relative performance verdict
                        st.markdown("### 🎯 Relative Performance Verdict")
                        col1, col2 = st.columns(2)
                        
                        sector_verdict = row.get('sector_verdict', 'Unknown')
                        vs_sector = row.get('vs_sector_1m')
                        
                        if sector_verdict == "Outperforming":
                            col1.success(f"**vs Sector ({row.get('sector_index_name', 'Index')})**: Outperforming (+{vs_sector:.2f}%)")
                        elif sector_verdict == "Underperforming":
                            col1.error(f"**vs Sector ({row.get('sector_index_name', 'Index')})**: Underperforming ({vs_sector:+.2f}%)")
                        else:
                            col1.info(f"**vs Sector ({row.get('sector_index_name', 'Index')})**: In-line ({vs_sector:+.2f}%)")
                        
                        market_verdict = row.get('market_verdict', 'Unknown')
                        vs_nifty = row.get('vs_nifty_1m')
                        
                        if market_verdict == "Outperforming":
                            col2.success(f"**vs NIFTY 50**: Outperforming (+{vs_nifty:.2f}%)")
                        elif market_verdict == "Underperforming":
                            col2.error(f"**vs NIFTY 50**: Underperforming ({vs_nifty:+.2f}%)")
                        else:
                            col2.info(f"**vs NIFTY 50**: In-line ({vs_nifty:+.2f}%)")
                        
                        st.markdown("---")
                        
                        st.markdown("### 🤖 AI Weekly Action Plan with Market Context")
                        with st.spinner("Analyzing stock with market regime and narrative intelligence..."):
                            metrics_for_ai = {
                                'Current Price': row.get('current_price'),
                                'Trend': row.get('trend_label', 'Unknown'),
                                'Volatility': f"{row.get('volatility_bucket', 'Unknown')} ({row.get('volatility_pct', 'N/A')}%)",
                                'Beta': row.get('beta'),
                                'Drawdown from 52W High': f"{row.get('drawdown_from_high', 'N/A')}%",
                                'PE Ratio': row.get('pe_ratio'),
                                'PB Ratio': row.get('pb_ratio'),
                                'ROE': row.get('roe'),
                                'Debt-to-Equity': row.get('debt_to_equity'),
                                'Market Cap (Cr)': row.get('market_cap_cr'),
                                'Dividend Yield': row.get('dividend_yield'),
                                'Earnings Proximity': row.get('earnings_proximity', 'Unknown'),
                                'Portfolio Weight': f"{row.get('portfolio_weight_pct', 0):.1f}%",
                                'Sector Weight': f"{row.get('sector_weight_pct', 0):.1f}%"
                            }
                            
                            # Get sector narrative
                            sector_narrative = get_sector_narrative(row['sector'])
                            
                            # Build market regime dict
                            market_regime_dict = {
                                'regime': regime,
                                'description': regime_desc
                            }
                            
                            profile_summary = get_profile_summary()
                            # ISSUE 4: Pass monthly budget to stock AI
                            ai_analysis = analyze_stock_with_ai(
                                row['stock_name'], 
                                row['ticker'], 
                                row['sector'], 
                                metrics_for_ai,
                                profile_summary,
                                portfolio_context,
                                benchmarks,
                                market_regime_dict,
                                sector_narrative,
                                st.session_state.monthly_budget
                            )
                            st.markdown(ai_analysis)
                        
                        st.markdown("---")
                        
                        # ISSUE 5 & 2 FIX: Add news section with latest date-aware search
                        st.markdown("### 📰 Recent News & Developments")
                        
                        # Add manual search option
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button("🔄 Refresh News", key=f"refresh_news_{row['ticker']}"):
                                st.rerun()
                        
                        with st.spinner("Searching for latest news from credible sources..."):
                            news_analysis = fetch_stock_news(row['stock_name'], row['ticker'], row['sector'])
                            st.markdown(news_analysis)
                            st.caption("*AI-searched sources: Moneycontrol, Economic Times, Business Standard, Reuters, Livemint. For real-time updates, visit these sites directly.*")
                        
                        # Add direct links to news sources
                        st.markdown("**Quick Links for Latest News:**")
                        search_term = row['stock_name'].replace(' ', '+')
                        col1, col2, col3 = st.columns(3)
                        col1.markdown(f"[Moneycontrol](https://www.moneycontrol.com/stocks/cptmarket/compsearchnew.php?search_data=&cid=&mbsearch_str={search_term})")
                        col2.markdown(f"[Economic Times](https://economictimes.indiatimes.com/topic/{search_term})")
                        col3.markdown(f"[Google News](https://news.google.com/search?q={search_term}+stock)")
                        
                        st.markdown("---")
                        
                        # Continue to Part 4 for remaining stock details UI...
                        # PART 4 - Remaining Stock Details UI (Final Part)
# This continues from Part 3 inside the stock expander loop

                        # Momentum & Trend Section
                        st.markdown("### 📈 Momentum & Trend Analysis")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        w1_change = row.get('1w_change_pct')
                        col1.metric("1-Week Change", f"{w1_change:+.2f}%" if w1_change is not None else "N/A")
                        
                        m1_change = row.get('1m_change_pct')
                        col2.metric("1-Month Change", f"{m1_change:+.2f}%" if m1_change is not None else "N/A")
                        
                        m3_change = row.get('3m_change_pct')
                        col3.metric("3-Month Change", f"{m3_change:+.2f}%" if m3_change is not None else "N/A")
                        
                        trend = row.get('trend_label', 'Unknown')
                        col4.metric("Trend", trend)
                        
                        # Visual trend indicator
                        if 'Strong Uptrend' in trend:
                            st.success(f"📈 {trend} - Positive momentum across timeframes")
                        elif 'Downtrend' in trend:
                            st.error(f"📉 {trend} - Negative momentum")
                        elif 'Sideways' in trend:
                            st.info(f"↔️ {trend} - No clear direction")
                        else:
                            st.warning(f"⚠️ {trend}")
                        
                        st.markdown("---")
                        
                        # Risk Metrics Section
                        st.markdown("### ⚠️ Risk Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        beta_val = row.get('beta')
                        if beta_val:
                            col1.metric("Beta", f"{beta_val:.2f}",
                                      help="<1: Less volatile than market, >1: More volatile")
                            if beta_val > 1.5:
                                col1.warning("High market sensitivity")
                            elif beta_val < 0.5:
                                col1.success("Low market sensitivity")
                        else:
                            col1.metric("Beta", "N/A")
                        
                        vol_bucket = row.get('volatility_bucket', 'Unknown')
                        vol_pct = row.get('volatility_pct')
                        col2.metric("Volatility", vol_bucket, 
                                  help=f"Annualized: {vol_pct:.1f}%" if vol_pct else "Price fluctuation level")
                        
                        drawdown = row.get('drawdown_from_high')
                        if drawdown:
                            col3.metric("Drawdown from High", f"{drawdown:.1f}%",
                                      help="How far below 52-week high")
                            if drawdown > 30:
                                col3.error("Significant drawdown")
                            elif drawdown > 15:
                                col3.warning("Moderate drawdown")
                        
                        stock_weight = row.get('portfolio_weight_pct', 0)
                        col4.metric("Portfolio Weight", f"{stock_weight:.1f}%",
                                  help="% of your total portfolio")
                        if stock_weight > 20:
                            col4.error("⚠️ Over-concentrated!")
                        elif stock_weight > 15:
                            col4.warning("⚠️ High concentration")
                        
                        st.markdown("---")
                        
                        # Events & Timing
                        st.markdown("### 📅 Events & Timing")
                        col1, col2 = st.columns(2)
                        
                        earnings_prox = row.get('earnings_proximity', 'Unknown')
                        col1.metric("Earnings", earnings_prox)
                        if 'This week' in earnings_prox or 'Within 2 weeks' in earnings_prox:
                            col1.warning("⚠️ Earnings coming up - expect volatility")
                        
                        recent_div = row.get('recent_dividend', False)
                        col2.metric("Recent Dividend", "Yes ✓" if recent_div else "No")
                        
                        st.markdown("---")
                        
                        st.markdown("### 📌 Quick Overview")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        current = row.get('current_price')
                        col1.metric("Current Price", f"₹{current:,.2f}" if current else "N/A", 
                                   delta=f"{pl_pct:+.2f}%" if pl_pct else None,
                                   help=get_metric_help('current_price'))
                        
                        pe = row.get('pe_ratio')
                        col2.metric("PE Ratio", f"{pe:.2f}" if pe else "N/A", 
                                   help=get_metric_help('pe_ratio', row['sector']))
                        
                        roe_val = row.get('roe')
                        col3.metric("ROE", f"{roe_val:.2f}%" if roe_val else "N/A",
                                   help=get_metric_help('roe', row['sector']))
                        
                        debt = row.get('debt_to_equity')
                        col4.metric("Debt/Equity", f"{debt:.2f}" if debt else "N/A",
                                   help=get_metric_help('debt_to_equity', row['sector']))
                        
                        div_yield_val = row.get('dividend_yield')
                        col5.metric("Div Yield", f"{div_yield_val:.2f}%" if div_yield_val else "N/A",
                                   help=get_metric_help('dividend_yield', row['sector']))
                        
                        st.markdown("---")
                        
                        if 'unrealized_pl' in row and row['unrealized_pl'] is not None:
                            st.markdown("### 💼 Your Investment")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Shares", f"{row['quantity']:,}")
                            col2.metric("Avg Buy Price", f"₹{row['avg_buy_price']:,.2f}")
                            col3.metric("Invested", f"₹{row['total_investment']:,.0f}")
                            
                            pl_value = row['unrealized_pl']
                            pl_color = "normal" if pl_value >= 0 else "inverse"
                            col4.metric("Profit/Loss", 
                                      f"₹{pl_value:,.0f}", 
                                      delta=f"{row['unrealized_pl_pct']:+.2f}%",
                                      delta_color=pl_color)
                            
                            current_val = row.get('current_value', 0)
                            if current_val > 0:
                                pl_bar_pct = min(abs(pl_pct), 100)
                                if pl_pct >= 0:
                                    st.success(f"📈 Gaining {pl_pct:.2f}%")
                                    st.progress(min(pl_bar_pct / 100, 1.0))
                                else:
                                    st.error(f"📉 Down {pl_pct:.2f}%")
                                    st.progress(min(pl_bar_pct / 100, 1.0))
                            
                            st.markdown("---")
                        
                        tab1, tab2, tab3 = st.tabs(["💰 Price Analysis", "📊 Valuation", "⚖️ Quality & Risk"])
                        
                        with tab1:
                            st.markdown("#### Price Movements")
                            col1, col2, col3 = st.columns(3)
                            
                            col1.metric("Today's High", f"₹{row.get('day_high', 0):,.2f}" if row.get('day_high') else "N/A")
                            col2.metric("Today's Low", f"₹{row.get('day_low', 0):,.2f}" if row.get('day_low') else "N/A")
                            
                            day_high = row.get('day_high', 0)
                            day_low = row.get('day_low', 0)
                            if day_high and day_low and current:
                                day_range = day_high - day_low
                                if day_range > 0:
                                    position = ((current - day_low) / day_range) * 100
                                    col3.metric("Position in Day Range", f"{position:.0f}%",
                                              help="Shows where current price sits in today's trading range")
                            
                            st.markdown("#### 52-Week Range")
                            col1, col2 = st.columns(2)
                            
                            w52_high = row.get('52w_high', 0)
                            w52_low = row.get('52w_low', 0)
                            
                            col1.metric("52W High", f"₹{w52_high:,.2f}" if w52_high else "N/A",
                                      help=get_metric_help('52w_high'))
                            col2.metric("52W Low", f"₹{w52_low:,.2f}" if w52_low else "N/A",
                                      help=get_metric_help('52w_low'))
                            
                            if w52_high and w52_low and current:
                                range_52w = w52_high - w52_low
                                if range_52w > 0:
                                    position_52w = ((current - w52_low) / range_52w) * 100
                                    st.markdown(f"**Current position in 52W range: {position_52w:.1f}%**")
                                    st.progress(position_52w / 100)
                                    
                                    if position_52w > 80:
                                        st.info("📈 Trading near yearly highs. Check if fundamentals support the price.")
                                    elif position_52w < 20:
                                        st.warning("📉 Trading near yearly lows. Check AI analysis to understand why.")
                            
                            col1, col2 = st.columns(2)
                            pct_high = row.get('pct_from_52w_high')
                            pct_low = row.get('pct_from_52w_low')
                            
                            col1.metric("% from 52W High", f"{pct_high:+.2f}%" if pct_high is not None else "N/A",
                                      help=get_metric_help('pct_from_52w_high', row['sector']))
                            col2.metric("% from 52W Low", f"{pct_low:+.2f}%" if pct_low is not None else "N/A",
                                      help=get_metric_help('pct_from_52w_low', row['sector']))
                        
                        with tab2:
                            st.markdown("#### Market Valuation")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            pe_val = row.get('pe_ratio')
                            col1.metric("PE Ratio", f"{pe_val:.2f}" if pe_val else "N/A",
                                      help=get_metric_help('pe_ratio', row['sector']))
                            
                            pb_val = row.get('pb_ratio')
                            col2.metric("PB Ratio", f"{pb_val:.2f}" if pb_val else "N/A",
                                      help=get_metric_help('pb_ratio', row['sector']))
                            
                            peg_val = row.get('peg_ratio')
                            col3.metric("PEG Ratio", f"{peg_val:.2f}" if peg_val else "N/A",
                                      help=get_metric_help('peg_ratio'))
                            
                            st.markdown("#### Company Size & Returns")
                            col1, col2 = st.columns(2)
                            
                            mcap = row.get('market_cap_cr')
                            if mcap:
                                col1.metric("Market Cap", f"₹{mcap:,.0f} Cr",
                                          help=get_metric_help('market_cap_cr'))
                                if mcap > 100000:
                                    col1.info("🏢 Large Cap (>1L Cr) - More stable, lower volatility")
                                elif mcap > 25000:
                                    col1.info("🏭 Mid Cap (25K-1L Cr) - Balanced risk-reward")
                                else:
                                    col1.info("🏪 Small Cap (<25K Cr) - Higher growth potential, higher risk")
                            else:
                                col1.metric("Market Cap", "N/A")
                            
                            div = row.get('dividend_yield')
                            col2.metric("Dividend Yield", f"{div:.2f}%" if div else "N/A",
                                      help=get_metric_help('dividend_yield', row['sector']))
                        
                        with tab3:
                            st.markdown("#### Financial Health")
                            
                            col1, col2 = st.columns(2)
                            
                            roe_val2 = row.get('roe')
                            col1.metric("Return on Equity (ROE)", f"{roe_val2:.2f}%" if roe_val2 else "N/A",
                                      help=get_metric_help('roe', row['sector']))
                            
                            debt_val = row.get('debt_to_equity')
                            col2.metric("Debt-to-Equity", f"{debt_val:.2f}" if debt_val else "N/A",
                                      help=get_metric_help('debt_to_equity', row['sector']))
                
                st.markdown("---")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis CSV",
                    data=csv,
                    file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Metric interpretations
                st.markdown("### 📌 Metric Interpretations")
                st.markdown("**Valuation:**")
                st.markdown("- **PE Ratio**: Lower PE may indicate undervaluation (compare with sector avg ~20-25 for Nifty)")
                st.markdown("- **PB Ratio**: < 1 means stock trading below book value")
                st.markdown("- **PEG Ratio**: < 1 suggests stock may be undervalued relative to growth")
                
                st.markdown("**Quality:**")
                st.markdown("- **ROE**: > 15% is generally good, > 20% is excellent")
                st.markdown("- **Debt-to-Equity**: < 1 is healthy, > 2 may indicate high risk")
                
                st.markdown("**Price Movement:**")
                st.markdown("- **% from 52W High**: Negative values mean stock is below its peak")
                st.markdown("- **% from 52W Low**: Positive values show recovery from bottom")
                
                st.markdown("**Momentum:**")
                st.markdown("- **1W/1M/3M Changes**: Shows recent price direction")
                st.markdown("- **Trend Label**: Overall momentum classification")
                
                st.markdown("**Risk:**")
                st.markdown("- **Beta**: Less than 1 means less volatile than market, more than 1 means more volatile")
                st.markdown("- **Volatility**: Low, Medium, or High classification based on price fluctuations")
    else:
        st.info("No portfolio loaded. Use Manual Entry or CSV Upload above to get started.")

# ========================================
# END OF APPLICATION
# ========================================

# INSTRUCTIONS FOR RUNNING:
# 1. Save all 4 parts in order in a single .py file
# 2. Ensure you have .env file with GOOGLE_API_KEY
# 3. Install dependencies: pip install streamlit pandas python-dotenv google-generativeai yfinance
# 4. Run: streamlit run your_file_name.py

# ALL FIXES IMPLEMENTED:
# ✅ Issue 1: Manual stock addition to existing portfolio (Part 3)
# ✅ Issue 2: Market regime calculation explanation (Part 1)
# ✅ Issue 3: Clarity on strategic vs tactical AI analysis (Parts 2 & 3)
# ✅ Issue 4: Monthly investment budget input and integration (Parts 1, 2, 3)
# ✅ Issue 5: News fetching from credible sources (Part 1 & 3)