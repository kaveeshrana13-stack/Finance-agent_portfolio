"""Index returns and market/sector regime classification."""
from tools._yf import yf, YFINANCE_AVAILABLE

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
