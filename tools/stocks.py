"""Per-stock fundamentals, momentum and risk metrics from yfinance."""
from tools._yf import yf, YFINANCE_AVAILABLE
from tools.tickers import clean_ticker_symbol

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
