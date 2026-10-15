import re

# Ticker helpers
def clean_ticker_symbol(symbol):
    """Clean and normalize ticker symbol"""
    if not symbol:
        return None
    symbol = str(symbol).strip().upper()
    symbol = symbol.replace('.NS', '').replace('.BO', '')
    symbol = re.sub(r'[^A-Z0-9&]', '', symbol)
    return symbol if symbol else None
