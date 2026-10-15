"""Optional yfinance import, shared by the data tools."""
import warnings

YFINANCE_AVAILABLE = False
yf = None
try:
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    yf = None
    YFINANCE_AVAILABLE = False
