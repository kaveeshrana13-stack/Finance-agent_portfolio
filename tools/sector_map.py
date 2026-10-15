"""Sector lookup: Yahoo Finance sector/industry mapped onto this app's sector names.

The app's sector vocabulary (used by SECTOR_BENCHMARKS in tools/market.py and POLICY_NARRATIVES) is:
IT, Banking, Pharma, FMCG, Auto, Energy, Metal, Infrastructure, Financial Services,
Cement, Aviation, Defence, Chemicals. Anything Yahoo reports outside that vocabulary is
returned under Yahoo's own sector name with mapped=False, so no information is lost.
"""
import logging
from functools import lru_cache

from tools._yf import yf, YFINANCE_AVAILABLE

# Industry is the more specific field, so it wins over sector.
INDUSTRY_TO_SECTOR = {
    "Information Technology Services": "IT",
    "Software - Application": "IT",
    "Software - Infrastructure": "IT",
    "Computer Hardware": "IT",
    "Banks - Regional": "Banking",
    "Banks - Diversified": "Banking",
    "Drug Manufacturers - Specialty & Generic": "Pharma",
    "Drug Manufacturers - General": "Pharma",
    "Biotechnology": "Pharma",
    "Auto Manufacturers": "Auto",
    "Auto Parts": "Auto",
    "Auto & Truck Dealerships": "Auto",
    "Steel": "Metal",
    "Aluminum": "Metal",
    "Copper": "Metal",
    "Other Industrial Metals & Mining": "Metal",
    "Gold": "Metal",
    "Other Precious Metals & Mining": "Metal",
    "Building Materials": "Cement",  # Yahoo's bucket for cement; also holds tiles etc.
    "Specialty Chemicals": "Chemicals",
    "Chemicals": "Chemicals",
    "Agricultural Inputs": "Chemicals",
    "Engineering & Construction": "Infrastructure",
    "Infrastructure Operations": "Infrastructure",
    "Airlines": "Aviation",
    "Airports & Air Services": "Aviation",
    "Aerospace & Defense": "Defence",
}

# Fallback when the industry isn't listed above.
SECTOR_TO_SECTOR = {
    "Technology": "IT",
    "Financial Services": "Financial Services",
    "Consumer Defensive": "FMCG",
    "Energy": "Energy",
}


def map_yahoo_sector(yahoo_sector, yahoo_industry):
    """Pure mapping from Yahoo's (sector, industry) to this app's sector name."""
    if yahoo_industry in INDUSTRY_TO_SECTOR:
        return {"sector": INDUSTRY_TO_SECTOR[yahoo_industry], "mapped": True}
    if yahoo_sector in SECTOR_TO_SECTOR:
        return {"sector": SECTOR_TO_SECTOR[yahoo_sector], "mapped": True}
    if yahoo_sector:
        return {"sector": yahoo_sector, "mapped": False}
    return {"sector": "Unknown", "mapped": False}


@lru_cache(maxsize=512)
def lookup_sector(symbol):
    """Fetch sector/industry for an NSE (or BSE) symbol from Yahoo and map it.

    Returns {"sector", "mapped", "yahoo_sector", "yahoo_industry"}.
    """
    empty = {"sector": "Unknown", "mapped": False, "yahoo_sector": None, "yahoo_industry": None}
    if not symbol or not YFINANCE_AVAILABLE or yf is None:
        return empty
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    for exchange in (".NS", ".BO"):
        try:
            info = yf.Ticker(f"{symbol}{exchange}").info or {}
        except Exception:
            continue
        ys, yi = info.get("sector"), info.get("industry")
        if ys or yi:
            return {**map_yahoo_sector(ys, yi), "yahoo_sector": ys, "yahoo_industry": yi}
    return empty
