"""Sector policy/narrative context."""

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
