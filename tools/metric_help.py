"""Plain-language explanations of metrics shown in the UI."""

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
