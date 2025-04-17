from functools import wraps
from core.cache import cache_result
from core.database import get_database
import re

def parse_revenue_bracket(revenue_str):
    """Parse revenue bracket string to a float (midpoint)."""
    if not revenue_str or revenue_str == "not_provided":
        return None
    match = re.match(r"\$(\d+,\d{3})\s*-\s*\$(\d+,\d{3})", revenue_str)
    if match:
        low = float(match.group(1).replace(",", ""))
        high = float(match.group(2).replace(",", ""))
        return (low + high) / 2
    return None

@cache_result(ttl_seconds=3600)
def get_industry_pe_multiple(collection, industry):
    """Fetch industry P/E multiple (placeholder)."""
    return 15.0  # Replace with MongoDB query if data available

@cache_result(ttl_seconds=3600)
def get_industry_ebitda_multiple(collection, industry):
    """Fetch industry EBITDA multiple (placeholder)."""
    return 8.0  # Replace with MongoDB query if data available

def calculate_valuation(company_data):
    """Calculate valuation using multiple methods based on available data."""
    db = get_database()
    results = {}

    # Try to fetch revenue from business_attributes if not provided
    if not company_data.get('revenue') and company_data.get('business_id'):
        attr = db['business_attributes'].find_one({"business_id": company_data['business_id']})
        if attr and attr.get("Business Attributes.Financial Metrics.Revenue Brackets (Annual)"):
            company_data['revenue'] = parse_revenue_bracket(
                attr["Business Attributes.Financial Metrics.Revenue Brackets (Annual)"]
            )

    # Book Value (Asset-Based)
    if company_data.get('assets') and company_data.get('liabilities'):
        results['book_value'] = company_data['assets'] - company_data['liabilities']

    # P/E Method
    if company_data.get('earnings') and company_data.get('earnings') > 0:
        industry = company_data.get('industry', 'Other')
        pe_multiple = get_industry_pe_multiple(db['business_listings'], industry)
        results['pe_valuation'] = company_data['earnings'] * pe_multiple

    # EV/EBITDA Method
    if company_data.get('ebitda') and company_data.get('ebitda') > 0:
        industry = company_data.get('industry', 'Other')
        ebitda_multiple = get_industry_ebitda_multiple(db['business_listings'], industry)
        results['ebitda_valuation'] = company_data['ebitda'] * ebitda_multiple

    # Revenue-based method for early-stage
    if company_data.get('revenue') and company_data.get('revenue') > 0:
        industry = company_data.get('industry', 'Other')
        growth = company_data.get('growth', 'Moderate')
        revenue_multiples = {
            'High': {'Beauty & Personal Care': 3.5, 'Tools': 2.5, 'Software/SaaS': 5.0, 'E-commerce': 3.0, 'Other': 3.0},
            'Moderate': {'Beauty & Personal Care': 2.0, 'Tools': 1.5, 'Software/SaaS': 3.0, 'E-commerce': 2.0, 'Other': 1.5},
            'Low': {'Beauty & Personal Care': 1.0, 'Tools': 0.8, 'Software/SaaS': 1.5, 'E-commerce': 1.0, 'Other': 0.8}
        }
        multiple = revenue_multiples[growth].get(industry, revenue_multiples[growth]['Other'])
        results['revenue_valuation'] = company_data['revenue'] * multiple

    return results

def calculate_valuation_confidence(company_data, valuation_results):
    """Calculate confidence scores for valuation estimates."""
    confidence_scores = {}
    required_fields = {
        'pe_valuation': ['earnings'],
        'ebitda_valuation': ['ebitda'],
        'book_value': ['assets', 'liabilities'],
        'revenue_valuation': ['revenue']
    }
    data_quality = {
        'earnings': 1.0 if company_data.get('earnings', 0) > 0 else 0.5,
        'ebitda': 1.0 if company_data.get('ebitda', 0) > 0 else 0.5,
        'revenue': 1.0,
        'assets': 0.9,
        'liabilities': 0.9
    }

    for method, value in valuation_results.items():
        if method not in required_fields:
            continue
        base_confidence = 0.7
        quality_score = 1.0
        for field in required_fields[method]:
            if field not in company_data or company_data[field] is None or company_data[field] == 0:
                quality_score *= 0.5
            else:
                quality_score *= data_quality.get(field, 0.8)
        confidence_scores[method] = base_confidence * quality_score

    return confidence_scores
