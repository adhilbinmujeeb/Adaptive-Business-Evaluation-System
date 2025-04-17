from utils.data_processing import safe_float
from core.cache import cache_result

@cache_result(ttl_seconds=86400)
def get_industry_pe_multiple(db_collection, industry):
    default_multiples = {"Software/SaaS": 25.0, "E-commerce": 18.0, "Retail": 12.0}
    return default_multiples.get(industry, 15.0)

@cache_result(ttl_seconds=86400)
def get_industry_ebitda_multiple(db_collection, industry):
    default_multiples = {"Software/SaaS": 15.0, "E-commerce": 10.0, "Manufacturing": 7.0}
    return default_multiples.get(industry, 8.0)

def calculate_dcf(cash_flows, discount_rate=0.12, terminal_growth=0.02):
    pv_cash_flows = sum(cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
    last_cf = cash_flows[-1] if cash_flows else 0
    if discount_rate > terminal_growth:
        terminal_value = (last_cf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** len(cash_flows))
    else:
        pv_terminal_value = 0
    return pv_cash_flows + pv_terminal_value

def calculate_valuation(company_data):
    results = {}
    valuation_methods_used = []
    
    assets = safe_float(company_data.get('assets'))
    liabilities = safe_float(company_data.get('liabilities'))
    earnings = safe_float(company_data.get('earnings'))
    ebitda = safe_float(company_data.get('ebitda'))
    revenue = safe_float(company_data.get('revenue'))
    cash_flows = [safe_float(cf) for cf in company_data.get('cash_flows_str', "0,0,0,0,0").split(",")]
    industry = company_data.get('industry', 'Other')
    growth = company_data.get('growth', 'Moderate')

    if assets is not None and liabilities is not None and (bv := assets - liabilities) > 0:
        results['book_value'] = bv
        valuation_methods_used.append("Book Value (Assets - Liabilities)")

    if earnings > 0:
        pe_multiple = get_industry_pe_multiple(None, industry)
        results['pe_valuation'] = earnings * pe_multiple
        valuation_methods_used.append(f"Price/Earnings (P/E) using x{pe_multiple:.1f} multiple")

    if ebitda > 0:
        ebitda_multiple = get_industry_ebitda_multiple(None, industry)
        results['ebitda_valuation'] = ebitda * ebitda_multiple
        valuation_methods_used.append(f"EV/EBITDA using x{ebitda_multiple:.1f} multiple")

    if cash_flows and any(cf != 0 for cf in cash_flows):
        dcf_val = calculate_dcf(cash_flows)
        if dcf_val > 0:
            results['dcf_valuation'] = dcf_val
            valuation_methods_used.append("Discounted Cash Flow (DCF)")

    if revenue > 0 and not (results.get('pe_valuation') or results.get('ebitda_valuation')):
        revenue_multiples = {
            'High': {'Software/SaaS': 8.0, 'E-commerce': 3.0, 'Other': 2.5},
            'Moderate': {'Software/SaaS': 5.0, 'E-commerce': 1.8, 'Other': 1.5},
            'Low': {'Software/SaaS': 2.5, 'E-commerce': 0.8, 'Other': 0.7}
        }
        multiple = revenue_multiples.get(growth, revenue_multiples['Moderate']).get(industry, 1.5)
        results['revenue_valuation'] = revenue * multiple
        valuation_methods_used.append(f"Revenue Multiple using x{multiple:.1f}")

    valid_valuations = [v for v in results.values() if v > 0]
    results['average_valuation'] = sum(valid_valuations) / len(valid_valuations) if valid_valuations else 0
    results['valuation_range'] = (min(valid_valuations), max(valid_valuations)) if valid_valuations else (0, 0)
    results['_methods_used'] = valuation_methods_used

    return results

def calculate_valuation_confidence(company_data, valuation_results):
    confidence_scores = {}
    overall_completeness = 0
    total_possible_score = 0

    required_fields = {
        'pe_valuation': ['earnings'],
        'ebitda_valuation': ['ebitda'],
        'dcf_valuation': ['cash_flows_str'],
        'book_value': ['assets', 'liabilities'],
        'revenue_valuation': ['revenue']
    }
    data_quality_factors = {'earnings': 1.0, 'ebitda': 1.0, 'revenue': 1.0, 'assets': 0.9, 'liabilities': 0.9, 'cash_flows_str': 0.7}
    
    for method in valuation_results:
        if method.startswith('_') or method in ['average_valuation', 'valuation_range']:
            continue
        if method not in required_fields:
            continue
        method_completeness = 1.0
        quality_score = 1.0
        total_possible_score += 1

        for field in required_fields[method]:
            field_value = company_data.get(field)
            is_present_and_valid = (
                field_value is not None and (
                    (field in ['earnings', 'ebitda', 'revenue'] and safe_float(field_value) > 0) or
                    (field in ['assets', 'liabilities'] and safe_float(field_value) >= 0) or
                    (field == 'cash_flows_str' and any(safe_float(cf) != 0 for cf in field_value.split(',')))
                )
            )
            if not is_present_and_valid:
                method_completeness *= 0.5
            else:
                quality_score *= data_quality_factors.get(field, 0.8)

        confidence = 0.6 * method_completeness * quality_score
        confidence_scores[method] = max(0.1, min(1.0, confidence))
        if method_completeness > 0.75:
            overall_completeness += 1

    avg_method_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
    completeness_ratio = overall_completeness / total_possible_score if total_possible_score else 0
    confidence_scores['_overall_confidence'] = max(0.1, min(1.0, (avg_method_confidence * 0.6 + completeness_ratio * 0.4)))
    confidence_scores['_overall_completeness'] = completeness_ratio

    return confidence_scores
