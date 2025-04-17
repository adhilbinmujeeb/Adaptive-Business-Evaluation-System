def validate_business_profile(extracted_profile, expected_schema):
    return isinstance(extracted_profile, dict) and all(key in extracted_profile for key in expected_schema)
