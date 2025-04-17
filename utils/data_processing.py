def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else default
        str_value = str(value).replace("$", "").replace(",", "").strip()
        return float(str_value) if str_value else default
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    if value is None:
        return default
    try:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else default
        str_value = str(value).replace("$", "").replace(",", "").strip()
        return int(float(str_value)) if str_value else default
    except (ValueError, TypeError):
        return default
