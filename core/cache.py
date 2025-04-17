import time
import json
import hashlib
from functools import wraps

_memory_cache = {}

def hash_args(*args, **kwargs):
    try:
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(f"{args_str}|{kwargs_str}".encode()).hexdigest()
    except TypeError:
        return hashlib.md5(f"{repr(args)}|{repr(kwargs)}".encode()).hexdigest()

def cache_result(ttl_seconds=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash_args(*args, **kwargs)}"
            if key in _memory_cache:
                timestamp, result = _memory_cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            result = func(*args, **kwargs)
            _memory_cache[key] = (time.time(), result)
            if len(_memory_cache) > 1000:
                cleanup_memory_cache()
            return result
        return wrapper
    return decorator

def cleanup_memory_cache(max_age_seconds=7200):
    current_time = time.time()
    expired_keys = [
        key for key, (timestamp, _) in _memory_cache.items()
        if current_time - timestamp > max_age_seconds
    ]
    for key in expired_keys:
        _memory_cache.pop(key, None)
