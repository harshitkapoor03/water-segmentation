import logging
import time
import functools

logger = logging.getLogger(__name__)


def log_inference_time(func):
    """
    A decorator that wraps a function and logs how long it took.

    A decorator is a function that takes another function as input
    and returns a modified version of it. The @log_inference_time
    syntax above a function definition is shorthand for:
        predict_single = log_inference_time(predict_single)

    functools.wraps(func) copies the original function's name and
    docstring onto the wrapper — without it, all decorated functions
    would appear as 'wrapper' in logs and stack traces.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # *args catches any positional arguments
        # **kwargs catches any keyword arguments
        # This makes the decorator work on any function signature
        start   = time.time()
        result  = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper
