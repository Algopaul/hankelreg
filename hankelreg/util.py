import functools
import logging
import time

import humanize


def log_duration(label: str = None, *, minimum_unit: str = "microseconds"):
  """
    Decorator that logs how long a function takes.
    
    Args:
      label: Optional name to show in the log; defaults to func.__name__.
      minimum_unit: humanize.naturaldelta's minimum_unit.
    """

  def decorator(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      start = time.time()
      try:
        return func(*args, **kwargs)
      finally:
        name = label or func.__name__
        elapsed = humanize.naturaldelta(
            time.time() - start, minimum_unit=minimum_unit)
        logging.info("%s complete. Took %s", name, elapsed)

    return wrapper

  return decorator
