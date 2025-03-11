import os
import signal
import concurrent.futures
import errno
import functools
import time
import logging
import asyncio
import sys


logger = logging.getLogger("src.env.utils")


class TimeoutError(Exception):
    pass

### signal based timeout is incompatible with multithreading.
### e.g., if def func_a() has a timeout wrapper, then func_a cannot be called from a thread/threadpool
# def timeout(seconds=2, error_message=os.strerror(errno.ETIME)):
#     def decorator(func):
#         def _handle_timeout(signum, frame):
#             raise TimeoutError(error_message)

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             start_time = time.time()
#             signal.signal(signal.SIGALRM, _handle_timeout)
#             signal.alarm(seconds)
#             try:
#                 result = func(*args, **kwargs)
#             except TimeoutError as e:
#                 logger.error(f"func {func.__name__} timed out after {time.time() - start_time} seconds")
#                 raise e
#             finally:
#                 signal.alarm(0)
#             file_path = sys.modules[func.__module__].__file__
#             logger.debug(f"Function {func.__name__} from {file_path} took {time.time() - start_time} seconds")
#             return result

#         return wrapper

#     return decorator


def timeout(seconds=2, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            futures = [future]
            
            done, _ = concurrent.futures.wait(
                futures,
                timeout=seconds,
            )
            executor.shutdown(wait=False)
            
            if len(done) == 0:
                logger.error(f"func {func.__name__} timed out after {seconds} seconds")
                raise TimeoutError(error_message)
            return future.result()
        return wrapper
    return decorator


def atimeout(seconds=2, error_message=os.strerror(errno.ETIME)):
    def decorator(afunc):
        async def awrap(*args, **kwargs):
            try:
                start_time = time.time()
                result = await asyncio.wait_for(afunc(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError as e:
                logger.error(f"func {afunc.__name__} timed out after {time.time() - start_time} seconds")
                raise TimeoutError(error_message)
            file_path = sys.modules[afunc.__module__].__file__
            logger.debug(f"Function {afunc.__name__} from {file_path} took {time.time() - start_time} seconds")
            return result
        return awrap

    return decorator


def retry_timeout(num_retry: int = 2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num_retry):
                try:
                    return func(*args, **kwargs)
                except TimeoutError as e:
                    logger.error(f"Retry {i + 1}/{num_retry} failed")
                    continue
            raise TimeoutError(f"Retry {num_retry} failed within time limit")
        return wrapper
    return decorator


def aretry_timeout(num_retry: int = 2):
    def decorator(afunc):
        async def awrap(*args, **kwargs):
            for i in range(num_retry):
                try:
                    return await afunc(*args, **kwargs)
                except TimeoutError as e:
                    logger.error(f"Retry {i + 1}/{num_retry} failed")
                    continue
            raise TimeoutError(f"Retry {num_retry} failed within time limit")

        return awrap
    return decorator