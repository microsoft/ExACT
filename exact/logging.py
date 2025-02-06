import logging
import time
import os
import sys
import datetime
import openai


logger = logging.getLogger("src.logging")


def time_it(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"Function {func.__name__} from {func.__code__.co_filename} took {time.time() - started_at} seconds")
        return result
    return wrap


def setup_logger(log_folder: str):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    file_handler = logging.FileHandler(
        os.path.join(log_folder, "normal-{:}.log.txt".format(datetime_str)), encoding="utf-8"
    )
    debug_handler = logging.FileHandler(
        os.path.join(log_folder, "debug-{:}.log.txt".format(datetime_str)), encoding="utf-8"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    sdebug_handler = logging.FileHandler(
        os.path.join(log_folder, "sdebug-{:}.log.txt".format(datetime_str)), encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    sdebug_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
    )
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    sdebug_handler.setFormatter(formatter)

    filterer = logging.Filterer()
    file_handler.addFilter(logging.Filter("desktopenv"))
    file_handler.addFilter(logging.Filter("src"))
    stdout_handler.addFilter(filterer)
    sdebug_handler.addFilter(filterer)


    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)

    ## shutoff some loggers that somehow passes the filter
    logging.getLogger('PIL').setLevel(logging.WARNING)
    openai._base_client.log.setLevel(logging.WARNING)
    logging.getLogger('azure.identity').setLevel(logging.WARNING)
    logging.getLogger('azure.core').setLevel(logging.WARNING)
    return


def atime_it(afunc):
    async def awrap(*args, **kwargs):
        started_at = time.time()
        result = await afunc(*args, **kwargs)
        logger.debug(f"Function {afunc.__name__} from {afunc.__code__.co_filename} took {time.time() - started_at} seconds")
        return result
    return awrap