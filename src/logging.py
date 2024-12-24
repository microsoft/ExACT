import logging
import time
import random
import os
import urllib3
import openai
import PIL
import httpcore
import httpx
from pathlib import Path
from src.constants import LOG_FOLDER


logger = logging.getLogger("logger")


def setup_logger(log_folder: str = ''):
    if log_folder == '':
        log_folder = LOG_FOLDER
    
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    LOG_FILE_NAME = os.path.join(
        log_folder,
        f"log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log.txt"
    )

    formatting = "[%(asctime)s] %(levelname)s@%(name)s [%(pathname)s:%(lineno)d] %(message)s"
    if os.environ.get("DEBUG", "") != "":
        print("[[[DEBUG MODE ON for logging]]]")
        logging.basicConfig(
            level=logging.DEBUG,
            force=True,
            format=formatting,
            handlers=[
                logging.FileHandler(LOG_FILE_NAME),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            force=True,
            format=formatting,
            handlers=[
                logging.FileHandler(LOG_FILE_NAME),
                logging.StreamHandler()
            ]
        )
    urllib3.connectionpool.log.setLevel(logging.WARNING)
    httpcore._sync.http11.logger.setLevel(logging.WARNING)
    logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
    httpx._config.logger.setLevel(logging.WARNING)
    openai._base_client.log.setLevel(logging.WARNING)
    PIL.PngImagePlugin.logger.setLevel(logging.WARNING)
    logging.getLogger('PIL.Image').setLevel(logging.WARNING)
    ## used by python -m scalene profiler
    logging.getLogger('markdown_it').setLevel(logging.WARNING)
    ## used by azureml
    logging.getLogger('azure.identity').setLevel(logging.WARNING)
    logging.getLogger('azure.core').setLevel(logging.WARNING)
    return


def time_it(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"Function {func.__name__} from {func.__code__.co_filename} took {time.time() - started_at} seconds")
        return result
    return wrap


def atime_it(afunc):
    async def awrap(*args, **kwargs):
        started_at = time.time()
        result = await afunc(*args, **kwargs)
        logger.debug(f"Function {afunc.__name__} from {afunc.__code__.co_filename} took {time.time() - started_at} seconds")
        return result
    return awrap