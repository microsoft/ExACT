from cachetools import Cache

SIMPLE_LLM_API_CACHE = Cache(maxsize=100)

TOKEN_USAGE = {}

OPENAI_API_BASE = "https://api.openai.com/v1"
LOG_FOLDER = "data/visualwebarena/log_files"