from cachetools import Cache

SIMPLE_LLM_API_CACHE = Cache(maxsize=500)

TOKEN_USAGE = {}

OPENAI_API_BASE = "https://api.openai.com/v1"
VWA_LOG_FOLDER = "data/visualwebarena/log_files"
OSWORLD_LOG_FOLDER = "data/osworld_data/log_files"