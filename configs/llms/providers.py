import yaml

with open('configs/llms/providers.yaml') as fread:
    _API_PROVIDERS = yaml.load(fread, Loader=yaml.FullLoader)

AVAILABLE_API_PROVIDERS = {
    "azure": f"""
export PROVIDER="{_API_PROVIDERS['azure']['provider']}"
export AGENT_LLM_API_BASE="{_API_PROVIDERS['azure']['llm_api_base']}"
export AGENT_LLM_API_KEY="$(echo $OPENAI_API_KEY)"
export VALUE_FUNC_PROVIDER="{_API_PROVIDERS['azure']['provider']}"
export VALUE_FUNC_API_BASE="{_API_PROVIDERS['azure']['llm_api_base']}"
export RLM_PROVIDER="{_API_PROVIDERS['azure']['provider']}"  # not used as it will be overwritten by $PROVIDER
export EMBEDDING_MODEL_PROVIDER="openai"
export AZURE_TOKEN_PROVIDER_BASE="$(echo $AZURE_TOKEN_PROVIDER_API_BASE)"
export AZURE_OPENAI_API_VERSION="{_API_PROVIDERS['azure']['llm_api_version']}"
""".strip(),
    "openai": f"""
export PROVIDER="{_API_PROVIDERS['openai']['provider']}"
export AGENT_LLM_API_BASE="{_API_PROVIDERS['openai']['llm_api_base']}"
export AGENT_LLM_API_KEY="$(echo $OPENAI_API_KEY)"
export VALUE_FUNC_PROVIDER="{_API_PROVIDERS['openai']['provider']}"
export VALUE_FUNC_API_BASE="{_API_PROVIDERS['openai']['llm_api_base']}"
export RLM_PROVIDER="{_API_PROVIDERS['openai']['provider']}"  # not used as it will become PROVIDER
export EMBEDDING_MODEL_PROVIDER="openai"
export AZURE_TOKEN_PROVIDER_BASE=""
export AZURE_OPENAI_API_VERSION=""
""".strip(),
    "sglang": f"""
export PROVIDER="{_API_PROVIDERS['sglang']['provider']}"
export AGENT_LLM_API_BASE="{_API_PROVIDERS['sglang']['llm_api_base']}"
export AGENT_LLM_API_KEY="EMPTY"
export VALUE_FUNC_PROVIDER="{_API_PROVIDERS['sglang']['provider']}"
export VALUE_FUNC_API_BASE="{_API_PROVIDERS['sglang']['llm_api_base']}"
export RLM_PROVIDER="{_API_PROVIDERS['sglang']['provider']}"  # not used as it will become PROVIDER
export EMBEDDING_MODEL_PROVIDER="openai"
export AZURE_TOKEN_PROVIDER_BASE=""
export AZURE_OPENAI_API_VERSION=""
""".strip()
}