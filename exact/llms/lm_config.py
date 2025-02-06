"""Config for language models."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer. This will be used to do turn cutting for long prompts.
        mode: The mode of the API calls, e.g., "chat" or "generation".
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    mode: str | None = None
    ## other LLM provider related configs
    api_base: str = "https://api.openai.com/v1"
    api_organization: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    api_token_provider_base: str | None = None
    ## gen configs
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)


def construct_llm_config(args: argparse.Namespace) -> LMConfig:
    llm_config = LMConfig(
        provider=args.provider, model=args.model, mode=args.mode
    )
    if args.provider in ["openai", "google", "sglang", "azure"]:
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["context_length"] = args.context_length
        llm_config.gen_config["max_tokens"] = args.max_tokens
        llm_config.gen_config["stop_token"] = args.stop_token
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
        llm_config.gen_config["max_retry"] = args.max_retry
    elif args.provider == "huggingface":
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["max_new_tokens"] = args.max_tokens
        llm_config.gen_config["stop_sequences"] = (
            [args.stop_token] if args.stop_token else None
        )
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
        llm_config.gen_config["model_endpoint"] = args.model_endpoint
        llm_config.gen_config["max_retry"] = args.max_retry
    else:
        raise NotImplementedError(f"provider {args.provider} not implemented")
    return llm_config


def construct_rlm_config(args: argparse.Namespace) -> LMConfig:
    llm_config = LMConfig(
        provider=args.rlm_provider, model=args.rlm_model, mode=args.rlm_mode
    )
    if args.rlm_provider in ["openai", "google", "sglang", "azure"]:
        llm_config.gen_config["temperature"] = args.rlm_temperature
        llm_config.gen_config["top_p"] = args.rlm_top_p
        llm_config.gen_config["context_length"] = args.rlm_context_length
        llm_config.gen_config["max_tokens"] = args.rlm_max_tokens
        llm_config.gen_config["stop_token"] = args.rlm_stop_token
        llm_config.gen_config["max_obs_length"] = args.rlm_max_obs_length
        llm_config.gen_config["max_retry"] = args.rlm_max_retry
    elif args.rlm_provider == "huggingface":
        llm_config.gen_config["temperature"] = args.rlm_temperature
        llm_config.gen_config["top_p"] = args.rlm_top_p
        llm_config.gen_config["max_new_tokens"] = args.rlm_max_tokens
        llm_config.gen_config["stop_sequences"] = (
            [args.rlm_stop_token] if args.rlm_stop_token else None
        )
        llm_config.gen_config["max_obs_length"] = args.rlm_max_obs_length
        llm_config.gen_config["model_endpoint"] = args.rlm_model_endpoint
        llm_config.gen_config["max_retry"] = args.rlm_max_retry
    else:
        raise NotImplementedError(f"provider {args.rlm_provider} not implemented")
    return llm_config
