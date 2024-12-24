from typing import Any
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai
import os

try:
    from vertexai.preview.generative_models import Image  # type: ignore
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')


from src.llms import lm_config
from src.llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_azure_openai_chat_completion,
    generate_from_openai_requestapi_chat_completion,
    generate_from_openai_completion,
)
from PIL import Image as PILImage
from src.logging import time_it


APIInput = str | list[Any] | dict[str, Any]


def is_vlm(lm_config: lm_config.LMConfig):
    if ("gemini" in lm_config.model 
        or ("gpt-4" in lm_config.model and "vision" in lm_config.model)
        or "gpt-4o" in lm_config.model):
        return True
    if "llava" in lm_config.model or "mantis" in lm_config.model or "InternVL2" in lm_config.model:
        return True
    return False


def _add_modality_key_for_sglang_messages(messages: list):
    for turn in messages:
        turn_content = turn["content"]
        if isinstance(turn_content, list):
            for parts in turn_content:
                if parts['type'] == "image_url":
                    parts["modalities"] = "multi-images"  # sglang needs this
        # noop if its just string
    return messages


@time_it
def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    num_outputs: int = 1,
) -> str:
    """Exclusively used by agent LLM"""
    response: str
    if lm_config.provider in ["openai", "sglang", "azure"]:
        api_base = os.getenv("AGENT_LLM_API_BASE", "")
        if lm_config.provider == "azure":
            token_provider_base = os.getenv("AZURE_TOKEN_PROVIDER_BASE", "")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

            azure_credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(azure_credential, token_provider_base)
            client = openai.AzureOpenAI(
                api_version=api_version,
                azure_endpoint=api_base,
                azure_ad_token_provider=token_provider
            )
        else:
            client = openai.OpenAI(
                api_key=os.getenv("AGENT_LLM_API_KEY", ""),
                base_url=api_base
            )

        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            # special case with o1 model
            # otherwise, its openai v.s. azure client
            if "o1" in lm_config.model:
                response = generate_from_openai_requestapi_chat_completion(
                    client,
                    messages=prompt,
                    model=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    context_length=lm_config.gen_config["context_length"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    stop_token=None,
                    num_outputs=num_outputs,
                )
            elif lm_config.provider in ["openai", "sglang"]:
                response = generate_from_openai_chat_completion(
                    client,
                    messages=prompt,
                    model=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    context_length=lm_config.gen_config["context_length"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    stop_token=None,
                    num_outputs=num_outputs,
                )
            else:
                response = generate_from_azure_openai_chat_completion(
                    client,
                    messages=prompt,
                    model=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    context_length=lm_config.gen_config["context_length"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    stop_token=None,
                    num_outputs=num_outputs,
                )
        elif lm_config.mode == "completion":
            # assert isinstance(prompt, str)
            # response = generate_from_openai_completion(
            #     prompt=prompt,
            #     engine=lm_config.model,
            #     temperature=lm_config.gen_config["temperature"],
            #     max_tokens=lm_config.gen_config["max_tokens"],
            #     top_p=lm_config.gen_config["top_p"],
            #     stop_token=lm_config.gen_config["stop_token"],
            # )
            raise NotImplementedError(
                f"Please use lm_config.mode='chat' for OpenAI models"
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) or isinstance(p, PILImage.Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
            # n=1  # Gemini only supports 1 output for now
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response