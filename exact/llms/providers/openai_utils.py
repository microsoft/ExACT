import logging
import os
import random
import time
import requests
from typing import Any, Union

import openai
from openai import OpenAI, AzureOpenAI
from exact.constants import TOKEN_USAGE, SIMPLE_LLM_API_CACHE


logger = logging.getLogger("src.llms")


def update_token_usage(model_name: str, token_stats: dict):
    global TOKEN_USAGE
    # expect token_stats to include completion_tokens, prompt_tokens, num_requests
    if model_name not in TOKEN_USAGE:
        TOKEN_USAGE[model_name] = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'num_requests': 0
        }

    prev_num_req = TOKEN_USAGE[model_name]['num_requests']
    TOKEN_USAGE[model_name]['completion_tokens'] += token_stats['completion_tokens']
    TOKEN_USAGE[model_name]['prompt_tokens'] += token_stats['prompt_tokens']
    TOKEN_USAGE[model_name]['num_requests'] += token_stats['num_requests']

    compl_token = TOKEN_USAGE[model_name]['completion_tokens']
    prompt_token = TOKEN_USAGE[model_name]['prompt_tokens']
    total_num_req = TOKEN_USAGE[model_name]['num_requests']
    # num request may increment by alot at once
    for i in range(token_stats['num_requests']):
        new_num_req = prev_num_req + i + 1
        if new_num_req % 10 == 0:
            logger.info(f"[{model_name}] Avg. completion tokens: {compl_token/total_num_req:.2f}, prompt tokens: {prompt_token/total_num_req:.2f}")
            logger.info(f"[{model_name}] Total. completion tokens so far: {compl_token}, prompt tokens so far: {prompt_token}")
            logger.info(f"[{model_name}] Total. requests so far: {total_num_req}")
    return


def reset_token_usage():
    global TOKEN_USAGE
    TOKEN_USAGE = {}
    return


def get_all_token_usage():
    global TOKEN_USAGE
    # returns a copy just in case
    all_token_usage = {}
    for m_name, m_stats in TOKEN_USAGE.items():
        all_token_usage[m_name] = {
            'completion_tokens': m_stats['completion_tokens'],
            'prompt_tokens': m_stats['prompt_tokens'],
            'num_requests': m_stats['num_requests']
        }
    return all_token_usage


def set_all_token_usage(all_token_usage: dict):
    global TOKEN_USAGE
    TOKEN_USAGE = all_token_usage
    return


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                logger.error(e, exc_info=True)
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    logger.error(f"Maximum number of retries ({max_retries}) exceeded.")
                    num_outputs = kwargs.get("num_outputs", 1)
                    if num_outputs > 1:
                        return ["ERROR"] * num_outputs
                    else:
                        return "ERROR"

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def _completion_args_to_cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    num_outputs: int = 1,
) -> str:
    return f"{model}_{messages}_{temperature}_{max_tokens}_{top_p}_{num_outputs}"


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int = -1,
    stop_token: str | None = None,
    num_outputs: int = 1,
) -> Union[str, list[str]]:
    cache_key = _completion_args_to_cache_key(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        num_outputs=num_outputs
    )
    if temperature == 0.0 and cache_key in SIMPLE_LLM_API_CACHE:
        logger.info(f"generate_from_openai_chat_completion hit cache")
        return SIMPLE_LLM_API_CACHE[cache_key]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=num_outputs
    )

    token_stats = {
        'completion_tokens': response.usage.completion_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'num_requests': 1
    }
    update_token_usage(
        model_name=model,
        token_stats=token_stats
    )

    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    if temperature == 0.0:
        SIMPLE_LLM_API_CACHE[cache_key] = answer
    return answer


@retry_with_exponential_backoff
def generate_from_ray_completion(
    api_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
):
    cache_key = _completion_args_to_cache_key(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=-1,
        top_p=-1,
        num_outputs=1
    )
    if temperature == 0.0 and cache_key in SIMPLE_LLM_API_CACHE:
        logger.info(f"generate_from_ray_completion hit cache")
        return SIMPLE_LLM_API_CACHE[cache_key]
    
    ### send request
    prompt = {
        "model": model_name,
        "messages": messages
    }
    logger.debug(f'generate_from_ray_completion sending request to {api_url}')
    pooling_response = requests.post(
        api_url,
        json=prompt,
        timeout=120
    )
    response = pooling_response.json()
    
    ### post-process
    token_stats = {
        'completion_tokens': response['usage']['completion_tokens'],
        'prompt_tokens': response['usage']['prompt_tokens'],
        'num_requests': 1
    }
    update_token_usage(
        model_name=model_name,
        token_stats=token_stats
    )
    
    if temperature == 0.0:
        SIMPLE_LLM_API_CACHE[cache_key] = response
    return response


@retry_with_exponential_backoff
def generate_from_azure_openai_chat_completion(
    client: AzureOpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int = -1,
    stop_token: str | None = None,
    num_outputs: int = 1,
) -> Union[str, list[str]]:
    cache_key = _completion_args_to_cache_key(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        num_outputs=num_outputs
    )
    if temperature == 0.0 and cache_key in SIMPLE_LLM_API_CACHE:
        logger.info(f"generate_from_azure_openai_chat_completion hit cache")
        return SIMPLE_LLM_API_CACHE[cache_key]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=num_outputs
    )

    token_stats = {
        'completion_tokens': response.usage.completion_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'num_requests': 1
    }
    update_token_usage(
        model_name=f"azure_{model}",
        token_stats=token_stats,
    )

    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    if temperature == 0.0:
        SIMPLE_LLM_API_CACHE[cache_key] = answer
    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int = -1,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer