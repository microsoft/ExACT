"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""
import requests
import asyncio
import logging
import os
import random
import time
from typing import Any, Union

import aiolimiter
import openai
from openai import AsyncOpenAI, OpenAI, AzureOpenAI
from tqdm.asyncio import tqdm_asyncio
from src.constants import TOKEN_USAGE, SIMPLE_LLM_API_CACHE


logger = logging.getLogger('logger')


DISABLE_LLM_CACHE = os.getenv("DISABLE_LLM_CACHE", False)
logger.info(f"DISABLE_LLM_CACHE: {DISABLE_LLM_CACHE}")


def update_token_usage(model_name: str, token_stats: dict, token_usage_tracker: dict):
    # expect token_stats to include completion_tokens, prompt_tokens, num_requests
    if model_name not in token_usage_tracker:
        token_usage_tracker[model_name] = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'num_requests': 0
        }

    token_usage_tracker[model_name]['completion_tokens'] += token_stats['completion_tokens']
    token_usage_tracker[model_name]['prompt_tokens'] += token_stats['prompt_tokens']
    token_usage_tracker[model_name]['num_requests'] += token_stats['num_requests']

    compl_token = token_usage_tracker[model_name]['completion_tokens']
    prompt_token = token_usage_tracker[model_name]['prompt_tokens']
    req = token_usage_tracker[model_name]['num_requests']  # may increment by alot
    if req % 10 == 0 or token_stats['num_requests'] > 5:
        logger.info(f"[{model_name}] Avg. completion tokens: {compl_token/req:.2f}, prompt tokens: {prompt_token/req:.2f}")
        logger.info(f"[{model_name}] Total. completion tokens so far: {compl_token}, prompt tokens so far: {prompt_token}")
        logger.info(f"[{model_name}] Total. requests so far: {req}")
    return


def get_all_token_usage(token_usage_tracker: dict):
    all_token_usage = {}
    for m_name, m_stats in token_usage_tracker.items():
        all_token_usage[m_name] = {
            'completion_tokens': m_stats['completion_tokens'],
            'prompt_tokens': m_stats['prompt_tokens'],
            'num_requests': m_stats['num_requests']
        }
    return all_token_usage


def _compute_token_usage_diff(prev_all_token_usage: dict, curr_all_token_usage: dict):
    token_usage_diff = {}
    for m_name, m_stats in curr_all_token_usage.items():
        if m_name not in prev_all_token_usage:
            # usually not the case
            token_usage_diff[m_name] = {
                'completion_tokens': m_stats['completion_tokens'],
                'prompt_tokens': m_stats['prompt_tokens'],
                'num_requests': m_stats['num_requests']
            }
        else:
            token_usage_diff[m_name] = {
                'completion_tokens': m_stats['completion_tokens'] - prev_all_token_usage[m_name]['completion_tokens'],
                'prompt_tokens': m_stats['prompt_tokens'] - prev_all_token_usage[m_name]['prompt_tokens'],
                'num_requests': m_stats['num_requests'] - prev_all_token_usage[m_name]['num_requests']
            }
    return token_usage_diff


class ManualRateLimitError(Exception):
    pass


INITIAL_DELAY = 1
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
        ManualRateLimitError
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = INITIAL_DELAY

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


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.completions.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    raise NotImplementedError("Please use generate_from_openai_chat_completion instead.")


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]


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
    context_length: int,
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
    if not DISABLE_LLM_CACHE and cache_key in SIMPLE_LLM_API_CACHE:
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
        token_stats=token_stats,
        token_usage_tracker=TOKEN_USAGE
    )

    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    SIMPLE_LLM_API_CACHE[cache_key] = answer
    return answer


@retry_with_exponential_backoff
def generate_from_azure_openai_chat_completion(
    client: AzureOpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
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
    if not DISABLE_LLM_CACHE and cache_key in SIMPLE_LLM_API_CACHE:
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
        token_usage_tracker=TOKEN_USAGE
    )

    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    SIMPLE_LLM_API_CACHE[cache_key] = answer
    return answer


def _reformat_o1_messages(messages):
    # o1 does not support system messages
    if messages[0]['role'] == 'system':
        system_content = messages[0]['content']
        messages = messages[1:]

        # make sure system content is a list
        if not isinstance(system_content, list):
            system_content = [{
                'type': 'text',
                'text': system_content
            }]
        
        # make sure first turn is also a list
        if not isinstance(messages[0]['content'], list):
            messages[0]['content'] = [{
                'type': 'text',
                'text': messages[0]['content']
            }]
        
        # concat
        messages[0]['content'] = system_content + messages[0]['content']
    return messages

# whoever wrote this decorator does not know how to allow for optional arguments
@retry_with_exponential_backoff
def generate_from_openai_requestapi_chat_completion(
    client: None,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
    num_outputs: int = 1,
) -> Union[str, list[str]]:
    assert 'o1' in model
    global INITIAL_DELAY
    INITIAL_DELAY = 30
    cache_key = _completion_args_to_cache_key(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=-1,
        top_p=1.0,
        num_outputs=1
    )
    if not DISABLE_LLM_CACHE and cache_key in SIMPLE_LLM_API_CACHE:
        logger.info(f"generate_from_azure_openai_chat_completion hit cache")
        return SIMPLE_LLM_API_CACHE[cache_key]
    messages = _reformat_o1_messages(messages)
    url = os.environ['AZURE_O1_API_BASE']
    api_key = os.environ['AZURE_O1_API_KEY']

    headers = {'Content-Type': 'application/json', 'api-key': api_key}
    data = {  
        'model': 'o1-mini',
        "messages": messages,
        'n': 1,
    }

    resp = requests.post(url, json=data, headers=headers)
    completion = resp.json()
    logger.debug(f"Completion: {completion}")
    try:
        answer = completion['choices'][0]['message']['content']
    except Exception as e:
        raise ManualRateLimitError(f"Error in completion: {completion}")

    completion_tokens = completion['usage']['completion_tokens']
    reasoning_token = completion['usage']['completion_tokens_details']['reasoning_tokens']
    prompt_tokens = completion['usage']['prompt_tokens']
    logger.debug(f"Reasoning tokens: {reasoning_token}")

    token_stats = {
        'completion_tokens': completion_tokens + reasoning_token,
        'prompt_tokens': prompt_tokens,
        'num_requests': 1
    }
    update_token_usage(
        model_name=f"azure_{model}",
        token_stats=token_stats,
        token_usage_tracker=TOKEN_USAGE
    )

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
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )

    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer