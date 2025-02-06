from typing import Any
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from exact.llms import lm_config
from exact.llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_azure_openai_chat_completion,
    generate_from_ray_completion,
)
from exact.logging import time_it
from exact.llms.lm_config import LMConfig
from exact.llms.tokenizer import Tokenizer
from math import ceil
from PIL import Image
from io import BytesIO
import base64
import openai
import copy
import logging


logger = logging.getLogger("src.llms")


APIInput = str | list[Any] | dict[str, Any]


def is_vlm(model_name: str):
    if ("gemini" in model_name 
        or ("gpt-4" in model_name and "vision" in model_name)
        or "gpt-4o" in model_name):
        return True
    if "llava" in model_name or "mantis" in model_name or "InternVL2" in model_name:
        return True
    return False


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def b64_to_pil(img_b64: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    if img_b64.startswith(prefix):
        img_b64 = img_b64[len(prefix):]
    img = Image.open(BytesIO(base64.b64decode(img_b64)))
    return img


def configure_llm_client(lm_config: LMConfig):
    if lm_config.provider in ["openai", "sglang", "azure"]:
        if lm_config.provider == "azure":
            azure_credential = DefaultAzureCredential(
                exclude_managed_identity_credential=True,
            )
            token_provider = get_bearer_token_provider(
                azure_credential,
                lm_config.api_token_provider_base
            )
            client = openai.AzureOpenAI(
                api_version=lm_config.api_version,
                azure_endpoint=lm_config.api_base,
                azure_ad_token_provider=token_provider
            )
        else:
            client = openai.OpenAI(
                api_key=lm_config.api_key,
                organization=lm_config.api_organization,
                base_url=lm_config.api_base
            )
    elif lm_config.provider == "huggingface":
        client = None
    elif lm_config.provider == "google":
        client = None
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )
    return client


def display_multimodal_openai_messages(messages):
    all_texts = ""
    for i, message in enumerate(messages):
        role = message["role"]
        all_texts += f"Turn {i + 1} with role {role}\n"
        content = message["content"]
        if isinstance(content, str):
            all_texts += f"{content}\n"
        else:
            for c in content:
                data_type = c["type"]
                if data_type == "text":
                    all_texts += f"{c['text']}\n"
                elif data_type == "image_url":
                    truncated_image_url = c["image_url"]["url"][:50]
                    all_texts += f"Image URL: {truncated_image_url}...\n"
        all_texts += "\n\n"
    return all_texts.strip()


def _estimate_image_tokens(width: int, height: int):
    # following https://platform.openai.com/docs/guides/vision#calculating-costs
    # and verified with the calculator at https://openai.com/api/pricing/
    max_dim = max(width, height)
    if max_dim > 2048:
        width = width * 2048 / max_dim
        height = height * 2048 / max_dim

    # scale again so that short side is 768
    min_dim = min(width, height)
    if min_dim > 768:
        width = width * 768 / min_dim
        height = height * 768 / min_dim
    
    n_w_tiles = ceil(width / 512)
    n_h_tiles = ceil(height / 512)

    total = 85 + 170 * (n_w_tiles * n_h_tiles)
    return total


def _estimate_n_tokens(messages: list, tokenier: Tokenizer):
    n_tokens = 0
    for i, message in enumerate(messages):
        content = message["content"]
        if isinstance(content, str):
            n_tokens += tokenier.count_tokens(content)
        else:
            for c in content:
                data_type = c["type"]
                if data_type == "text":
                    n_tokens += tokenier.count_tokens(c["text"])
                elif data_type == "image_url":
                    image_b64 = c["image_url"]["url"]
                    image = b64_to_pil(image_b64)
                    width, height = image.size
                    n_tokens += _estimate_image_tokens(width, height)
    return n_tokens


def _flatten_chat_msg_turns(chat_msg, engine="vllm"):
    """converts the content=[{xxx}] into content="xxx.." so that apply_chat_template can be used
    e.g. vllm uses '\n' as a deliminater to join text content
    see https://github.com/vllm-project/vllm/blob/3aec49e56f60c8ccafe108a8922c731e235a8fcc/vllm/entrypoints/chat_utils.py#L764
    e.g. sglang directly separates them as a new turn, but with the same role
    see https://github.com/sgl-project/sglang/blob/a42213dbd4d952e9484ce0415ea53939d74a51db/python/sglang/srt/openai_api/adapter.py#L903
    """
    assert engine in ["vllm", "sglang"]
    
    combined_chat_msg = []
    if engine == "vllm":
        deliminater = "\n"
        for turn in chat_msg:
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                combined_chat_msg.append({
                    "role": role,
                    "content": content,
                })
            else:
                assert all([c['type'] == 'text' for c in content])
                joined_content = deliminater.join([c['text'] for c in content])
                combined_chat_msg.append({
                    "role": role,
                    "content": joined_content,
                })
    elif engine == "sglang":
        for turn in chat_msg:
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                combined_chat_msg.append({
                    "role": role,
                    "content": content,
                })
            else:
                assert all([c['type'] == 'text' for c in content])
                for c in content:
                    combined_chat_msg.append({
                        "role": role,
                        "content": c['text'],
                    })
    return combined_chat_msg


@time_it
def _truncate_prompt_to_max_tokens(
    prompt: APIInput,
    tokenizer: Tokenizer,
    keep_first_user_turn: bool = False,
    margin: float = 0.9,
):
    """truncate a list of openai prompt to max_tokens * margin by cutting turns from the start.

    Args:
        prompt (APIInput): _description_
        tokenizer (Tokenizer): _description_
        keep_first_user_turn (bool): _description_

    Returns:
        list[dict]: a COPY of the prompt with turns removed from the start, if modified
    """
    # keep system and start removing from the start
    curr_tokens = _estimate_n_tokens(prompt, tokenizer)
    if curr_tokens <= tokenizer.max_context_length * margin:
        logger.debug(f"Prompt is already within max tokens: {curr_tokens} <= {tokenizer.max_context_length * margin}")
        return prompt
    else:
        logger.debug(f"Truncating {len(prompt)} to max tokens: {tokenizer.max_context_length * margin}")
        truncated_prompt = copy.deepcopy(prompt)

        ## start prompting after system instruction
        first_role = truncated_prompt[0]["role"]
        if first_role == "system":
            start_idx = 1
        else:
            start_idx = 0
        assert truncated_prompt[start_idx]["role"] == "user", f"First turn is {truncated_prompt[start_idx]['role']=}?"
        
        if keep_first_user_turn and len(truncated_prompt[start_idx:]) > 2:
            # keep user's turn, so we start poping from turn 3 if (sys, usr, assistant)
            start_idx += 2
        truncated_prompt.pop(start_idx)  # pop first user

        while len(truncated_prompt) > start_idx:
            first_turn = truncated_prompt[start_idx]
            role = first_turn["role"]
            if role == 'assistant':
                # pop assistant and check again
                truncated_prompt.pop(start_idx)
                curr_tokens = _estimate_n_tokens(truncated_prompt, tokenizer)
                if curr_tokens <= tokenizer.max_context_length * margin:
                    # we are done
                    break
            else:
                # its user or system
                truncated_prompt.pop(start_idx)

        last_turn_len = _estimate_n_tokens([prompt[-1]], tokenizer)
        assert len(truncated_prompt) > start_idx, f"Truncated prompt is empty {truncated_prompt=}\n{last_turn_len=} with {prompt[-1]=}"
        assert truncated_prompt[-1]['role'] == 'user', f"Last turn should be user but is now {truncated_prompt=}"

        logger.debug(f"Truncated prompt to {len(truncated_prompt)} turns")
        return truncated_prompt


def _force_truncate_prompt_to_max_tokens(
    prompt: list[dict],
    tokenizer: Tokenizer,
    keep_first_user_turn: bool = False,
    margin: float = 0.9,
):
    """_truncate_prompt_to_max_tokens, by ALWAYS returns something
    """
    try:
        # this truncates by FULL turns. This works when model has decent context length, e.g., 32k tokens
        # but obviously training often cannot reach that sq length, hence we need to truncate last turn's content if this failed
        truncated_prompt = _truncate_prompt_to_max_tokens(
            prompt=prompt,
            tokenizer=tokenizer,
            keep_first_user_turn=keep_first_user_turn,
            margin=margin
        )
        return truncated_prompt
    except Exception as e:
        ## configure the initial turns that has to be there. Typically its just (system), but it may be (system, first user, first assistant)
        logger.info(f"Failed to truncate prompt by turns. Truncating the last turn's content instead.")
        force_truncated_prompt = []
        first_turn = prompt[0]
        if first_turn['role'] == 'system':
            force_truncated_prompt.append(first_turn)
        if keep_first_user_turn and prompt[1] != prompt[-1]:
            assert prompt[1]['role'] == 'user', f"Second turn should be user but is now {prompt[1]=}"
            force_truncated_prompt.append(prompt[1])
            assert prompt[2]['role'] == 'assistant', f"Third turn should be assistant but is now {prompt[2]=}"
            force_truncated_prompt.append(prompt[2])
            
        curr_tokens = _estimate_n_tokens(force_truncated_prompt, tokenizer)
        assert prompt[-1]['role'] == 'user', f"Last turn should be user but is now {prompt=}"
        force_truncated_prompt.append(prompt[-1])
        ### no need to truncate last turn, already screwed up
        if curr_tokens > tokenizer.max_context_length * margin:
            logger.warning(f"Impossible to truncate prompt to fit in max tokens as its already {curr_tokens=} with {force_truncated_prompt=}. Returning as is.")
            return force_truncated_prompt
        
        ### now we truncate the last turn to fit in the max tokens
        force_truncated_prompt = _flatten_chat_msg_turns(force_truncated_prompt, engine="vllm")
        curr_tokens = _estimate_n_tokens(force_truncated_prompt, tokenizer)
        if curr_tokens < tokenizer.max_context_length * margin:
            logger.warning(f"This should never happen as _truncate_prompt_to_max_tokens would have succeeded. {force_truncated_prompt=}")
            return force_truncated_prompt
        else:
            turn_to_truncate = force_truncated_prompt[-1]
            n_tokens_to_cut = int(curr_tokens - tokenizer.max_context_length * margin) + 1
            logger.debug(f"Truncating last turn {len(tokenizer.encode(turn_to_truncate['content']))=} by {n_tokens_to_cut=}")
            truncated_content = tokenizer.decode(tokenizer.encode(turn_to_truncate['content'])[n_tokens_to_cut:])
            force_truncated_prompt[-1]['content'] = truncated_content
            curr_tokens = _estimate_n_tokens(force_truncated_prompt, tokenizer)
            logger.info(f"Truncated prompt to {len(force_truncated_prompt)} turns with {curr_tokens=}.")
            return force_truncated_prompt


@time_it
def call_llm(
    client: Any,
    lm_config: LMConfig,
    prompt: APIInput,
    num_outputs: int = 1,
) -> str | list[str]:
    """Exclusively used by agent LLM"""
    logger.debug(f"Calling LLM with prompt:\n{display_multimodal_openai_messages(prompt)}")
    est_prompt_tokens = _estimate_n_tokens(prompt, lm_config.tokenizer_cls)
    logger.debug(f"Estimated prompt tokens: {est_prompt_tokens}")

    if lm_config.provider in ["openai", "sglang", "azure"]:
        assert isinstance(prompt, list)
        assert client is not None, "Client must be provided for OpenAI, Azure, or SGlang"
        if lm_config.provider in ["openai", "sglang"]:
            response = generate_from_openai_chat_completion(
                client,
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
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
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
                num_outputs=num_outputs,
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
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    logger.debug(f"Response from LLM:\n{response}")
    return response


def call_classification_llm(
    client: Any,
    lm_config: LMConfig,
    prompt: APIInput,
    num_outputs: int = 1,
) -> dict:
    logger.debug(f"Calling model {lm_config.model} at {lm_config.api_base} with pooling")
    assert num_outputs == 1, "Pooling only supports single output"
    
    api_url = f"{lm_config.api_base}/classify"
    return generate_from_ray_completion(
        api_url=api_url,
        model_name=lm_config.model,
        messages=prompt,
    )