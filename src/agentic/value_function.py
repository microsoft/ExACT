import os
import re
import copy
from concurrent.futures import ThreadPoolExecutor
from cachetools import Cache
from typing import Optional
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from browser_env.utils import pil_to_b64
from src.llms.providers.openai_utils import (
    update_token_usage,
    generate_from_openai_chat_completion, generate_from_azure_openai_chat_completion
)
from src.constants import TOKEN_USAGE
from src.logging import time_it
from src.agent.utils import _pil_image_to_str
from src.prompts.utils import display_multimodal_openai_messages
import numpy as np
from PIL import Image
import re
import logging


logger = logging.getLogger("value_function")


VALUE_FUNCTION_PROVIDER = os.getenv("VALUE_FUNC_PROVIDER", "")


def configure_client():
    api_base = os.getenv("VALUE_FUNC_API_BASE", "")
    if VALUE_FUNCTION_PROVIDER in ["openai", "sglang"]:
        if api_base != "":
            logger.warn(f"VALUE_FUNC_API_BASE is set to {api_base}. This is expected if you are hosting sglang/vllm.")
        else:
            logger.warn(f"VALUE_FUNC_API_BASE is not set. Using GPT-4o as value function.")
            api_base = "https://api.openai.com/v1"
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
    elif VALUE_FUNCTION_PROVIDER == "azure":
        token_provider_base = os.getenv("AZURE_TOKEN_PROVIDER_BASE", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
        logger.info(f"Using Azure OpenAI with {api_base=}, {token_provider_base=}, {api_version=}")

        azure_credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(azure_credential, token_provider_base)
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=api_base,
            azure_ad_token_provider=token_provider
        )
    else:
        logging.error(f"Unknown provider: {VALUE_FUNCTION_PROVIDER} for value function.")
        client = None
    return client


client: OpenAI | AzureOpenAI = configure_client()


def create_chat_completion_wrapper(
    messages: list[dict],
    model: str,
    temperature: float = 0.9,
    max_tokens: int = 256,
    top_p: float = 1.0,
    context_length: int = -1,
    num_outputs: int = 1,
):
    # these functions records token usages + does retries
    # this function should be used by value_func and rvalue_func classes. Policy uses call_llm
    if VALUE_FUNCTION_PROVIDER == "azure":
        response = generate_from_azure_openai_chat_completion(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            context_length=context_length,
            num_outputs=num_outputs,
        )
    else:
        response = generate_from_openai_chat_completion(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            context_length=context_length,
            num_outputs=num_outputs,
        )
    return response


VFUNC_INTRO = f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
""".strip()


class ValueFunction:
    def __init__(self, *args, **kwargs):
        return
    
    @staticmethod
    def evaluate_success(
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        screenshots_text: Optional[list[str]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        raise NotImplementedError


VFUNC_DIRECT_COT_INTRO = f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's last response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

Below is the user's intent, and the last few state/action pairs of the agent's attempt to solve the provided task.
""".strip()


VFUNC_DIRECT_COT_FINAL_PROMPT = f"""
Now, it is your turn to evaluate the success/failure of the agent's execution so far.

Remeber that user's intent can be broadly categorized into three types:
1. Information seeking; 2. Site navigation; 3. Content modification.

*IMPORTANT*
To evaluate the success/failure of current state, choose one of the following status codes and provide your thought process.
STATUS CODES:
A. The agent's last action already contains the correct answer (for Task 1), or the current state fulfilled all requirements in the user's intent (for Task 2 and 3). No further action is needed.
B. The agent is very close to finishing the task. Only one more action (e.g., provide the correct answer) is needed to fulfill the user's intent.
C. The agent is on the right track to successfully complete the task, but more than one actions are still needed to fulfill the user's intent.
D. The current state may or may not be a failure. It is unclear now if the agent is on the right track to success.
E. The agent's last response/current state does not make progress towards the task completion. The agent is not on the right track to success.

Format your response into three lines as shown below. Keep your response concise.
Rubric: <based on the user's intent, come up with a few heuristics to determine if the task is successful>
Thoughts: <your thoughts and reasoning process>
STATUS CODE: A, B, C, D, or E
""".strip()


class DirectCoTValueFunction(ValueFunction):
    @staticmethod
    def _construct_prompt(
        screenshots: list[Image.Image],  # all screenshots after the initial one
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: Optional[list[Image.Image]] = None,
    ):
        logger.debug(f'evaluating last response={actions[-1]}')
        logger.debug(f"received {len(screenshots)=} and {len(actions)=}")

        max_turns = 2  # in the prompt, there will be max_turns + 1 screenshots and actions
        ## prepare screenshots and actions
        # trajectory is either (s,a,s,a,s), or (s,a,s,a) or stop actions
        if len(screenshots) > len(actions):
            # the first case (s,a,s,a,s)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_action = None

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        else:
            # the second case (s,a,s,a)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_action = actions.pop(-1)

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        if intent_images is not None:
            start_content = []
            for i_img_idx, i_img in enumerate(intent_images):
                start_content.extend([
                    {
                        "type": "text",
                        "text": f"INTENT IMAGE: ({i_img_idx})",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(i_img),
                        },
                    }
                ])
            start_content.append({
                "type": "text",
                "text": (
                    f"User Intent: {intent}"
                )
            })
            start_content.extend([
                {
                    "type": "text",
                    "text": "IMAGES: (1) current page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ])
        else:
            start_content = [
                {
                    "type": "text",
                    "text": (
                        f"User Intent: {intent}"
                    )
                },
                {
                    "type": "text",
                    "text": "IMAGES: (1) current page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ]
        messages = [
            {
                "role": "system",
                "content": VFUNC_DIRECT_COT_INTRO
            },
            {
                "role": "user",
                "content": start_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": init_action
                    },
                ]
            }
        ]
        for i, (screenshot, action) in enumerate(zip(last_k_screenshots, last_k_actions)):
            if i == 0 and has_truncation:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "(fast forwarding to the last few state/action pairs...)\nIMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action
                    },
                ]
            })
        # last turn prompt
        if last_action is not None:
            # eval stop actions
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VFUNC_DIRECT_COT_FINAL_PROMPT
                    },
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\nIMAGES: (1) last page screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": f"Agent's final action: {last_action}"
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VFUNC_DIRECT_COT_FINAL_PROMPT
                    },
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\nIMAGES: (1) last page screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                ]
            })
        return messages

    @time_it
    @staticmethod
    def evaluate_success(
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        """Compute the value of a state using the value function.

        screenshots should REALLY be all screenshots involved in the current task, including the initial one.
        """
        screenshots = screenshots[-4:]
        messages = DirectCoTValueFunction._construct_prompt(
            screenshots=copy.deepcopy(screenshots),  # histories
            actions=copy.deepcopy(actions),  # histories
            last_state_url=current_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images
        )

        all_responses = []
        for model in models:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                top_p=top_p,
                n=n // len(models)
            )

            token_stats = {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'num_requests': 1
            }
            if VALUE_FUNCTION_PROVIDER == "azure":
                model_log_name = f"azure_{model}"
            else:
                model_log_name = model
            
            update_token_usage(
                model_name=model_log_name,
                token_stats=token_stats,
                token_usage_tracker=TOKEN_USAGE
            )
            all_responses.extend(response.choices)
        
        ### parse the responses to scores
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            logger.debug(f"Response {r_idx}: {r.message.content}")
            try:
                pred = re.search(r'.*STATUS CODE: (\w).*', r.message.content).group(1)
                if 'A' in pred:
                    score = 1.0
                elif 'B' in pred:
                    score = 0.9
                elif 'C' in pred:
                    score = 0.5
                elif 'D' in pred:
                    score = 0.2
                else:
                    score = 0.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0
            
            all_scores.append(score)
        
        score = np.mean(all_scores)
        logger.debug(f"Final score from vfunc {models}: {score} from {all_scores}")
        return score


VFUNC_GEN_RUBRIC_PROMPT = """
Your task now is to provide a short rubric that can be used to evaluate if the user's intent has been successfully fulfilled.
The rubric should be based on the user's intent and the relevant images provided, and will be used to evaluate if the agent's final action has successfully completed the task.

To produce a good rubric, consider:
- What are some (simple) mistakes the agent could make that would lead to an incorrect answer?
- Any incorrect answer/action should fail at least one of the rubric points.
- Items in the rubric should be simple, clear, and easy to evaluate.
- You should have no more than four items in the rubric.

For example, if the user intent is to find a pink dress that costs less than $50, a good rubric could be:
Rubric:
[RUBRIC START]
1. Is the selected product a dress?
2. Is the selected dress pink?
3. Does the selected dress cost less than $50?
[RUBRIC END]
""".strip()


VFUNC_RUBRIC_FINAL_PROMPT = """
Now, it is your turn to evaluate the success/failure of the agent's execution so far.

Remeber that user's intent can be broadly categorized into three types:
1. Information seeking; 2. Site navigation; 3. Content modification.

*IMPORTANT*
To evaluate the success/failure of current state, choose one of the following status codes and provide your thought process.
STATUS CODES:
A. The agent's last action already contains the correct answer (for Task 1), or the current state fulfilled all requirements in the user's intent (for Task 2 and 3). No further action is needed.
B. The agent is very close to finishing the task. Only one more action (e.g., provide the correct answer) is needed to fulfill the user's intent.
C. The agent is on the right track to successfully complete the task, but more than one actions are still needed to fulfill the user's intent.
D. The current state may or may not be a failure. It is unclear now if the agent is on the right track to success.
E. The agent's last response/current state does not make progress towards the task completion. The agent is not on the right track to success.

To better verify if user's intent is fulfilled correctly, you may find the following rubric helpful:
{rubric}

Format your response into two lines as shown below. Keep your response concise.
Thoughts: <your thoughts and reasoning process>
STATUS CODE: A, B, C, D, or E
""".strip()


class RubricBasedValueFunctionMixin:
    _rubrics_cache = Cache(maxsize=100)  # intent + encoded intent images -> rubric

    @staticmethod
    def generate_rubrics(
        intent: str,
        intent_images: Optional[list[Image.Image]],
        init_screenshot: Image.Image,
        model: str
    ) -> str:
        raise NotImplementedError


class CoTwRubricValueFunction(DirectCoTValueFunction, RubricBasedValueFunctionMixin):
    @staticmethod
    def _construct_prompt(
        screenshots: list[Image.Image],  # all screenshots after the initial one
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_rubrics: str,
        intent_images: Optional[list[Image.Image]] = None,
    ):
        logger.debug(f'evaluating last response={actions[-1]}, reasoning={last_reasoning}')
        logger.debug(f"received {len(screenshots)=} and {len(actions)=}")

        max_turns = 2  # in the prompt, there will be max_turns + 1 screenshots and actions
        ## prepare screenshots and actions
        # trajectory is either (s,a,s,a,s), or (s,a,s,a) or stop actions
        if len(screenshots) == 1 and len(actions) == 1:
            # (s, stop)
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = None
            last_action = actions[0]

            last_k_screenshots = []
            last_k_actions = []

            has_truncation = False
        elif len(screenshots) > len(actions):
            # the first case (s,a,s,...,a,s)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_action = None

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        else:
            # the second case (s,a,s,...,a)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_action = actions.pop(-1)

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        if intent_images is not None:
            start_content = []
            for i_img_idx, i_img in enumerate(intent_images):
                start_content.extend([
                    {
                        "type": "text",
                        "text": f"INTENT IMAGE: ({i_img_idx})",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(i_img),
                        },
                    }
                ])
            start_content.append({
                "type": "text",
                "text": (
                    f"User Intent: {intent}"
                )
            })
            start_content.extend([
                {
                    "type": "text",
                    "text": "IMAGES: (1) start page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ])
        else:
            start_content = [
                {
                    "type": "text",
                    "text": (
                        f"User Intent: {intent}"
                    )
                },
                {
                    "type": "text",
                    "text": "IMAGES: (1) start page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ]
        messages = [
            {
                "role": "system",
                "content": VFUNC_DIRECT_COT_INTRO
            },
            {
                "role": "user",
                "content": start_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": init_action
                    },
                ]
            }
        ]
        for i, (screenshot, action) in enumerate(zip(last_k_screenshots, last_k_actions)):
            if i == 0 and has_truncation:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "(fast forwarding to the last few state/action pairs...)\nIMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action
                    },
                ]
            })
        # last turn prompt
        if last_screenshot is None:
            # the (s, stop) case
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Agent's final action: {last_action}"
                    },
                    {
                        "type": "text",
                        "text": VFUNC_RUBRIC_FINAL_PROMPT.format(rubric=intent_rubrics)
                    },
                ]
            })
        elif last_action is not None:
            # eval stop actions
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\nIMAGES: (1) last page screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": f"Agent's final action: {last_action}"
                    },
                    {
                        "type": "text",
                        "text": VFUNC_RUBRIC_FINAL_PROMPT.format(rubric=intent_rubrics)
                    },
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\nIMAGES: (1) last page screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": VFUNC_RUBRIC_FINAL_PROMPT.format(rubric=intent_rubrics)
                    },
                ]
            })
        return messages

    @staticmethod
    def _construct_rubric_prompt(
        intent: str,
        intent_images: Optional[list[Image.Image]],
        init_screenshot: Image.Image
    ) -> list:
        # init_screenshot is needed since sometimes the intent refers to "... on this page"
        if intent_images is not None:
            start_content = []
            for i_img_idx, i_img in enumerate(intent_images):
                start_content.extend([
                    {
                        "type": "text",
                        "text": f"INTENT IMAGE: ({i_img_idx})",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(i_img),
                        },
                    }
                ])
            start_content.append({
                "type": "text",
                "text": (
                    f"User Intent: {intent}"
                )
            })
            start_content.extend([
                {
                    "type": "text",
                    "text": "IMAGES: (1) start page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ])
        else:
            start_content = [
                {
                    "type": "text",
                    "text": (
                        f"User Intent: {intent}"
                    )
                },
                {
                    "type": "text",
                    "text": "IMAGES: (1) start page screenshot",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(init_screenshot)},
                },
            ]
        start_content.append({
            "type": "text",
            "text": "Rubric:\n",
        })
        messages = [
            {
                "role": "system",
                "content": VFUNC_DIRECT_COT_INTRO
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": VFUNC_GEN_RUBRIC_PROMPT
                }]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Got it! Please provide the user's intent or any relevant information I should use to create an evaluuation rubric."
                    },
                ]
            },
            {
                "role": "user",
                "content": start_content
            }
        ]
        return messages

    @staticmethod
    def _extract_rubric(raw_answer_str: str):
        rubric_start_idx = raw_answer_str.find("[RUBRIC START]") + len("[RUBRIC START]")
        rubric_end_idx = raw_answer_str.find("[RUBRIC END]")
        rubric_str = raw_answer_str[rubric_start_idx:rubric_end_idx]
        return rubric_str.strip()

    @staticmethod
    def generate_rubrics(
        intent: str,
        intent_images: Optional[list[Image.Image]],
        init_screenshot: Image.Image,
        model: str
    ) -> str:
        # convert image to base64
        if intent_images is not None:
            encoded_image_str = ""
            for img in intent_images:
                encoded_image_str += pil_to_b64(img)
        else:
            encoded_image_str = ""
        
        if (intent, encoded_image_str) in CoTwRubricValueFunction._rubrics_cache:
            logger.debug("fetching rubric from cache")
            return CoTwRubricValueFunction._rubrics_cache[(intent, encoded_image_str)]

        ### generate
        logger.debug("generating new rubrics")
        messages = CoTwRubricValueFunction._construct_rubric_prompt(
            intent,
            intent_images,
            init_screenshot
        )
        raw_rubric = create_chat_completion_wrapper(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=256,
        )
        try:
            extracted_rubric = CoTwRubricValueFunction._extract_rubric(raw_rubric)
            # save to cache
            CoTwRubricValueFunction._rubrics_cache[(intent, encoded_image_str)] = extracted_rubric
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.error(f"Error extracting rubric")
            extracted_rubric = "None"
        return extracted_rubric
    
    @time_it
    def evaluate_success(
        self,
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        screenshots_text: Optional[list[str]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        """Compute the value of a state using the value function.

        TODO: screenshots should be ALL screenshots involved in the current task, not just the last few starting from the root node
        """
        assert init_screenshot is not None   # should not be none, only for backward compatibility
        intent_rubrics = self.generate_rubrics(
            intent,
            intent_images,
            init_screenshot=init_screenshot,
            model=models[0]
        )
        logger.info(f"Value function using rubrics:\n{intent_rubrics}")
        messages = self._construct_prompt(
            screenshots=copy.deepcopy(screenshots),  # histories
            actions=copy.deepcopy(actions),  # histories
            last_state_url=current_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_rubrics=intent_rubrics,
            intent_images=intent_images
        )

        logger.debug(f"vfunc constructed full prompt:\n{display_multimodal_openai_messages(messages)}")

        all_responses = []
        for model in models:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                temperature=0.7,
                top_p=top_p,
                n=n // len(models)
            )

            token_stats = {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'num_requests': 1
            }
            if VALUE_FUNCTION_PROVIDER == "azure":
                model_log_name = f"azure_{model}"
            else:
                model_log_name = model
            
            update_token_usage(
                model_name=model_log_name,
                token_stats=token_stats,
                token_usage_tracker=TOKEN_USAGE
            )
            all_responses.extend(response.choices)
        
        ### parse the responses to scores
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            logger.debug(f"Response {r_idx}: {r.message.content}")
            try:
                pred = re.search(r'.*STATUS CODE: (\w).*', r.message.content).group(1)
                if 'A' in pred:
                    score = 1.0
                elif 'B' in pred:
                    score = 0.9
                elif 'C' in pred:
                    score = 0.5
                elif 'D' in pred:
                    score = 0.2
                else:
                    score = 0.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0
            
            all_scores.append(score)
        
        score = np.mean(all_scores)
        logger.debug(f"Final score from vfunc {models}: {score} from {all_scores}")
        return score


VFUNC_DEBATE_INTRO = f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's last response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

The agent's actions fall into several categories:
Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content] [1/0]```: Use this to type the content into the field with id, followed by pressing ``Enter`` to submit the form [1] or no submission [0].
```hover [id]```: Hover over an element with id.
```press [key_comb]```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab. Note that most tasks can be completed with the tabs we provided.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index (e.g., ```tab_focus [1]``` switches to the SECOND tab).
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer/url]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer (e.g., price), provide the answer in the bracket. If the objective is to find a link(s) to an item/post(s), provide the exact url(s) in the bracket (for example, stop [http://xxx]).


Below is the user's intent, and the last few state/action pairs of the agent's attempt to solve the provided task.
""".strip()


VFUNC_SUPPORTING_OPINION_PROMPT = """
Now, it is your turn to evaluate the success/failure of the agent's execution so far.

Remeber that:
- The user's intent can be broadly categorized into three types: Type 1. Information seeking; Type 2. Site navigation; Type 3. Content modification.
- The agent should only issue an action that is valid given the current (i.e., last) observation/state.
- A ```stop``` action is issued when the agent thinks the user's intent/task is achieved. No actions can be issued after ```stop```.

And your goal is to find the most convincing evidence why the agent is on the right track to successfully complete the task. You should consider:
- Maybe the last (stop) action contains the correct answer to the user's intent (type 1 tasks)?
- Maybe the current state fulfilled all requirements specified in the user's intent (type 2 and 3 tasks)?
- Maybe the agent is very close to finishing the task. Only one more action (e.g., provide the correct answer) is needed to fulfill the user's intent?

*IMPORTANT*: To find the most convincing evidence, you should:
1. Carefully check if the user's intent is fulfilled correctly (e.g., does it have the desired color, price, material, date, or is identical/similar to the intent image if provided).
2. Be specific, concise, and accurate: pointing out the exact aspect that the agent is doing well.
3. **Your response needs to be FACTUALLY CORRECT based on the user's intent, the observations, and the agent's executed actions so far.**
Note that being able to find the correct item, category, user, repository, subreddit, or post page is often the strongest evidence for (future) success.

Keep your response within 100 words.
""".strip()


VFUNC_OPPOSING_OPINION_PROMPT = """
Now, it is your turn to evaluate the success/failure of the agent's execution so far.

Remeber that:
- The user's intent can be broadly categorized into three types: Type 1. Information seeking; Type 2. Site navigation; Type 3. Content modification.
- The agent should only issue an action that is valid given the current (i.e., last) observation/state.
- A ```stop``` action is issued when the agent thinks the user's intent/task is achieved. No actions can be issued after ```stop```.

And your goal is to find the most convincing evidence why the agent is NOT on the right track to complete the task. You should consider:
- Maybe the agent's answer missed some important aspects of the user's intent (type 1 tasks)?
- Maybe the current state missed some of the all requirements specified in the user's intent (type 2 and 3 tasks)?
- Maybe the current state or the agent's actions are unreasonable given the user's intent?

*IMPORTANT*: To find the most convincing evidence, you should:
1. Carefully check if the user's intent is fulfilled correctly (e.g., does it have the desired color, price, material, date, or is identical/similar to the intent image if provided).
2. Be specific, concise, and accurate: pointing out the exact issue that the agent is facing.
3. **Your response needs to be FACTUALLY CORRECT based on the user's intent, the observations, and the agent's executed actions so far.**
Note that failing to find the right item, category, user, repository, subreddit, or post page is often the strongest evidence for failing the task.

Keep your response within 100 words.
""".strip()


VFUNC_FINAL_DECISION_PROMPT = """
Now, it is your turn to evaluate whether the agent's actions so far are successfully aligned with the user's intent.

Remeber that:
- The user's intent can be broadly categorized into three types: Type 1. Information seeking; Type 2. Site navigation; Type 3. Content modification.
- The agent should only issue an action that is valid given the current (i.e., last) observation/state.
- A ```stop``` action is issued ONLY WHEN the agent thinks the user's intent/task is FINISHED. No actions can be issued after ```stop```.

To better verify if user's intent is fulfilled correctly, you may find the following opinions helpful:
{opinion_1}
{opinion_2}
Note that these opinions may or may NOT be correct. You should make your own judgment based on the user's intent, the observations, and the agent's executed actions so far.

*IMPORTANT*
To make a final decision, choose one of the following status codes and provide your thought process.
STATUS CODES:
A. The agent's last action is ```stop``` and it contains the correct answer (for Type 1), or the current state fulfilled all requirements in the user's intent (for Type 2 and 3). No further action is needed.
B. The agent is very close to finishing the task. Only one more action (e.g., provide the correct answer) is needed to fulfill the user's intent.
C. The agent needs a few more actions to successfully complete the task, and the current action is likely the *OPTIMAL* choice to meet the user's intent as quickly as possible.
D. The agent needs a few more actions to complete the task, but the current action is likely *NOT* the best choice to meet the user's intent as quickly as possible.
E. The agent's last response/current state does not make progress towards the task completion, proposed action is invalid, or the last ```stop``` action contains an incorrect answer.

Format your response into two lines as shown below. Keep your response concise.
Thoughts: <your thoughts and reasoning process>
STATUS CODE: A, B, C, D, or E
""".strip()


class DebateBasedValueFunctionMixin:
    _debate_cache = Cache(maxsize=100)  # intent + encoded intent images -> rubric

    @staticmethod
    def _encode_eval_success_input(
        screenshots: list[Image.Image],  # all screenshots after the initial one
        actions: list[str],
        intent: str,
        intent_images: list[Image.Image],
    ) -> str:
        all_images = screenshots + intent_images
        all_str_joined = f"{intent};{'->'.join(actions)}"
        all_images_encoded = _pil_image_to_str(all_images)
        return f"{all_str_joined};{all_images_encoded}"

    def generate_opposing_opinions(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        model: str
    ) -> str:
        raise NotImplementedError

    def generate_supporting_opinions(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        model: str
    ) -> str:
        raise NotImplementedError

    def generate_final_decisions(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],  # used by reinforced debate value function
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        opposing_opinions: str,
        supporting_opinions: str,
        model: str
    ) -> list[str]:
        raise NotImplementedError



class CoTwDebateValueFunction(DirectCoTValueFunction, DebateBasedValueFunctionMixin):
    @staticmethod
    def _extract_opinion(raw_answer_str: str):
        # op_start_idx = raw_answer_str.find("[OPINION START]") + len("[OPINION START]")
        # op_end_idx = raw_answer_str.find("[OPINION END]")
        # op_str = raw_answer_str[op_start_idx:op_end_idx]
        return raw_answer_str.strip()
    
    def _construct_opinion_prompt(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        side: str # supporting or opposing
    ) -> list:
        assert side in ["supporting", "opposing"], f"Invalid side: {side}"
        logger.debug(f'evaluating last response={actions[-1]}, reasoning={last_reasoning}')
        logger.debug(f"received {len(screenshots)=} and {len(actions)=}")

        max_turns = 4  # in the prompt, there will be max_turns + 1 screenshots and actions
        ## prepare screenshots and actions
        # trajectory is either (s,a,s,a,s), or (s,a,s,a) or stop actions
        if len(screenshots) == 1 and len(actions) == 1:
            # (s, stop)
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = None
            last_action = actions[0]

            last_k_screenshots = []
            last_k_actions = []

            has_truncation = False
        elif len(screenshots) > len(actions):
            # the first case (s,a,s,...,a,s)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_screenshot_text = screenshots_text.pop(-1)
            last_action = None

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        else:
            # the second case (s,a,s,...,a)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_screenshot_text = screenshots_text.pop(-1)
            last_action = actions.pop(-1)

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)

        ### 1. start content and init screenshot
        start_content = [
            {
                "type": "text",
                "text": (
                    f"User Intent: {intent}"
                )
            }
        ]
        i_img_idx = -1
        if intent_images is not None:
            intent_images_content = []
            for i_img_idx, i_img in enumerate(intent_images):
                intent_images_content.extend([
                    {
                        "type": "text",
                        "text": f"INTENT IMAGE: ({i_img_idx+1})",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(i_img),
                        },
                    }
                ])
            start_content.extend(intent_images_content)
        start_content.extend([
            {
                "type": "text",
                "text": f"IMAGES: ({i_img_idx+2}) start page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_screenshot)},
            }
        ])
        
        #### 2. first action. Now we have (s0,a0)
        messages = [
            {
                "role": "system",
                "content": VFUNC_DEBATE_INTRO
            },
            {
                "role": "user",
                "content": start_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": init_action
                    },
                ]
            }
        ]

        ##### 3. the rest of (s, a, ...) after the first one
        ## 3.1 loop enter all turns before the last turn
        for i, (screenshot, action) in enumerate(zip(last_k_screenshots, last_k_actions)):
            if i == 0 and has_truncation:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "(fast forwarding to the last few state/action pairs...)\nIMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action
                    },
                ]
            })
        ## 3.2 last turn prompt, which could be either (s), (s,a), or nothing
        if last_screenshot is None:
            # the (s, stop) case
            # there is nothing
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Agent's final action was: {last_action}"
                    },
                    {
                        "type": "text",
                        "text": VFUNC_SUPPORTING_OPINION_PROMPT if side == "supporting" else VFUNC_OPPOSING_OPINION_PROMPT
                    },
                ]
            })
        elif last_action is not None:
            # eval stop actions
            # there is a (s,a) pair
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\n",
                    },
                    # {   "type": "text",
                    #     "text": f"OBSERVATION:\n{last_screenshot_text}"  # when there are caption errors, this makes it worse
                    # },
                    {
                        "type": "text",
                        "text": "IMAGES: (1) last page screenshot."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": f"Agent's final action: {last_action}"
                    },
                    {
                        "type": "text",
                        "text": VFUNC_SUPPORTING_OPINION_PROMPT if side == "supporting" else VFUNC_OPPOSING_OPINION_PROMPT
                    },
                ]
            })
        else:
            # there is a (s) case
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\n",
                    },
                    # {   "type": "text",
                    #     "text": f"OBSERVATION:\n{last_screenshot_text}"  # when there are caption errors, this makes it worse
                    # },
                    {
                        "type": "text",
                        "text": "IMAGES: (1) last page screenshot."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": VFUNC_SUPPORTING_OPINION_PROMPT if side == "supporting" else VFUNC_OPPOSING_OPINION_PROMPT
                    },
                ]
            })
        return messages

    def _construct_final_decision_prompt(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        opposing_opinions: str,
        supporting_opinions: str,
        opposing_first: bool
    ) -> list:
        logger.debug(f'evaluating final decision for last response={actions[-1]}')
        logger.debug(f"received {len(screenshots)=} and {len(actions)=}")

        max_turns = 4  # in the prompt, there will be max_turns + 1 screenshots and actions
        ## prepare screenshots and actions
        # trajectory is either (s,a,s,a,s), or (s,a,s,a) or stop actions
        if len(screenshots) == 1 and len(actions) == 1:
            # (s, stop)
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = None
            last_action = actions[0]

            last_k_screenshots = []
            last_k_actions = []

            has_truncation = False
        elif len(screenshots) > len(actions):
            # the first case (s,a,s,...,a,s)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_screenshot_text = screenshots_text.pop(-1)
            last_action = None

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)
        else:
            # the second case (s,a,s,...,a)
            assert len(screenshots) >= 2
            init_screenshot = screenshots[0]
            init_action = actions[0]

            last_screenshot = screenshots.pop(-1)
            last_screenshot_text: str = screenshots_text.pop(-1)
            last_action = actions.pop(-1)

            last_k_screenshots = screenshots[1:]
            last_k_screenshots = last_k_screenshots[-max_turns:]

            last_k_actions = actions[1:]
            last_k_actions = last_k_actions[-max_turns:]

            has_truncation = len(last_k_screenshots) + 2 < len(screenshots)

        ### 1. start content and init screenshot
        start_content = [
            {
                "type": "text",
                "text": (
                    f"User Intent: {intent}"
                )
            }
        ]
        i_img_idx = -1
        if intent_images is not None:
            intent_images_content = []
            for i_img_idx, i_img in enumerate(intent_images):
                intent_images_content.extend([
                    {
                        "type": "text",
                        "text": f"INTENT IMAGE: ({i_img_idx+1})",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(i_img),
                        },
                    }
                ])
            start_content.extend(intent_images_content)
        start_content.extend([
            {
                "type": "text",
                "text": f"IMAGES: ({i_img_idx+2}) start page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_screenshot)},
            }
        ])
        
        #### 2. first action. Now we have (s0,a0)
        messages = [
            {
                "role": "system",
                "content": VFUNC_DEBATE_INTRO
            },
            {
                "role": "user",
                "content": start_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": init_action
                    },
                ]
            }
        ]

        ##### 3. the rest of (s, a, ...) after the first one
        ## 3.1 loop enter all turns before the last turn
        for i, (screenshot, action) in enumerate(zip(last_k_screenshots, last_k_actions)):
            if i == 0 and has_truncation:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "(fast forwarding to the last few state/action pairs...)\nIMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(screenshot)},
                        },
                    ]
                })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action
                    },
                ]
            })
        ## 3.2 last turn prompt, which could be either (s), (s,a), or nothing
        if last_screenshot is None:
            # the (s, stop) case
            # there is nothing
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Agent's final action was: {last_action}"
                    }
                ]
            })
        elif last_action is not None:
            # eval stop actions
            # there is a (s,a) pair
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\n",
                    },
                    # {   "type": "text",
                    #     "text": f"OBSERVATION:\n{last_screenshot_text}"  # when there are caption errors, this makes it worse
                    # },
                    {
                        "type": "text",
                        "text": "IMAGES: (1) last page screenshot."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    },
                    {
                        "type": "text",
                        "text": f"Agent's final action: {last_action}"
                    }
                ]
            })
        else:
            # there is a (s) case
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Last page URL={last_state_url}\n"
                    },
                    # {   "type": "text",
                    #     "text": f"OBSERVATION:\n{last_screenshot_text}"  # when there are caption errors, this makes it worse
                    # },
                    {
                        "type": "text",
                        "text": "IMAGES: (1) last page screenshot."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(last_screenshot)},
                    }
                ]
            })
        if opposing_first:
            messages[-1]['content'].append({
                "type": "text",
                "text": VFUNC_FINAL_DECISION_PROMPT.format(
                    opinion_1="Reasons why the agent is NOT on the right track:\n" + opposing_opinions,
                    opinion_2="Reasons why the agent is on the right track:\n" + supporting_opinions
                )
            })
        else:
            messages[-1]['content'].append({
                "type": "text",
                "text": VFUNC_FINAL_DECISION_PROMPT.format(
                    opinion_1="Reasons why the agent is on the right track:\n" + supporting_opinions,
                    opinion_2="Reasons why the agent is NOT on the right track:\n" + opposing_opinions
                )
            })
        return messages
    
    def generate_supporting_opinions(
        self,
        screenshots: list[Image.Image],  # all screenshots after the initial one
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        model: str
    ) -> str:
        ### generate
        logger.debug("generating supporting arguments")
        messages = self._construct_opinion_prompt(
            screenshots=screenshots,
            screenshots_text=screenshots_text,
            actions=actions,
            last_state_url=last_state_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            side="supporting"
        )
        logger.debug(f"vfunc constructed supporting prompt:\n{display_multimodal_openai_messages(messages)}")
        raw_opinion = create_chat_completion_wrapper(
            model=model,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            max_tokens=256
        )
        try:
            extracted_opinion = self._extract_opinion(raw_opinion)
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.error(f"Error extracting supporting opinon from response: {raw_opinion}")
            extracted_opinion = "None"
        return extracted_opinion

    def generate_opposing_opinions(
        self,
        screenshots: list[Image.Image],
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        model: str
    ) -> str:
        ### generate
        logger.debug("generating opposing arguments")
        messages = self._construct_opinion_prompt(
            screenshots=screenshots,
            screenshots_text=screenshots_text,
            actions=actions,
            last_state_url=last_state_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            side="opposing"
        )
        logger.debug(f"vfunc constructed opposing prompt:\n{display_multimodal_openai_messages(messages)}")
        raw_opinion = create_chat_completion_wrapper(
            model=model,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            max_tokens=256
        )
        try:
            extracted_opinion = self._extract_opinion(raw_opinion)
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.error(f"Error extracting opposing opinon from response: {raw_opinion}")
            extracted_opinion = "None"
        return extracted_opinion
    
    def generate_final_decisions(
        self,
        screenshots: list[Image.Image],
        screenshots_text: list[str],
        actions: list[str],
        last_state_url: str,
        last_reasoning: str,
        intent: str,
        intent_images: list[Image.Image],
        opposing_opinions: str,
        supporting_opinions: str,
        model: str
    ) -> list[str]:
        ### generate
        logger.debug("generating final decisions")
        oppose_first_messages = self._construct_final_decision_prompt(
            screenshots=copy.deepcopy(screenshots),
            screenshots_text=copy.deepcopy(screenshots_text),
            actions=copy.deepcopy(actions),
            last_state_url=last_state_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            opposing_opinions=opposing_opinions,
            supporting_opinions=supporting_opinions,
            opposing_first=True
        )
        support_first_messages = self._construct_final_decision_prompt(
            screenshots=copy.deepcopy(screenshots),
            screenshots_text=copy.deepcopy(screenshots_text),
            actions=copy.deepcopy(actions),
            last_state_url=last_state_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            opposing_opinions=opposing_opinions,
            supporting_opinions=supporting_opinions,
            opposing_first=False
        )
        logger.debug(f"vfunc constructed final decision prompt 1:\n{display_multimodal_openai_messages(oppose_first_messages)}")
        logger.debug(f"vfunc constructed final decision prompt 2:\n{display_multimodal_openai_messages(support_first_messages)}")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            futures.append(executor.submit(
                create_chat_completion_wrapper,
                model=model,
                messages=oppose_first_messages,
                max_tokens=256,
                temperature=1.0,
                top_p=0.95,
                num_outputs=3
            ))
            futures.append(executor.submit(
                create_chat_completion_wrapper,
                model=model,
                messages=support_first_messages,
                max_tokens=256,
                temperature=1.0,
                top_p=0.95,
                num_outputs=3
            ))

            oppose_first_response, support_first_response = [f.result() for f in futures]

        # oppose_first_judge_ans = [r.message.content for r in oppose_first_response.choices]
        oppose_first_judge_ans = [r for r in oppose_first_response]
        oppose_first_judge_ans_str = '\n'.join(oppose_first_judge_ans)
        logger.debug(f"Oppose first judge answers:\n{oppose_first_judge_ans_str}")

        # support_first_judge_ans = [r.message.content for r in support_first_response.choices]
        support_first_judge_ans = [r for r in support_first_response]
        support_first_judge_ans_str = '\n'.join(support_first_judge_ans)
        logger.debug(f"Support first judge answers:\n{support_first_judge_ans_str}")
        return oppose_first_judge_ans + support_first_judge_ans

    @time_it
    def evaluate_success(
        self,
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        screenshots_text: Optional[list[str]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        """Compute the value of a state using the value function.

        TODO: screenshots should be ALL screenshots involved in the current task, not just the last few starting from the root node
        """
        assert screenshots_text is not None, "Need to provide text for the (last) screenshots (though it is not used yet)"
        cache_key = CoTwDebateValueFunction._encode_eval_success_input(
            screenshots=screenshots,
            actions=actions,
            intent=intent,
            intent_images=intent_images or []
        )
        if cache_key in CoTwDebateValueFunction._debate_cache:
            data_dict = CoTwDebateValueFunction._debate_cache[cache_key]
            return data_dict['v']
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            futures.append(executor.submit(
                self.generate_supporting_opinions,
                copy.deepcopy(screenshots),  # histories
                copy.deepcopy(screenshots_text),  # histories
                copy.deepcopy(actions),  # histories
                current_url,
                last_reasoning,
                intent,
                intent_images,
                models[0]
            ))
            futures.append(executor.submit(
                self.generate_opposing_opinions,
                copy.deepcopy(screenshots),  # histories
                copy.deepcopy(screenshots_text),  # histories
                copy.deepcopy(actions),  # histories
                current_url,
                last_reasoning,
                intent,
                intent_images,
                models[0]
            ))

            supporting_opinions, opposing_opinions = [f.result() for f in futures]
            logger.info(f"Value function generated supporing opinions:\n{supporting_opinions}")
            logger.info(f"Value function generated opposing opinions:\n{opposing_opinions}")
        
        final_decisions = self.generate_final_decisions(
            screenshots=screenshots,  # will be copied in the function
            screenshots_text=screenshots_text,
            actions=actions,  # will be copied in the function
            last_state_url=current_url,
            last_reasoning=last_reasoning,
            intent=intent,
            intent_images=intent_images,
            opposing_opinions=opposing_opinions,
            supporting_opinions=supporting_opinions,
            model=models[0]
        )

        ### parse the responses to scores
        all_scores = []
        for r_idx, resp_text in enumerate(final_decisions):
            logger.debug(f"Final decision {r_idx}: {resp_text}")
            try:
                pred = re.search(r'.*STATUS CODE: (\w).*', resp_text).group(1)
                if 'A' in pred:
                    score = 1.0
                elif 'B' in pred:
                    score = 0.9
                elif 'C' in pred:
                    score = 0.5
                elif 'D' in pred:
                    score = 0.2
                else:
                    score = 0.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0
            
            all_scores.append(score)
        score = np.mean(all_scores)
        logger.debug(f"Final score from vfunc {models}: {score} from {all_scores}")
        
        ### save this to cache
        if cache_key not in CoTwDebateValueFunction._debate_cache:
            CoTwDebateValueFunction._debate_cache[cache_key] = {
                'v': score,
                'supporting_reasons': supporting_opinions,
                'opposing_reasons': opposing_opinions,
                'final_decisions': final_decisions
            }
        return score


class Always0p5ValueFunction(ValueFunction):
    @time_it
    @staticmethod
    def evaluate_success(
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        last_reasoning: str,
        intent: str,
        models: list[str],
        init_screenshot: Optional[Image.Image] = None,
        intent_images: Optional[list[Image.Image]] = None,
        n: int = 20, top_p: float = 1.0, should_log: bool = False
    ) -> float:
        logger.debug("Always returning 0.5")
        return 0.5



# legacy code
@time_it
def evaluate_success(screenshots: list[Image.Image], actions: list[str], current_url: str, last_reasoning: str,
                     intent: str, models: list[str], init_screenshot: Optional[Image.Image] = None, intent_images: Optional[list[Image.Image]] = None,
                     n: int = 20, top_p: float = 1.0, should_log: bool = False) -> float:
    """Compute the value of a state using the value function.

    Args:
        state (str): The state to compute the value of.
        action (list[str]): The action to take in the state.
        intent (str): The intent to compute the value of.
        intent_images (list[Image.Image], optional): The images corresponding to the intent. Defaults to None.
        file_prefix (str, optional): The prefix to use for the file name. Defaults to ''.
    Returns:
        float: The value of the state.
    """
    last_actions_str = '\n'.join(actions[:-1])
    last_response = actions[-1]
    if intent_images is None:
        content = []
        for screenshot in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "detail": "high"
                },
            })

        content.append({
            "type": "text",
            "text": f"""User Intent: {intent}
Action History: {last_actions_str}
Bot response to the user: {last_response}
Current URL: {current_url}
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
"""
        })

    else:
        content = []

        for img in intent_images:
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(img)
                    },
                }
            ])
        content.append({
            "type": "text",
            "text": f"\nUser Intent: {intent}\n"
        })

        for screenshot in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "detail": "high"
                },
            })

        content.append({
            "type": "text",
            "text": f"""
Action History: {last_actions_str}
Bot response to the user: {last_response}
Current URL: {current_url}
The images corresponding to the user intent are shown in the FIRST {len(intent_images)} images (before the User Intent).
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the LAST {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
"""
        })

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]

    all_responses = []
    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=256,
            top_p=top_p,
            n=n // len(models)
        )

        token_stats = {
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'num_requests': 1
        }
        if VALUE_FUNCTION_PROVIDER == "azure":
            model_log_name = f"azure_{model}"
        else:
            model_log_name = model
        update_token_usage(
            model_name=model_log_name,
            token_stats=token_stats,
            token_usage_tracker=TOKEN_USAGE
        )

        all_responses.extend(response.choices)

    if should_log:
        print('=' * 30)
        print("Value function input:", content[-1])
    all_scores = []
    for r_idx, r in enumerate(all_responses):
        if should_log:
            print(f"Output {r_idx}: {r.message.content}")
        try:
            pred = re.search(r'Status: "?(.+)"?', r.message.content).group(1)
            if 'success' in pred.lower():
                score = 1.0
            else:
                # Check if it's on the path to success
                on_path = re.search(r'On the right track to success: "?(.+)"?', r.message.content).group(1)
                if 'yes' in on_path.lower():
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0
        
        all_scores.append(score)
    
    score = np.mean(all_scores)
    logger.debug(f"Final score from vfunc {models}: {score} from {all_scores}")
    if should_log:
        print(f"Final score: {score}")
        print('=' * 30)
    return score