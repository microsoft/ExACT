from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from exact.prompts.value_function import (
    VFUNC_COT_PYAUTOGUI_INTRO,
    VFUNC_COT_FINAL_PROMPT,
    VFUNC_SINGLE_DEBATE_FINAL_PROMPT,
    VFUNC_TRAINED_FINAL_PROMPT
)
from exact.llms.lm_config import LMConfig
from exact.llms.tokenizer import Tokenizer
from exact.llms.utils import (
    configure_llm_client,
    call_llm, call_classification_llm, 
    _truncate_prompt_to_max_tokens, _force_truncate_prompt_to_max_tokens,
    _flatten_chat_msg_turns
)
from exact.logging import time_it
from exact.args import ValueArgs
import logging
import os
import re
import copy
import numpy as np


logger = logging.getLogger("src.agentic")


class ValueFunction(ABC):
    name: str = "v_func"

    @abstractmethod
    def predict(
        self,
        instruction: str,
        obss: list[dict],
        actions: list[str],
        thoughts: list[str],
    ) -> float:
        """returns a numeric estimate of future success, given a trajectory of observations and actions (and its corresponding LLM response)
        NOTE: all obss should be either raw or processed, not mixed

        Args:
            instruction (str): _description_
            obss (list[dict]): _description_
            actions (list[str]): _description_
            thoughts (list[str]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float: _description_
        """
        raise NotImplementedError


@dataclass
class CoTValueArgs(ValueArgs):
    value_func: str = field(
        default="cot_value_func",
        metadata={"help": "The name of the value function."}
    )
    vf_model: str = field(
        default="gpt-4o",
        metadata={"help": "The model to use for the value function."}
    )
    vf_serve_model_name: str = field(
        default="",
        metadata={"help": "The model name used to query the LLM API. Default uses vf_model unless "}
    )
    vf_max_trajectory_length: int = field(
        default=3,
        metadata={"help": "The maximum length of past (obs, action) pairs to be included in the prompt."}
    )
    vf_n: int = field(
        default=20,
        metadata={"help": "The number of completions to average and obtain value estimate."}
    )
    vf_temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature to use for sampling completions."}
    )
    vf_top_p: float = field(
        default=0.9,
        metadata={"help": "The top_p to use for sampling completions."}
    )
    vf_max_tokens: int = field(
        default=256,
        metadata={"help": "The maximum number of tokens to generate in completions."}
    )


class CoTValueFunction(ValueFunction):
    name: str = "cot_value_func"
    def __init__(
        self,
        args: CoTValueArgs,
        observation_type: str,
        action_space: str,
    ):
        self.args = args
        self.observation_type = observation_type
        self.action_space = action_space

        self.lm_config, self.llm_client = self._configure_client()
        self._final_prompt = VFUNC_COT_FINAL_PROMPT
        return

    def _configure_client(self):
        model_name = self.args.vf_serve_model_name or self.args.vf_model
        lm_config = LMConfig(
            provider=self.args.vf_model_api_provider,
            model=model_name,
            mode="chat",
            tokenizer_cls=Tokenizer(
                provider=self.args.vf_model_api_provider,
                model_name=self.args.vf_model,
                max_context_length=self.args.vf_max_context_length,
            ),
            api_base=os.environ.get("VALUE_LLM_API_BASE", "http://127.0.0.1:30000/v1"),
            api_key=os.environ.get("VALUE_LLM_API_KEY", "empty"),
            api_version=os.environ.get("VALUE_LLM_API_VERSION", ""),
            api_token_provider_base=os.environ.get("VALUE_LLM_TOKEN_PROVIDER_BASE", ""),
            gen_config={
                'temperature': self.args.vf_temperature,
                'top_p': self.args.vf_top_p,
                'max_tokens': self.args.vf_max_tokens,
            }
        )
        client = configure_llm_client(lm_config)
        return lm_config, client

    def _get_action_str(self, action_list: list):
        if self.action_space == "pyautogui":
            formatted_actions = [f"```python\n{action}\n```" for action in action_list]
            action_str = "\nthen execute\n".join(formatted_actions)
        elif self.action_space == "computer_13":
            raise NotImplementedError("computer_13 action space not supported yet")
        else:
            raise ValueError(f"Invalid action_space: {self.action_space}")
        return action_str

    def _construct_prompt(
        self,
        instruction: str,
        obss: list[dict],
        actions: list[str],
        thoughts: list[str],
    ):
        # Prepare the payload for the API call
        messages = []

        task_str = f"You are asked to complete the following task: {instruction}"
        if self.action_space == "pyautogui":
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": VFUNC_COT_PYAUTOGUI_INTRO + "\n" + task_str
                    },
                ]
            })
        else:
            raise ValueError(f"Invalid action_space: {self.action_space}")

        # Append trajectory
        assert len(obss) == len(actions) + 1, f"We should have one more observation, got {len(obss)} and {len(actions)}"
        assert len(actions) == len(thoughts), f"The number of actions and thoughts should be the same, got {len(actions)} and {len(thoughts)}"

        #### context history
        for previous_obs, previous_action, previous_thought in zip(obss, actions, thoughts):
            if self.observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{task_str}\nCurrent screenshot and info from accessibility tree:\n{_linearized_accessibility_tree}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "som":
                _screenshot = previous_obs["screenshot"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{task_str}\nCurrent tagged screenshot:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "screenshot":
                _screenshot = previous_obs["screenshot"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{task_str}\nCurrent screenshot:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "a11y_tree":
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{task_str}\nCurrent info from accessibility tree:\n{_linearized_accessibility_tree}"
                        },
                    ]
                })
            else:
                raise ValueError("Invalid observation_type type: " + self.observation_type)
            
            ### use action instead of response to remove noise due to LM's misleading reasonings
            action_str = self._get_action_str(previous_action)
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        # "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                        "text": action_str.strip() if len(previous_action) > 0 else "No valid action"
                    },
                ]
            })

        #### current observation
        obs = obss[-1]
        if self.observation_type == "screenshot":
            base64_image = obs["screenshot"]
            linearized_accessibility_tree = obs["accessibility_tree"]

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{task_str}\nCurrent screenshot:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": self._final_prompt
                    }
                ]
            })
        elif self.observation_type == "screenshot_a11y_tree":
            base64_image = obs["screenshot"]
            linearized_accessibility_tree = obs["accessibility_tree"]

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{task_str}\nCurrent screenshot and info from accessibility tree:\n{_linearized_accessibility_tree}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": self._final_prompt
                    }
                ]
            })
        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = obs["accessibility_tree"]

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{task_str}\nCurrent info from accessibility tree:\n{linearized_accessibility_tree}"
                    },
                    {
                        "type": "text",
                        "text": self._final_prompt
                    }
                ]
            })
        elif self.observation_type == "som":
            base64_image = obs["screenshot"]
            linearized_accessibility_tree = obs["accessibility_tree"]

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{task_str}\nCurrent tagged screenshot and info from accessibility tree:\n{linearized_accessibility_tree}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": self._final_prompt
                    }
                ]
            })
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)
        return messages

    @time_it
    def predict(
        self,
        instruction: str,
        obss: list[dict],
        actions: list[str],
        thoughts: list[str],
    ) -> float:
        _obss = copy.deepcopy(obss)
        _actions = copy.deepcopy(actions)
        _thoughts = copy.deepcopy(thoughts)

        last_obs = _obss.pop(-1)
        _obss = _obss[-self.args.vf_max_trajectory_length:] + [last_obs]
        _actions = _actions[-self.args.vf_max_trajectory_length:]
        _thoughts = _thoughts[-self.args.vf_max_trajectory_length:]

        messages = self._construct_prompt(
            instruction=instruction,
            obss=_obss,
            actions=_actions,
            thoughts=_thoughts,
        )

        all_responses = self.call_llm(messages)
        
        ### parse the responses to scores
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            logger.debug(f"Response {r_idx}: {r}")
            try:
                pred = re.search(r'.*EVALUATION: (\w).*', r).group(1)
                if 'A' in pred:
                    score = 1.0
                elif 'B' in pred:
                    score = 0.5
                elif 'C' in pred:
                    score = 0.2
                elif 'D' in pred:
                    score = -0.2
                else:
                    score = -1.0
                all_scores.append(score)
            except Exception as e:
                print(f"Error parsing response: {e}")
        
        score = -1.0 if len(all_scores) == 0 else np.mean(all_scores)
        logger.info(f"Final score from vfunc {self.args.vf_model}: {score} from {all_scores}")
        return score

    @time_it
    def call_llm(self, messages):
        ## 1. truncate the prompt instead of simple left truncate if we let tokenizer do it
        if self.args.vf_force_context_truncation:
            messages = _force_truncate_prompt_to_max_tokens(
                prompt=messages,
                tokenizer=self.lm_config.tokenizer_cls,
            )
        else:
            messages = _truncate_prompt_to_max_tokens(
                prompt=messages,
                tokenizer=self.lm_config.tokenizer_cls
            )
        ## 2. flatten the chat messages (useful if the models are also TRAINED this way)
        if self.args.vf_flatten_chat_msg:
            engine = self.args.vf_flatten_engine
            messages = _flatten_chat_msg_turns(messages, engine=engine)
        
        ## 3. call the model
        responses = call_llm(
            self.llm_client,
            self.lm_config,
            prompt=messages,
            num_outputs=self.args.vf_n
        )
        return responses



class SingleDebateValueFunction(CoTValueFunction):
    name: str = "sad_value_func"
    def __init__(
        self,
        args: CoTValueArgs,
        observation_type: str,
        action_space: str,
    ):
        super().__init__(args, observation_type, action_space)
        self._final_prompt = VFUNC_SINGLE_DEBATE_FINAL_PROMPT
        return
    
    
@dataclass
class TrainedValueArgs(CoTValueArgs):
    value_func: str = field(
        default="ft_value_func",
        metadata={"help": "The name of the value function."}
    )
    vf_clip: bool = field(
        default=False,
        metadata={"help": "Whether to clip the value function output to [-1, 1]."}
    )


class TrainedValueFunction(CoTValueFunction):
    name: str = "ft_value_func"
    
    def __init__(
        self,
        args: TrainedValueArgs,
        observation_type: str,
        action_space: str,
    ):
        super().__init__(args, observation_type, action_space)
        self.args = args
        self._final_prompt = VFUNC_TRAINED_FINAL_PROMPT
        return
    
    @time_it
    def predict(
        self,
        instruction: str,
        obss: list[dict],
        actions: list[str],
        thoughts: list[str],
    ) -> float:
        _obss = copy.deepcopy(obss)
        _actions = copy.deepcopy(actions)
        _thoughts = copy.deepcopy(thoughts)

        last_obs = _obss.pop(-1)
        _obss = _obss[-self.args.vf_max_trajectory_length:] + [last_obs]
        _actions = _actions[-self.args.vf_max_trajectory_length:]
        _thoughts = _thoughts[-self.args.vf_max_trajectory_length:]

        messages = self._construct_prompt(
            instruction=instruction,
            obss=_obss,
            actions=_actions,
            thoughts=_thoughts,
        )
        all_responses = self.call_llm(messages)
        
        ### parse the responses to scores
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            logger.info(f"Response {r_idx}: {r}")
            # use prob of class 1, so the second index
            try:
                score = r["data"]
                if self.args.vf_clip:
                    score = np.clip(score, -1.0, 1.0)
                all_scores.append(score)
            except Exception as e:
                logger.debug(f"Error parsing response: {e}")
        score = -1.0 if len(all_scores) == 0 else np.mean(all_scores)
        logger.info(f"Final score from vfunc {self.args.vf_model}: {score} from {all_scores}")
        return score

    @time_it
    def call_llm(self, messages):
        messages = _truncate_prompt_to_max_tokens(
            prompt=messages,
            tokenizer=self.lm_config.tokenizer_cls
        )
        responses = call_classification_llm(
            self.llm_client,
            self.lm_config,
            prompt=messages,
            num_outputs=self.args.vf_n
        )
        if not isinstance(responses, list):
            responses = [responses]
        return responses
