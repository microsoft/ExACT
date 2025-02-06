import copy
from dataclasses import dataclass, field
from exact.prompts.reasoning_policy_prompt import (
    USR_PROMPT_REASONING_IN_A11Y_OUT_CODE,
    USR_PROMPT_REASONING_IN_BOTH_OUT_CODE,
    USR_PROMPT_REASONING_IN_SCREENSHOT_OUT_CODE,
)
from exact.prompts.explorative_policy_prompt import (
    USR_PROMPT_EXPLORATIVE_IN_A11Y_OUT_CODE,
    USR_PROMPT_EXPLORATIVE_IN_BOTH_OUT_CODE,
    USR_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_CODE,
)


@dataclass
class ReACTPolicyArgs:
    name: str = field(
        default="react",
        metadata={"help": "The policy to be used for the agent."}
    )
    max_trajectory_length: int = field(
        default=3,
        metadata={"help": "The maximum length of past (obs, action) pairs to be included in the prompt."}
    )
    user_prompt_prefix: str = field(
        default="",
        metadata={
            "help": "The user prompt to be added to the prompt.",
            "choices": ["", "reasoning_v1"]
        }
    )


class ReACTPolicy:
    def __init__(self, args: ReACTPolicyArgs, system_message, observation_type: str, action_space: str):
        self.args = args
        self.system_message = system_message
        self.observation_type = observation_type
        self.action_space = action_space
        return

    def get_messages(
        self,
        instruction: str,
        past_obs: list,
        past_actions: list,
        past_thoughts: list
    ):
        # TODO: the process obs logic should be handled by the agent, as it could be used by other modules as well
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        messages = []
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        # Append trajectory
        assert len(past_obs) == len(past_actions) + 1, f"there should be one more obs: {len(past_obs)} != {len(past_actions)} + 1"
        assert len(past_actions) == len(past_thoughts), f"num actions is not equal to num thoughts: {len(past_actions)} != {len(past_thoughts)}"

        past_obs = copy.deepcopy(past_obs)
        last_obs = past_obs.pop(-1)
        if len(past_obs) > self.args.max_trajectory_length:
            if self.args.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = past_obs[-self.args.max_trajectory_length:]
                _actions = past_actions[-self.args.max_trajectory_length:]
                _thoughts = past_thoughts[-self.args.max_trajectory_length:]
        else:
            _observations = past_obs
            _actions = past_actions
            _thoughts = past_thoughts

        #### context history
        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):
            if self.observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                _user_prompt = "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                if self.args.user_prompt_prefix == "reasoning_v1":
                    if self.action_space == "pyautogui":
                        _user_prompt = _user_prompt + "\n" + USR_PROMPT_REASONING_IN_BOTH_OUT_CODE
                    elif self.action_space == "computer_13":
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                    else:
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                elif self.args.user_prompt_prefix == "explorative_v1":
                    if self.action_space == "pyautogui":
                        _user_prompt = _user_prompt + "\n" + USR_PROMPT_EXPLORATIVE_IN_BOTH_OUT_CODE
                    elif self.action_space == "computer_13":
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                    else:
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _user_prompt
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
                
                _user_prompt = "Given the tagged screenshot as below. What's the next step that you will do to help with the task?"
                if self.args.user_prompt_prefix == "reasoning_v1":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _user_prompt
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
                _user_prompt = "Given the screenshot as below. What's the next step that you will do to help with the task?"
                if self.args.user_prompt_prefix == "reasoning_v1":
                    if self.action_space == "pyautogui":
                        _user_prompt = _user_prompt + "\n" + USR_PROMPT_REASONING_IN_SCREENSHOT_OUT_CODE
                    elif self.action_space == "computer_13":
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                    else:
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                elif self.args.user_prompt_prefix == "explorative_v1":
                    if self.action_space == "pyautogui":
                        _user_prompt = _user_prompt + "\n" + USR_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_CODE
                    elif self.action_space == "computer_13":
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                    else:
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _user_prompt
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
                _user_prompt = "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                
                if self.args.user_prompt_prefix == "reasoning_v1":
                    if self.action_space == "pyautogui":
                        _user_prompt = _user_prompt + "\n" + USR_PROMPT_REASONING_IN_A11Y_OUT_CODE
                    elif self.action_space == "computer_13":
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                    else:
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                elif self.args.user_prompt_prefix == "explorative_v1":
                    if self.action_space == "pyautogui":
                        _user_prompt = _user_prompt + "\n" + USR_PROMPT_EXPLORATIVE_IN_A11Y_OUT_CODE
                    elif self.action_space == "computer_13":
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                    else:
                        raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _user_prompt
                        }
                    ]
                })
            else:
                raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                    },
                ]
            })

        #### current observation
        obs_post = last_obs
        if self.observation_type == "screenshot":
            base64_image = obs_post["screenshot"]
            _user_prompt = "Given the screenshot as below. What's the next step that you will do to help with the task?"
            if self.args.user_prompt_prefix == "reasoning_v1":
                if self.action_space == "pyautogui":
                    _user_prompt = _user_prompt + "\n" + USR_PROMPT_REASONING_IN_SCREENSHOT_OUT_CODE
                elif self.action_space == "computer_13":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                else:
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
            elif self.args.user_prompt_prefix == "explorative_v1":
                if self.action_space == "pyautogui":
                    _user_prompt = _user_prompt + "\n" + USR_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_CODE
                elif self.action_space == "computer_13":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                else:
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        elif self.observation_type == "screenshot_a11y_tree":
            base64_image = obs_post["screenshot"]
            linearized_accessibility_tree = obs_post["accessibility_tree"]
            _user_prompt = "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
            
            if self.args.user_prompt_prefix == "reasoning_v1":
                if self.action_space == "pyautogui":
                    _user_prompt = _user_prompt + "\n" + USR_PROMPT_REASONING_IN_BOTH_OUT_CODE
                elif self.action_space == "computer_13":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                else:
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
            elif self.args.user_prompt_prefix == "explorative_v1":
                if self.action_space == "pyautogui":
                    _user_prompt = _user_prompt + "\n" + USR_PROMPT_EXPLORATIVE_IN_BOTH_OUT_CODE
                elif self.action_space == "computer_13":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                else:
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = obs_post["accessibility_tree"]
            _user_prompt = "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
            
            if self.args.user_prompt_prefix == "reasoning_v1":
                if self.action_space == "pyautogui":
                    _user_prompt = _user_prompt + "\n" + USR_PROMPT_REASONING_IN_A11Y_OUT_CODE
                elif self.action_space == "computer_13":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                else:
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
            elif self.args.user_prompt_prefix == "explorative_v1":
                if self.action_space == "pyautogui":
                    _user_prompt = _user_prompt + "\n" + USR_PROMPT_EXPLORATIVE_IN_A11Y_OUT_CODE
                elif self.action_space == "computer_13":
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
                else:
                    raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                ]
            })
        elif self.observation_type == "som":
            base64_image = obs_post["screenshot"]
            linearized_accessibility_tree = obs_post["accessibility_tree"]
            _user_prompt = "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
            
            if self.args.user_prompt_prefix == "reasoning_v1":
                raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")
            elif self.args.user_prompt_prefix == "explorative_v1":
                raise ValueError(f"Invalid action space {self.action_space=} for {self.args.user_prompt_prefix=}")

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)
        return messages