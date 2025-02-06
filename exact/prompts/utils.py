from mm_agents.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE, SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE, SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG
)
from exact.prompts.reasoning_policy_prompt import (
    SYS_PROMPT_REASONING_IN_SCREENSHOT_OUT_CODE, SYS_PROMPT_REASONING_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_REASONING_IN_A11Y_OUT_CODE, SYS_PROMPT_REASONING_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_REASONING_IN_BOTH_OUT_CODE, SYS_PROMPT_REASONING_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_REASONING_IN_SOM_OUT_TAG
)
from exact.prompts.explorative_policy_prompt import (
    SYS_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_CODE, SYS_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_EXPLORATIVE_IN_A11Y_OUT_CODE, SYS_PROMPT_EXPLORATIVE_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_EXPLORATIVE_IN_BOTH_OUT_CODE, SYS_PROMPT_EXPLORATIVE_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_EXPLORATIVE_IN_SOM_OUT_TAG
)


def configure_system_prompt(observation_type: str, action_space: str, prompt_prefix: str) -> str:
    if observation_type == "screenshot":
        if action_space == "computer_13":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_SCREENSHOT_OUT_ACTION
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_ACTION
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        elif action_space == "pyautogui":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_SCREENSHOT_OUT_CODE
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_SCREENSHOT_OUT_CODE
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        else:
            raise ValueError("Invalid action space: " + action_space)
    elif observation_type == "a11y_tree":
        if action_space == "computer_13":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_A11Y_OUT_ACTION
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_A11Y_OUT_ACTION
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_A11Y_OUT_ACTION
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        elif action_space == "pyautogui":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_A11Y_OUT_CODE
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_A11Y_OUT_CODE
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_A11Y_OUT_CODE
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        else:
            raise ValueError("Invalid action space: " + action_space)
    elif observation_type == "screenshot_a11y_tree":
        if action_space == "computer_13":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_BOTH_OUT_ACTION
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_BOTH_OUT_ACTION
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_BOTH_OUT_ACTION
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        elif action_space == "pyautogui":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_BOTH_OUT_CODE
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_BOTH_OUT_CODE
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_BOTH_OUT_CODE
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        else:
            raise ValueError("Invalid action space: " + action_space)
    elif observation_type == "som":
        if action_space == "computer_13":
            raise ValueError("Invalid action space: " + action_space)
        elif action_space == "pyautogui":
            if prompt_prefix == "":
                return SYS_PROMPT_IN_SOM_OUT_TAG
            elif prompt_prefix == "reasoning_v1":
                return SYS_PROMPT_REASONING_IN_SOM_OUT_TAG
            elif prompt_prefix == "explorative_v1":
                return SYS_PROMPT_EXPLORATIVE_IN_SOM_OUT_TAG
            else:
                raise ValueError("Invalid prompt prefix: " + prompt_prefix)
        else:
            raise ValueError("Invalid action space: " + action_space)
    else:
        raise ValueError("Invalid experiment type: " + observation_type)