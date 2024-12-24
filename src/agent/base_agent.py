import copy
import logging
from typing import Any, Optional
from PIL import Image
from beartype import beartype

from agent.prompts import PromptConstructor
from browser_env import ActionParsingError, Trajectory
from src.envs.types import StateInfo
from src.envs.actions import (
    create_id_based_action,
    create_none_action,
    create_playwright_action
)
from src.logging import atime_it
from src.envs.actions import Action
from src.llms import lm_config
from src.llms.utils import call_llm, is_vlm


logger = logging.getLogger('logger')


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any, **kwargs
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class FastAgent(Agent):
    """Base class for the agent in a fast environment"""

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any, **kwargs
    ) -> Action:
        raise ValueError("Please use anext_action for fast environment")

    async def anext_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any, **kwargs
    ) -> Action:
        """Predict the next action given an async environment (e.g., inside meta_data)"""
        raise NotImplementedError

    def on_task_start(self, task_info: dict, **kwargs) -> None:
        """Called when the task start. Used for reinforced MCTS"""
        return

    def on_task_end(self, trajectory: Trajectory, score: float, task_info: dict, meta_data: Any, **kwargs) -> None:
        """Called when the task ends. Used for reinforced MCTS"""
        return


class PromptAgent(FastAgent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if is_vlm(self.lm_config) and prompt_constructor.is_multimodal:
            logging.info("Using multimodal input in prompt.")
            self.multimodal_inputs = True
        else:
            logging.info("Model is not multimodal.")
            self.multimodal_inputs = False
        return

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag
        return

    @atime_it
    async def anext_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        additional_inputs: dict[str, Any]
    ) -> Action:
        output_response = False
        task_info = additional_inputs["task_info"]
        images: Optional[list[Image.Image]] = task_info["images"]

        state_info: StateInfo = trajectory[-1]
        observation_metadata = state_info['info']['observation_metadata']

        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
                logger.info(f"Updated intent to: {intent}")
            elif not self.multimodal_inputs:
                logger.warn(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt, num_outputs=1)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                logger.info(f'Agent: {response}')
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action['metadata']['obs_metadata'] = copy.deepcopy(observation_metadata)

                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        ### check if element id is found on the page
        logger.debug(f"LLM Top actions:")
        logger.debug(f" {action['raw_prediction']}")
        # check if is valid
        obs_text = state_info["observation"]["text"]
        action_element_id = action['element_id']
        if action_element_id != "":
            if action_element_id in obs_text:
                logger.debug(f"  [{action_element_id}] is found on the page!")
            else:
                logger.debug(f"  [{action_element_id}] is NOT found on the page!")
        return action

    def reset(self, test_config_file: str) -> None:
        return