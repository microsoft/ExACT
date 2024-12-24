import logging
import copy
from pathlib import Path
from typing import Any, TypedDict
from PIL import Image

from browser_env import Trajectory
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from llms import lm_config
from src.llms.tokenizer import Tokenizer
from src.llms.utils import _add_modality_key_for_sglang_messages
from src.prompts.utils import display_multimodal_openai_messages
from src.envs.actions import Action
from llms.utils import APIInput
from agent.prompts import CoTPromptConstructor


logger = logging.getLogger("logger")


class MCoTPolicyPConstructor_OLD(CoTPromptConstructor):
    """Multimodal CoT based prompt constructor. Migrated (with minor modifications) from https://github.com/kohjingyu/search-agents
    """
    is_multimodal = True

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]
        return

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        logger.info(f"constructed prompt with len={len(prompt)}")
        return prompt

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if self.lm_config.provider in ["openai", "sglang", "azure"]:
            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": x},
                                {
                                    "type": "text",
                                    "text": "IMAGES: (1) current page screenshot",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_to_b64(example_img)
                                    },
                                },
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i+2}) input image {image_i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})
                if self.lm_config.provider == "sglang":
                    message = _add_modality_key_for_sglang_messages(message)
                return message
        elif "google" in self.lm_config.provider:
            if self.lm_config.mode == "completion":
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(example_img),
                        ]
                    )
                    message.append(f"Action: {y}")
                message.append("Now make prediction given the observation")
                message.append(f"Observation\n:{current}\n")
                message.extend(
                    [
                        "IMAGES:",
                        "(1) current page screenshot:",
                        pil_to_vertex(page_screenshot_img),
                    ]
                )
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i+2}) input image {image_i+1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Action:")
                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )


class MCoTPolicyPConstructor(MCoTPolicyPConstructor_OLD):
    """default multimodal CoT prompt used by non-reflective agents
    """
    is_multimodal = True

    def _is_long_history(self, all_prev_state_actions: Trajectory) -> bool:
        return len(all_prev_state_actions) >= 5 # (s,a,s,a,s)

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        intent: str,
        intent_image: list[Image.Image],
        all_prev_state_actions: Trajectory,
        all_prev_action_strs: list[str],
        context_specific_instruction: str,
    ):
        """Return the require format for an API"""
        assert self.lm_config.provider in ["openai", "sglang", "azure"], f"Provider {self.lm_config.provider} not implemented"
        assert self.lm_config.mode == "chat", f"Mode {self.lm_config.mode} not implemented"
        assert len(all_prev_state_actions) % 2 == 1, f"all_prev_state_actions should be in (s,a, ...,s,a,s), but got {len(all_prev_state_actions)=}"
        assert context_specific_instruction == '', "context_specific_instruction should be empty in non-reflective agents"

        message: list[dict[str, str]] | str | list[str | Image.Image]

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]

        ### 1. add few shot examples
        if self._is_long_history(all_prev_state_actions):
            examples = []  # no need
        for (x, y, z) in examples:
            example_img = Image.open(z)
            message.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": x},
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": pil_to_b64(example_img)
                            },
                        },
                    ],
                }
            )
            message.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": y}],
                }
            )
        ### 2. add trajectory history
        ### 2.1 format init state
        history_prompt = []
        template = self.instruction["template"]
        init_template = self.instruction["init_template"]

        init_s = all_prev_state_actions[0]
        init_state_img = Image.fromarray(init_s["observation"]["image"])
        init_state_text = init_s["observation"]["text"]
        init_url = init_s["info"]["page"].url
        init_prev_action_str = all_prev_action_strs[0]
        hist_init = init_template.format(
            objective=intent,
            url=self.map_url_to_real(init_url),
            observation=init_state_text,
            previous_action=init_prev_action_str,
        )
        hist_init = (
            f"!IMPORTANT! Below is the task you need to solve. Please read the user's intent and input images (if any) carefully.\n"
            f"{hist_init}"
        )
        history_prompt.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": hist_init
                },
            ]
        })
        for image_i, image in enumerate(intent_image):
            history_prompt[0]["content"].extend(
                [
                    {
                        "type": "text",
                        "text": f"({image_i+1}) input image {image_i+1}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image)},
                    },
                ]
            )
        history_prompt[0]["content"].extend([
            {
                "type": "text",
                "text": "IMAGES: (1) current page screenshot",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(init_state_img)},
            }
        ])
        ### 2.2 format history actions
        past_action_idx = 0
        # skip first state. This should end with the last/current state
        for s_or_a in all_prev_state_actions[1:]:
            if isinstance(s_or_a, Action):
                history_prompt.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{s_or_a.raw_prediction}"
                            },
                        ]
                    }
                )
                past_action_idx += 1
            else:
                state_img = Image.fromarray(s_or_a["observation"]["image"])
                state_text = s_or_a["observation"]["text"]
                page = s_or_a["info"]["page"]
                url = page.url
                previous_action_str = all_prev_action_strs[past_action_idx]
                hist_current = template.format(
                    url=self.map_url_to_real(url),
                    observation=state_text,
                    previous_action=previous_action_str,
                )
                history_prompt.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": hist_current
                            },
                            {
                                "type": "text",
                                "text": "IMAGES: (1) current page screenshot",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(state_img)},
                            },
                        ]
                    }
                )
        ### 2.3 inject reflection into current state
        additional_instructions = None
        if len(history_prompt) > 1:
            additional_instructions = (
                "\nNOTE: Remember that you should consider user's intent and previous histories "
                "to better plan the next action."
            )
        if additional_instructions is not None:
            history_prompt[-1]["content"][0]["text"] += additional_instructions
        
        #### done
        message.extend(history_prompt)
        if self.lm_config.provider == "sglang":
            message = _add_modality_key_for_sglang_messages(message)
        return message

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
    ):
        # simply format prompt using the FULL trajectory
        intro = self.instruction["intro"]
        intro_wo_icl = self.instruction["intro_wo_icl"]
        examples = self.instruction["examples"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        images = images or []

        
        if self._is_long_history(trajectory):
            intro = intro_wo_icl
        
        ### format all past actions
        none_padded_action_history_str = copy.deepcopy(meta_data["action_history"])
        if none_padded_action_history_str[0].lower() != "none":
            none_padded_action_history_str.insert(0, "None")

        prompt = self.get_lm_api_input(
            intro, examples,
            intent=intent,
            intent_image=images,
            all_prev_state_actions=trajectory,
            all_prev_action_strs=none_padded_action_history_str,
            context_specific_instruction="",  # ablate with reflections used in rmcts
        )
        logger.info(f"constructed prompt with len={len(prompt)}")
        logger.debug(f"constructed full prompt:\n{display_multimodal_openai_messages(prompt)}")
        return prompt


class CoTPolicyPConstructor(CoTPromptConstructor):
    """same as MCoTPolicyPConstructor but without images"""
    is_multimodal = False

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]
        return

    def _is_long_history(self, all_prev_state_actions: Trajectory) -> bool:
        return len(all_prev_state_actions) >= 5 # (s,a,s,a,s)
    
    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        intent: str,
        all_prev_state_actions: Trajectory,
        all_prev_action_strs: list[str],
    ):
        """Return the require format for an API"""
        assert self.lm_config.provider in ["openai", "sglang", "azure"], f"Provider {self.lm_config.provider} not implemented"
        assert self.lm_config.mode == "chat", f"Mode {self.lm_config.mode} not implemented"
        assert len(all_prev_state_actions) % 2 == 1, f"all_prev_state_actions should be in (s,a, ...,s,a,s), but got {len(all_prev_state_actions)=}"
        message: list[dict[str, str]] | str | list[str | Image.Image]

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]

        ### 1. add few shot examples
        if self._is_long_history(all_prev_state_actions):
            examples = []  # no need
        
        for (x, y) in examples:
            message.append(
                {
                    "role": "user",
                    "content": x,
                }
            )
            message.append(
                {
                    "role": "assistant",
                    "content": y,
                }
            )
        ### 2. add trajectory history
        ### 2.1 format init state
        history_prompt = []
        template = self.instruction["template"]
        init_template = self.instruction["init_template"]

        init_s = all_prev_state_actions[0]
        init_state_text = init_s["observation"]["text"]
        init_url = init_s["info"]["page"].url
        init_prev_action_str = all_prev_action_strs[0]
        hist_init = init_template.format(
            objective=intent,
            url=self.map_url_to_real(init_url),
            observation=init_state_text,
            previous_action=init_prev_action_str,
        )
        hist_init = (
            f"!IMPORTANT! Below is the task you need to solve. Please read the user's intent and input images (if any) carefully.\n"
            f"{hist_init}"
        )
        history_prompt.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": hist_init
                },
            ]
        })
        ### 2.2 format history actions
        past_action_idx = 0
        # skip first state. This should end with the last/current state
        for s_or_a in all_prev_state_actions[1:]:
            if isinstance(s_or_a, Action):
                history_prompt.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{s_or_a.raw_prediction}"
                            },
                        ]
                    }
                )
                past_action_idx += 1
            else:
                state_text = s_or_a["observation"]["text"]
                page = s_or_a["info"]["page"]
                url = page.url
                previous_action_str = all_prev_action_strs[past_action_idx]
                hist_current = template.format(
                    url=self.map_url_to_real(url),
                    observation=state_text,
                    previous_action=previous_action_str,
                )
                history_prompt.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": hist_current
                            }
                        ]
                    }
                )
        #### done
        message.extend(history_prompt)
        return message

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,  # intent contains both the ori intent and captioned intent image
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        intro_wo_icl = self.instruction["intro_wo_icl"]
        examples = self.instruction["examples"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        
        if self._is_long_history(trajectory):
            intro = intro_wo_icl
        
        ### format all past actions
        none_padded_action_history_str = copy.deepcopy(meta_data["action_history"])
        if none_padded_action_history_str[0].lower() != "none":
            none_padded_action_history_str.insert(0, "None")

        prompt = self.get_lm_api_input(
            intro, examples,
            intent=intent,
            all_prev_state_actions=trajectory,
            all_prev_action_strs=none_padded_action_history_str,
        )
        logger.info(f"constructed prompt with len={len(prompt)}")
        logger.debug(f"constructed full prompt:\n{display_multimodal_openai_messages(prompt)}")
        return prompt


class ExploratoryCoTPolicyPConstructor(CoTPolicyPConstructor):
    """+ prmopts for more exploration and backtracking"""
    is_multimodal = False

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        intent: str,
        all_prev_state_actions: Trajectory,
        all_prev_action_strs: list[str],
    ):
        """Return the require format for an API"""
        assert self.lm_config.provider in ["openai", "sglang", "azure"], f"Provider {self.lm_config.provider} not implemented"
        assert self.lm_config.mode == "chat", f"Mode {self.lm_config.mode} not implemented"
        assert len(all_prev_state_actions) % 2 == 1, f"all_prev_state_actions should be in (s,a, ...,s,a,s), but got {len(all_prev_state_actions)=}"
        message: list[dict[str, str]] | str | list[str | Image.Image]

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": intro}],
            }
        ]

        ### 1. add few shot examples
        if self._is_long_history(all_prev_state_actions):
            examples = []  # no need
        
        assert len(examples) == 0, "examples should be empty anyway"
        ### 2. add trajectory history
        ### 2.1 format init state
        history_prompt = []
        template = self.instruction["template"]
        init_template = self.instruction["init_template"]

        init_s = all_prev_state_actions[0]
        init_state_text = init_s["observation"]["text"]
        init_url = init_s["info"]["page"].url
        init_prev_action_str = all_prev_action_strs[0]
        hist_init = init_template.format(
            objective=intent,
            url=self.map_url_to_real(init_url),
            observation=init_state_text,
            previous_action=init_prev_action_str,
        )
        hist_init = (
            f"!IMPORTANT! Below is the task you need to solve. Please read the user's intent and input images (if any) carefully.\n"
            f"{hist_init}"
        )
        history_prompt.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": hist_init
                },
            ]
        })
        ### 2.2 format history actions
        past_action_idx = 0
        # skip first state. This should end with the last/current state
        for s_or_a in all_prev_state_actions[1:]:
            if isinstance(s_or_a, Action):
                history_prompt.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{s_or_a.raw_prediction}"
                            },
                        ]
                    }
                )
                past_action_idx += 1
            else:
                state_text = s_or_a["observation"]["text"]
                page = s_or_a["info"]["page"]
                url = page.url
                previous_action_str = all_prev_action_strs[past_action_idx]
                hist_current = template.format(
                    url=self.map_url_to_real(url),
                    observation=state_text,
                    previous_action=previous_action_str,
                )
                history_prompt.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": hist_current
                            }
                        ]
                    }
                )
        note = (
            "Note: do not be afraid to explore different actions, re-evaluate current progress, and take a step (```go_back```) or multiple steps back (```goto [url]```).\n"
            "Making slow and steady progress is BETTER than rushing and making simple mistakes."
        )
        history_prompt[-1]["content"].append({
            "type": "text",
            "text": note
        })
        #### done
        message.extend(history_prompt)
        return message